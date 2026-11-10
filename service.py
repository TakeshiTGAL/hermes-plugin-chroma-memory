"""Profile-scoped Chroma store. Secrets are read on every call."""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

try:
    from agent.secret_scope import UnscopedSecretError
except ImportError:  # hermes-free unit tests
    class UnscopedSecretError(RuntimeError):
        pass

try:
    from . import fetch
    from .embed import MiniLM
    from .safe_write import WriteRefused, directory_is_real, write_bytes
except ImportError:
    import fetch
    from embed import MiniLM
    from safe_write import WriteRefused, directory_is_real, write_bytes

logger = logging.getLogger("chroma-memory")

SECRET_NAMES = (
    "CHROMA_SERVER_URL",
    "CHROMA_API_KEY",
    "CHROMA_TENANT",
    "CHROMA_DATABASE",
    "CHROMA_COLLECTION",
    "CHROMA_PATH",
    "CHROMA_MODEL_DOWNLOAD",
)
_COLLECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]$")
_IPV4_RE = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")
_PATH_RE = re.compile(r"/[^\s\"']{8,}")
_DOWNLOAD_OK = {"1", "true", "yes"}
_REMOTE_JOIN_S = 4.0
_LOCK_WAIT_S = 0.2
_TURN_CAP = 8000
_PASSAGE_CAP = 600
_PREFETCH_CAP = 3500
_QUERY_CAP = 500
_N_RESULTS = 5
_DEFAULT_TENANT = "default_tenant"
_DEFAULT_DATABASE = "default_database"
_CLOUD_HOST = "api.trychroma.com"

_MESSAGES = {
    "model_missing": (
        "The embedding archive is not in this profile. "
        f"Pinned file onnx.tar.gz is {fetch.ARCHIVE_BYTES} bytes, sha256 {fetch.ARCHIVE_SHA256}, "
        f"from {fetch.ARCHIVE_URL}. "
        "Set CHROMA_MODEL_DOWNLOAD to 1 and call chroma_recall, or place that file."
    ),
    "chroma_missing": (
        "chromadb is not installed in this Hermes environment. "
        "Enable the plugin again so the declared dependency can be installed."
    ),
    "collection_incompatible": (
        "This collection uses a different embedding function. "
        "It was left unchanged. Choose one embedded with the default MiniLM function."
    ),
    "chroma_busy": "Still answering a previous request. The conversation can continue. Try again.",
    "chroma_broken": "Chroma did not answer. The conversation can continue. Try again.",
    "server_url": (
        "The server URL must be an http or https origin with no user name, password, path, or query. "
        "The default http port here is 8000. "
        "Over plain http, a key in the header X-Chroma-Token is sent unencrypted."
    ),
    "cloud_config": "Cloud needs tenant, database, and key together. Nothing was sent.",
    "collection_name": (
        "Use 3 to 512 characters from letters, digits, dot, underscore, and hyphen. "
        "Start and end with a letter or digit. "
        "Do not use two periods in a row or an IPv4 address."
    ),
    "collection_missing": (
        "That collection is not in this store. "
        "Set CHROMA_COLLECTION to a name that is already there. Nothing was written."
    ),
    "path_refused": "The store path is not a real directory. Nothing was written.",
    "empty_query": "Pass a query of 1 to 500 characters.",
    "query_too_long": "Pass a query of 1 to 500 characters.",
    "bad_arguments": "The only field is query, and it must be a string.",
    "hash_mismatch": (
        "The archive hash does not match the pin. It was not extracted. "
        "Replace the file or fetch it again."
    ),
    "store_failed": "The turn was not stored. The conversation can continue. Try again.",
    "recall_failed": "The collection did not answer. The conversation can continue. Try again.",
}


def fail(error_type: str, error: str) -> dict:
    return {"ok": False, "error_type": error_type, "error": error}


def _message(error_type: str, detail: str = "") -> str:
    base = _MESSAGES[error_type]
    if detail:
        return f"{base} {detail}"
    return base


@dataclass(frozen=True)
class Target:
    kind: str
    path: str = ""
    host: str = ""
    port: int = 0
    ssl: bool = False
    tenant: str = _DEFAULT_TENANT
    database: str = _DEFAULT_DATABASE
    api_key: str = ""
    collection: str = "hermes"
    cloud_host: str = ""
    create_path: bool = False

    def cache_key(self) -> tuple:
        return (
            self.kind,
            self.path,
            self.host,
            self.port,
            self.ssl,
            self.tenant,
            self.database,
            self.api_key,
            self.cloud_host,
        )


def chromadb_module():
    import chromadb

    return chromadb


def read_secrets(secret_fn) -> dict[str, str]:
    found: dict[str, str] = {}
    for name in SECRET_NAMES:
        value = secret_fn(name)
        if value is None:
            found[name] = ""
        else:
            found[name] = str(value)
    return found


def scrub(text: str, secrets: list[str], home: str) -> str:
    out = str(text).replace("\x00", "")
    for secret in secrets:
        if secret and len(secret) >= 4:
            out = out.replace(secret, "[redacted]")
    if home:
        out = out.replace(str(home), "[profile]")
    out = _PATH_RE.sub("[path]", out)
    if len(out) > 300:
        out = out[:300]
    return out


def _secret_values(secrets: dict[str, str]) -> list[str]:
    # The collection name is not a secret. A 404 that names it has to stay readable.
    values = [
        value
        for name, value in secrets.items()
        if name != "CHROMA_COLLECTION" and value and len(value) >= 4
    ]
    values.sort(key=len, reverse=True)
    return values


def config_path(home: Path) -> Path:
    return Path(home) / "plugin-data" / "chroma-memory" / "config.json"


def read_config(home: Path) -> dict[str, str]:
    path = config_path(home)
    if path.is_symlink():
        raise WriteRefused("symlink", "config.json")
    if not path.exists():
        return {}
    if not path.is_file():
        raise WriteRefused("outside", "config.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    kept = {}
    for key in ("collection", "path"):
        value = data.get(key)
        if isinstance(value, str):
            kept[key] = value
    return kept


def save_profile_config(home, values: dict) -> None:
    kept = {}
    for key in ("collection", "path"):
        if key in values and values[key] is not None:
            kept[key] = str(values[key])
    root = Path(home) / "plugin-data" / "chroma-memory"
    root.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(kept, sort_keys=True).encode("utf-8")
    write_bytes(root, "config.json", raw, create=True)


def _text(value: str) -> str:
    return (value or "").strip()


def _collection_name(secrets: dict[str, str], config: dict[str, str]) -> str:
    name = _text(secrets.get("CHROMA_COLLECTION", "")) or _text(config.get("collection", "")) or "hermes"
    return name


def _valid_ipv4(name: str) -> bool:
    if not _IPV4_RE.fullmatch(name):
        return False
    return all(0 <= int(part) <= 255 for part in name.split("."))


def collection_name_ok(name: str) -> bool:
    """Chroma 1.5.9 PersistentClient accepts this name."""
    if not _COLLECTION_RE.fullmatch(name):
        return False
    if ".." in name:
        return False
    if _valid_ipv4(name):
        return False
    return True


def _shown_name(name: str, secrets: dict[str, str]) -> str:
    shown = (name or "").replace("\x00", "")
    for secret in _secret_values(secrets):
        shown = shown.replace(secret, "[redacted]")
    if len(shown) > 80:
        shown = shown[:80]
    return shown


def _collection_name_error(name: str, secrets: dict[str, str]) -> str:
    shown = _shown_name(name, secrets)
    return f"Collection name {shown} is not a Chroma collection name. {_message('collection_name')}"


def _collection_missing_error(name: str, found: list[str], secrets: dict[str, str]) -> str:
    shown = _shown_name(name, secrets)
    visible = []
    for item in sorted(set(found)):
        visible.append(_shown_name(item, secrets))
    listed = visible[:20]
    extra = len(visible) - len(listed)
    joined = ", ".join(listed) if listed else "(none)"
    more = f" {extra} more are not listed." if extra else ""
    return (
        f"Collection {shown} is not in this store. "
        f"Collections here: {joined}.{more} "
        "Set CHROMA_COLLECTION to one of those names. Nothing was written."
    )


def parse_origin(url: str) -> tuple[str, int, bool]:
    if any(char in url for char in ("\x00", " ", "\t", "\n", "\r")):
        raise ValueError("origin")
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("origin")
    if parsed.username or parsed.password:
        raise ValueError("origin")
    if parsed.query or parsed.fragment:
        raise ValueError("origin")
    if parsed.path not in ("", "/"):
        raise ValueError("origin")
    host = parsed.hostname
    if not host:
        raise ValueError("origin")
    if parsed.port is not None:
        port = int(parsed.port)
    elif parsed.scheme == "http":
        port = 8000
    else:
        port = 443
    return host, port, parsed.scheme == "https"


def resolve_target(home: Path, secrets: dict[str, str], config: dict[str, str]):
    collection = _collection_name(secrets, config)
    if not collection_name_ok(collection):
        return fail("collection_name", _collection_name_error(collection, secrets))
    url = _text(secrets.get("CHROMA_SERVER_URL", ""))
    key = _text(secrets.get("CHROMA_API_KEY", ""))
    tenant = _text(secrets.get("CHROMA_TENANT", ""))
    database = _text(secrets.get("CHROMA_DATABASE", ""))
    if url:
        try:
            host, port, ssl = parse_origin(url)
        except ValueError:
            return fail("server_url", _message("server_url"))
        return Target(
            kind="server",
            host=host,
            port=port,
            ssl=ssl,
            tenant=tenant or _DEFAULT_TENANT,
            database=database or _DEFAULT_DATABASE,
            api_key=key,
            collection=collection,
        )
    if key or tenant or database:
        if not (key and tenant and database):
            return fail("cloud_config", _message("cloud_config"))
        return Target(
            kind="cloud",
            tenant=tenant,
            database=database,
            api_key=key,
            collection=collection,
            cloud_host=_CLOUD_HOST,
            port=443,
            ssl=True,
        )
    raw_path = _text(secrets.get("CHROMA_PATH", "")) or _text(config.get("path", ""))
    if raw_path:
        return _path_target(raw_path, collection, create_path=False)
    default = Path(home) / "plugin-data" / "chroma-memory" / "store"
    return _path_target(str(default), collection, create_path=True)


def _path_target(raw: str, collection: str, *, create_path: bool):
    path = Path(raw)
    if not path.is_absolute():
        return fail("path_refused", _message("path_refused"))
    if path.is_symlink():
        return fail("path_refused", _message("path_refused"))
    resolved = path.resolve()
    try:
        directory_is_real(resolved)
    except WriteRefused:
        return fail("path_refused", _message("path_refused"))
    if resolved.exists() and not resolved.is_dir():
        return fail("path_refused", _message("path_refused"))
    if not resolved.exists() and not create_path:
        return fail("path_refused", _message("path_refused"))
    return Target(kind="embedded", path=str(resolved), collection=collection, create_path=create_path)


def _clip(text: str, limit: int) -> str:
    cleaned = (text or "").replace("\x00", "")
    if len(cleaned) > limit:
        return cleaned[:limit]
    return cleaned


def _download_on(value: str) -> bool:
    return value.strip().lower() in _DOWNLOAD_OK


def format_prefetch(passages: list[dict]) -> str:
    if not passages:
        return ""
    lines = ["Chroma passages:"]
    for row in passages:
        lines.append(f"distance {row['distance']}: {row['text']}")
    text = "\n".join(lines)
    if len(text) > _PREFETCH_CAP:
        return text[:_PREFETCH_CAP]
    return text


def _parse_query(arguments, *, source: str):
    if not isinstance(arguments, dict):
        return fail("bad_arguments", _message("bad_arguments"))
    if set(arguments) - {"query"}:
        return fail("bad_arguments", _message("bad_arguments"))
    query = arguments.get("query")
    if not isinstance(query, str) or "\x00" in query:
        return fail("bad_arguments", _message("bad_arguments"))
    query = query.strip()
    if source == "prefetch":
        if not query:
            return ""
        if len(query) > _QUERY_CAP:
            return query[:_QUERY_CAP]
        return query
    if not query:
        return fail("empty_query", _message("empty_query"))
    if len(query) > _QUERY_CAP:
        return fail("query_too_long", _message("query_too_long"))
    return query


class Store:
    def __init__(self, home, secret_fn, embedder=None, client_factory=None, fetcher=None):
        self.home = Path(home)
        self.secret_fn = secret_fn
        self.embedder = embedder
        self.client_factory = client_factory or default_client
        self.fetcher = fetcher or fetch.fetch_pinned
        self._client = None
        self._key = None
        self._lock = threading.Lock()

    def close(self) -> None:
        self._client = None
        self._key = None

    def recall(self, arguments, *, allow_download: bool = False, source: str = "tool") -> dict:
        parsed = _parse_query(arguments, source=source)
        if isinstance(parsed, dict):
            return parsed
        if source == "prefetch" and parsed == "":
            return {"ok": True, "count": 0, "passages": [], "text": ""}
        try:
            secrets = read_secrets(self.secret_fn)
            config = read_config(self.home)
            target = resolve_target(self.home, secrets, config)
        except UnscopedSecretError:
            raise
        except WriteRefused:
            return fail("path_refused", _message("path_refused"))
        except Exception as exc:
            return self._broken(exc, {})
        if isinstance(target, dict):
            return target
        try:
            embedder = self._prepare_embedder(secrets, allow_download=allow_download and source == "tool")
        except UnscopedSecretError:
            raise
        except _Step as step:
            return step.payload
        except ImportError:
            return fail("chroma_missing", _message("chroma_missing"))
        except Exception as exc:
            return self._broken(exc, secrets)

        def run(client):
            try:
                collection = self._collection(client, target, embedder, secrets, write=False)
            except _EmptyStore:
                return {"ok": True, "count": 0, "passages": [], "text": ""}
            return self._query(collection, parsed)

        try:
            return self._run(target, run, secrets)
        except UnscopedSecretError:
            raise
        except _Step as step:
            return step.payload
        except ImportError:
            return fail("chroma_missing", _message("chroma_missing"))
        except Exception as exc:
            return self._broken(exc, secrets)

    def add_turn(self, user: str, assistant: str, session_id: str) -> dict | None:
        user_text = _clip(user, _TURN_CAP)
        assistant_text = _clip(assistant, _TURN_CAP)
        if not user_text.strip() and not assistant_text.strip():
            return None
        try:
            secrets = read_secrets(self.secret_fn)
            config = read_config(self.home)
            target = resolve_target(self.home, secrets, config)
        except UnscopedSecretError:
            raise
        except WriteRefused:
            logger.warning("chroma-memory path_refused")
            return fail("path_refused", _message("path_refused"))
        except Exception:
            logger.warning("chroma-memory store_failed")
            return fail("store_failed", _message("store_failed"))
        if isinstance(target, dict):
            logger.warning("chroma-memory %s", target.get("error_type", "store_failed"))
            return target
        try:
            embedder = self._prepare_embedder(secrets, allow_download=False)
        except UnscopedSecretError:
            raise
        except _Step as step:
            logger.warning("chroma-memory %s", step.payload.get("error_type", "store_failed"))
            return step.payload
        except Exception:
            logger.warning("chroma-memory store_failed")
            return fail("store_failed", _message("store_failed"))

        def run(client):
            collection = self._collection(client, target, embedder, secrets, write=True)
            document = f"user: {user_text}\nassistant: {assistant_text}"
            metadata = {"kind": "hermes-turn", "session_id": _clip(session_id or "", 200)}
            collection.add(ids=[f"hermes-{uuid.uuid4().hex}"], documents=[document], metadatas=[metadata])
            return None

        try:
            return self._run(target, run, secrets)
        except UnscopedSecretError:
            raise
        except _Step as step:
            logger.warning("chroma-memory %s", step.payload.get("error_type", "store_failed"))
            return step.payload
        except ImportError:
            logger.warning("chroma-memory chroma_missing")
            return fail("chroma_missing", _message("chroma_missing"))
        except Exception:
            logger.warning("chroma-memory store_failed")
            return fail("store_failed", _message("store_failed"))

    def stored_documents(self) -> list[tuple[str, str]]:
        secrets = read_secrets(self.secret_fn)
        config = read_config(self.home)
        target = resolve_target(self.home, secrets, config)
        if isinstance(target, dict):
            return []
        embedder = self.embedder
        if embedder is None:
            return []

        def run(client):
            try:
                collection = self._collection(client, target, embedder, secrets, write=False)
            except _EmptyStore:
                return []
            data = collection.get(include=["documents"])
            ids = data.get("ids") or []
            docs = data.get("documents") or []
            return list(zip(ids, docs))

        result = self._run(target, run, secrets)
        if isinstance(result, dict):
            return []
        return result

    def _prepare_embedder(self, secrets: dict[str, str], *, allow_download: bool):
        if self.embedder is not None:
            return self.embedder
        model_dir = self.home / "plugin-data" / "chroma-memory" / "model"
        state = _model_state(model_dir)
        if state == "ready":
            return MiniLM(model_dir)
        if state == "hash":
            raise _Step(fail("hash_mismatch", _message("hash_mismatch")))
        if state == "missing" and allow_download and _download_on(secrets.get("CHROMA_MODEL_DOWNLOAD", "")):
            try:
                data = self.fetcher()
                fetch.verify_archive(data)
            except fetch.FetchRefused as exc:
                if exc.kind == "hash":
                    raise _Step(fail("hash_mismatch", _message("hash_mismatch"))) from exc
                raise _Step(fail("model_missing", _message("model_missing"))) from exc
            model_dir.mkdir(parents=True, exist_ok=True)
            write_bytes(model_dir, "onnx.tar.gz", data, create=True)
            fetch.extract_members(data, model_dir)
            return MiniLM(model_dir)
        raise _Step(fail("model_missing", _message("model_missing")))

    def _run(self, target: Target, fn, secrets: dict[str, str]):
        if target.kind == "embedded":
            return fn(self._open(target))
        if not self._lock.acquire(timeout=_LOCK_WAIT_S):
            return fail("chroma_busy", _message("chroma_busy"))
        holder: dict = {}

        def work() -> None:
            try:
                holder["value"] = fn(self._open(target))
            except Exception as exc:
                holder["error"] = exc
            finally:
                self._lock.release()

        thread = threading.Thread(target=work, name="chroma-memory-remote", daemon=True)
        thread.start()
        thread.join(_REMOTE_JOIN_S)
        if thread.is_alive():
            return fail("chroma_broken", _message("chroma_broken"))
        if "error" in holder:
            raise holder["error"]
        return holder.get("value")

    def _open(self, target: Target):
        key = target.cache_key()
        if self._client is not None and self._key == key:
            return self._client
        self.close()
        if target.kind == "embedded" and target.create_path:
            Path(target.path).mkdir(parents=True, exist_ok=True)
        self._client = self.client_factory(target)
        self._key = key
        return self._client

    def _collection(self, client, target: Target, embedder, secrets: dict[str, str], *, write: bool):
        ef = embedder.chroma()
        try:
            collection = client.get_collection(target.collection, embedding_function=ef)
        except Exception as exc:
            if not _missing_collection(exc):
                raise
            found = _listed_names(client)
            if target.collection in found:
                collection = client.get_collection(target.collection, embedding_function=ef)
            elif found or not write:
                if found:
                    raise _Step(fail(
                        "collection_missing",
                        _collection_missing_error(target.collection, found, secrets),
                    ))
                raise _EmptyStore()
            else:
                collection = _create_collection(client, target.collection, ef, embedder)
                return collection
        if not _compatible(collection, embedder):
            raise _Step(fail("collection_incompatible", _message("collection_incompatible")))
        return collection

    def _query(self, collection, query: str) -> dict:
        count = int(collection.count())
        if count <= 0:
            return {"ok": True, "count": 0, "passages": [], "text": ""}
        raw = collection.query(query_texts=[query], n_results=_N_RESULTS, include=["documents", "distances", "metadatas"])
        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        passages = []
        for index, doc in enumerate(docs):
            text = "" if doc is None else str(doc)
            clipped_text, clipped = text, False
            if len(text) > _PASSAGE_CAP:
                clipped_text, clipped = text[:_PASSAGE_CAP], True
            distance = distances[index] if index < len(distances) else None
            ident = ids[index] if index < len(ids) else ""
            passages.append(
                {"id": ident, "text": clipped_text, "distance": distance, "clipped": clipped}
            )
        return {"ok": True, "count": len(passages), "passages": passages, "text": format_prefetch(passages)}

    def _broken(self, exc: Exception, secrets: dict[str, str]) -> dict:
        if isinstance(exc, _Step):
            return exc.payload
        detail = scrub(f"{type(exc).__name__}: {exc}", _secret_values(secrets), str(self.home))
        logger.warning("chroma-memory chroma_broken")
        return fail("chroma_broken", _message("chroma_broken", detail))


class _Step(Exception):
    def __init__(self, payload: dict):
        self.payload = payload


class _EmptyStore(Exception):
    """The named collection is absent and the store has no collections."""


def _listed_names(client) -> list[str]:
    listed = client.list_collections()
    names = []
    for item in listed or []:
        if isinstance(item, str):
            names.append(item)
            continue
        name = getattr(item, "name", "")
        if name:
            names.append(str(name))
    return names


def _create_collection(client, name: str, ef, embedder):
    metadata = {"hermes_embedding": embedder.marker} if getattr(embedder, "marker", None) else None
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*legacy embedding function.*")
            if metadata:
                return client.create_collection(name, embedding_function=ef, metadata=metadata)
            return client.create_collection(name, embedding_function=ef)
    except Exception as exc:
        text = str(exc).lower()
        if type(exc).__name__ in {"InvalidArgumentError", "ValueError"} and "name" in text:
            shown = name if len(name) <= 80 else name[:80]
            raise _Step(fail(
                "collection_name",
                f"Collection name {shown} is not a Chroma collection name. {_message('collection_name')}",
            )) from exc
        raise


def _missing_collection(exc: Exception) -> bool:
    name = type(exc).__name__
    return name in {"NotFoundError", "InvalidCollectionException"}


def _compatible(collection, embedder) -> bool:
    config = collection.configuration_json or {}
    stored = config.get("embedding_function") or {}
    name = stored.get("name")
    if name and name not in embedder.allowed_names:
        return False
    metadata = collection.metadata or {}
    marker = metadata.get("hermes_embedding")
    if marker not in (None, "", "onnx_mini_lm_l6_v2"):
        return False
    model = getattr(collection, "_model", None)
    dimension = getattr(model, "dimension", None)
    if dimension is not None and int(dimension) != int(embedder.dimension):
        return False
    return True


def _model_state(model_dir: Path) -> str:
    archive = model_dir / "onnx.tar.gz"
    if archive.is_symlink():
        return "hash"
    if not archive.is_file():
        return "missing"
    digest = fetch.sha256_file(archive)
    if digest != fetch.ARCHIVE_SHA256 or archive.stat().st_size != fetch.ARCHIVE_BYTES:
        return "hash"
    marker = model_dir / "extract.sha256"
    onnx = model_dir / "onnx" / "model.onnx"
    if marker.is_symlink() or onnx.is_symlink():
        return "hash"
    if marker.is_file() and onnx.is_file():
        text = marker.read_text(encoding="utf-8", errors="replace")
        prefix = f"model.onnx "
        expected = ""
        for line in text.splitlines():
            if line.startswith(prefix):
                expected = line[len(prefix):].strip()
        if f"archive {fetch.ARCHIVE_SHA256}" in text and expected and fetch.sha256_file(onnx) == expected:
            return "ready"
    try:
        data = archive.read_bytes()
        fetch.extract_members(data, model_dir)
    except fetch.FetchRefused:
        return "hash"
    except WriteRefused:
        return "hash"
    return "ready"


def server_headers(target: Target) -> dict[str, str] | None:
    """Header for a server key. Plain http does not encrypt this value."""
    if not target.api_key:
        return None
    return {"X-Chroma-Token": target.api_key}


def default_client(target: Target):
    chromadb = chromadb_module()
    from chromadb.config import Settings

    settings = Settings(anonymized_telemetry=False)
    if target.kind == "embedded":
        return chromadb.PersistentClient(
            path=target.path,
            settings=settings,
            tenant=_DEFAULT_TENANT,
            database=_DEFAULT_DATABASE,
        )
    if target.kind == "server":
        headers = server_headers(target)
        return chromadb.HttpClient(
            host=target.host,
            port=target.port,
            ssl=target.ssl,
            headers=headers,
            settings=settings,
            tenant=target.tenant,
            database=target.database,
        )
    return chromadb.CloudClient(
        tenant=target.tenant,
        database=target.database,
        api_key=target.api_key,
        settings=settings,
        cloud_host=_CLOUD_HOST,
        cloud_port=443,
        enable_ssl=True,
    )
