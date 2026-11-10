"""Hermes-free checks for the Chroma store."""

from __future__ import annotations

import hashlib
import io
import tarfile
from email.message import EmailMessage
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request

import pytest

import fetch
import service
from service import Store, parse_origin, resolve_target, save_profile_config, server_headers


def _secrets(mapping):
    def read(name):
        return mapping.get(name)

    return read


class Toy:
    dimension = 32
    allowed_names = frozenset({"default", "onnx_mini_lm_l6_v2"})
    marker = None

    def chroma(self):
        return self

    def name(self):
        return "default"

    def get_config(self):
        return NotImplemented


class FakeCollection:
    def __init__(self, *, name="legacy", dimension=32, marker=None):
        stored = {"type": "legacy"} if name == "legacy" else {"type": "known", "name": name}
        self.configuration_json = {"embedding_function": stored}
        self.metadata = {} if marker is None else {"hermes_embedding": marker}
        self._model = SimpleNamespace(dimension=dimension)
        self.queries = []
        self.added = []

    def count(self):
        return 1

    def query(self, query_texts, n_results, include):
        self.queries.append(query_texts[0])
        return {
            "ids": [["a"]],
            "documents": [["user: hi\nassistant: there"]],
            "distances": [[0.2]],
            "metadatas": [[{}]],
        }

    def get(self, include):
        return {"ids": ["a"], "documents": ["user: hi\nassistant: there"]}

    def add(self, ids, documents, metadatas):
        self.added.append(documents[0])


class FakeClient:
    def __init__(self, collection):
        self.collection = collection

    def get_collection(self, name, embedding_function=None):
        return self.collection


def _store(tmp_path, mapping=None, **kwargs):
    return Store(tmp_path, _secrets(mapping or {}), **kwargs)


def test_bad_arguments_skip_the_client(tmp_path):
    def factory(target):
        raise AssertionError("client")

    store = _store(tmp_path, client_factory=factory, embedder=Toy())
    extra = store.recall({"query": "hi", "limit": 1})
    empty = store.recall({"query": "   "})
    long = store.recall({"query": "a" * 501})
    null = store.recall({"query": "a\x00b"})
    wrong = store.recall(["hi"])
    assert extra["error_type"] == "bad_arguments"
    assert empty["error_type"] == "empty_query"
    assert long["error_type"] == "query_too_long"
    assert null["error_type"] == "bad_arguments"
    assert wrong["error_type"] == "bad_arguments"


def test_tool_query_at_500_reaches_the_collection(tmp_path):
    collection = FakeCollection()

    def factory(target):
        return FakeClient(collection)

    store = _store(tmp_path, client_factory=factory, embedder=Toy())
    result = store.recall({"query": "b" * 500})
    assert result["ok"] is True
    assert result["count"] == 1
    assert collection.queries == ["b" * 500]


def test_prefetch_clips_a_long_query(tmp_path):
    collection = FakeCollection()

    def factory(target):
        return FakeClient(collection)

    store = _store(tmp_path, client_factory=factory, embedder=Toy())
    result = store.recall({"query": "c" * 800}, source="prefetch")
    assert result["ok"] is True
    assert collection.queries == ["c" * 500]


def test_prefetch_empty_query_is_silent(tmp_path):
    def factory(target):
        raise AssertionError("client")

    store = _store(tmp_path, client_factory=factory, embedder=Toy())
    result = store.recall({"query": "  "}, source="prefetch")
    assert result["ok"] is True
    assert result["count"] == 0
    assert result["text"] == ""


def test_other_name_is_left_unread(tmp_path):
    collection = FakeCollection(name="other-model")

    def factory(target):
        return FakeClient(collection)

    store = _store(tmp_path, client_factory=factory, embedder=Toy())
    result = store.recall({"query": "hi"})
    assert result["error_type"] == "collection_incompatible"
    assert collection.queries == []
    assert collection.added == []


def test_other_dimension_is_left_unread(tmp_path):
    collection = FakeCollection(dimension=999)

    def factory(target):
        return FakeClient(collection)

    store = _store(tmp_path, client_factory=factory, embedder=Toy())
    result = store.recall({"query": "hi"})
    assert result["error_type"] == "collection_incompatible"
    assert collection.queries == []


def test_foreign_marker_is_left_unread(tmp_path):
    collection = FakeCollection(marker="other-archive")

    def factory(target):
        return FakeClient(collection)

    store = _store(tmp_path, client_factory=factory, embedder=Toy())
    result = store.recall({"query": "hi"})
    assert result["error_type"] == "collection_incompatible"


def test_model_missing_does_not_fetch(tmp_path):
    called = []

    def fetcher():
        called.append(1)
        return b""

    store = _store(tmp_path, fetcher=fetcher)
    result = store.recall({"query": "hi"}, allow_download=False, source="tool")
    assert result["error_type"] == "model_missing"
    assert str(fetch.ARCHIVE_BYTES) in result["error"]
    assert fetch.ARCHIVE_SHA256 in result["error"]
    assert fetch.ARCHIVE_URL in result["error"]
    assert called == []


def test_prefetch_does_not_fetch_when_the_flag_is_set(tmp_path):
    called = []

    def fetcher():
        called.append(1)
        return b"nope"

    store = _store(tmp_path, {"CHROMA_MODEL_DOWNLOAD": "1"}, fetcher=fetcher)
    result = store.recall({"query": "hi"}, allow_download=True, source="prefetch")
    assert result["error_type"] == "model_missing"
    assert called == []


def test_tool_fetch_runs_once_when_the_flag_is_set(tmp_path):
    called = []

    def fetcher():
        called.append(1)
        return b"short"

    store = _store(tmp_path, {"CHROMA_MODEL_DOWNLOAD": "yes"}, fetcher=fetcher)
    result = store.recall({"query": "hi"}, allow_download=True, source="tool")
    assert result["error_type"] == "hash_mismatch"
    assert called == [1]


def test_bad_archive_on_disk_is_a_hash_mismatch(tmp_path):
    model = tmp_path / "plugin-data" / "chroma-memory" / "model"
    model.mkdir(parents=True)
    (model / "onnx.tar.gz").write_bytes(b"not-the-archive")
    called = []
    store = _store(tmp_path, fetcher=lambda: called.append(1))
    result = store.recall({"query": "hi"})
    assert result["error_type"] == "hash_mismatch"
    assert called == []


def test_chroma_missing(tmp_path, monkeypatch):
    def missing():
        raise ImportError("chromadb")

    monkeypatch.setattr(service, "chromadb_module", missing)
    store = _store(tmp_path, embedder=Toy())
    result = store.recall({"query": "hi"})
    assert result["error_type"] == "chroma_missing"


def test_broken_client_returns_the_next_step(tmp_path):
    def factory(target):
        raise RuntimeError("disk broke")

    store = _store(tmp_path, embedder=Toy(), client_factory=factory)
    result = store.recall({"query": "hi"})
    assert result["error_type"] == "chroma_broken"
    assert "disk broke" in result["error"]


def test_secret_in_a_cloud_error_is_scrubbed(tmp_path):
    key = "cloud-key-example"
    seen = []

    def factory(target):
        seen.append((target.cloud_host, target.port, target.ssl, target.tenant, target.database))
        raise RuntimeError(f"rejected {target.api_key}")

    mapping = {
        "CHROMA_API_KEY": key,
        "CHROMA_TENANT": "tenant-example",
        "CHROMA_DATABASE": "db-example",
    }
    store = _store(tmp_path, mapping, embedder=Toy(), client_factory=factory)
    result = store.recall({"query": "hi"})
    assert result["error_type"] == "chroma_broken"
    assert key not in result["error"]
    assert seen == [("api.trychroma.com", 443, True, "tenant-example", "db-example")]


def test_unused_secret_is_scrubbed(tmp_path):
    token = "download-token-zz"

    def factory(target):
        raise RuntimeError(f"disk {token} {tmp_path}")

    store = _store(tmp_path, {"CHROMA_MODEL_DOWNLOAD": token}, embedder=Toy(), client_factory=factory)
    result = store.recall({"query": "hi"})
    assert token not in result["error"]
    assert str(tmp_path) not in result["error"]


def test_incomplete_cloud_does_not_open_a_client(tmp_path):
    def factory(target):
        raise AssertionError("client")

    store = _store(tmp_path, {"CHROMA_API_KEY": "only-key"}, embedder=Toy(), client_factory=factory)
    result = store.recall({"query": "hi"})
    assert result["error_type"] == "cloud_config"


def test_server_url_rules(tmp_path):
    home = tmp_path
    accepted = (
        ("http://chroma.example", ("chroma.example", 8000, False)),
        ("https://chroma.example", ("chroma.example", 443, True)),
        ("http://chroma.example:9", ("chroma.example", 9, False)),
        ("https://chroma.example:8443", ("chroma.example", 8443, True)),
        ("HTTP://chroma.example", ("chroma.example", 8000, False)),
        ("https://chroma.example/", ("chroma.example", 443, True)),
        ("http://chroma.example/", ("chroma.example", 8000, False)),
        ("http://127.0.0.1", ("127.0.0.1", 8000, False)),
        ("http://[::1]:8000", ("::1", 8000, False)),
        ("http://chroma.example:8000", ("chroma.example", 8000, False)),
        ("http://chroma.example#", ("chroma.example", 8000, False)),
    )
    for url, expected in accepted:
        assert parse_origin(url) == expected, url
    refused = (
        "http://user:pass@chroma.example",
        "http://chroma.example/extra",
        "http://chroma.example?q=1",
        "http://chroma.example#frag",
        "ftp://chroma.example",
        "http://chro ma.example",
        "chroma.example:8000",
        "http://",
        "https://user@chroma.example",
    )
    for url in refused:
        target = resolve_target(home, {"CHROMA_SERVER_URL": url}, {})
        assert isinstance(target, dict), url
        assert target["error_type"] == "server_url", url
        assert "unencrypted" in target["error"]
    plain = resolve_target(
        home,
        {"CHROMA_SERVER_URL": "http://chroma.example", "CHROMA_API_KEY": "plain-key"},
        {},
    )
    assert plain.ssl is False
    assert plain.api_key == "plain-key"
    assert server_headers(plain) == {"X-Chroma-Token": "plain-key"}
    secure = resolve_target(
        home,
        {"CHROMA_SERVER_URL": "https://chroma.example", "CHROMA_API_KEY": "tls-key"},
        {},
    )
    assert secure.ssl is True
    assert server_headers(secure) == {"X-Chroma-Token": "tls-key"}
    open_server = resolve_target(home, {"CHROMA_SERVER_URL": "http://chroma.example"}, {})
    assert server_headers(open_server) is None


def test_plain_http_disclosure_is_public():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    manifest = (root / "plugin.yaml").read_text(encoding="utf-8")
    skill = (root / "skills" / "chroma-memory" / "SKILL.md").read_text(encoding="utf-8")
    for text in (readme, manifest, skill):
        assert "unencrypted" in text
    assert "X-Chroma-Token" in readme


def test_readme_install_follows_the_floor_flag():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "hermes plugins install --help" in readme
    assert 'hermes plugins install "file://${PWD}" --enable --yes-deps' in readme
    assert 'hermes plugins install "file://${PWD}" --enable\n' in readme
    help_at = readme.index("hermes plugins install --help")
    flagged = readme.index('hermes plugins install "file://${PWD}" --enable --yes-deps')
    plain = readme.index('hermes plugins install "file://${PWD}" --enable\n')
    assert help_at < flagged < plain


def test_relative_and_missing_paths_are_refused(tmp_path):
    relative = resolve_target(tmp_path, {"CHROMA_PATH": "store"}, {})
    assert relative["error_type"] == "path_refused"
    missing = resolve_target(tmp_path, {"CHROMA_PATH": str(tmp_path / "absent")}, {})
    assert missing["error_type"] == "path_refused"
    link = tmp_path / "link"
    real = tmp_path / "real"
    real.mkdir()
    link.symlink_to(real)
    linked = resolve_target(tmp_path, {"CHROMA_PATH": str(link)}, {})
    assert linked["error_type"] == "path_refused"


def test_collection_name_rules(tmp_path):
    refused = (
        "x",
        "ab",
        "José",
        'a/b?"',
        "a..b",
        "1.2.3.4",
        "0.0.0.0",
        "ab.",
        ".ab",
        "a" * 513,
        "::1",
        "a b c",
    )
    for name in refused:
        result = resolve_target(tmp_path, {"CHROMA_COLLECTION": name}, {})
        assert result["error_type"] == "collection_name", name
        assert name[:80] in result["error"]
    accepted = (
        "hermes",
        "notes",
        "Jose",
        "a.b",
        "n_1",
        "A-b",
        "a" * 63,
        "b" * 64,
        "c" * 512,
        "hermes.notes",
        "a--b",
        "x.y.z",
        "999.999.999.999",
        "256.1.1.1",
        "1.2.3",
    )
    for name in accepted:
        target = resolve_target(tmp_path, {"CHROMA_COLLECTION": name}, {})
        assert getattr(target, "collection", None) == name, name


def test_symlink_config_is_refused(tmp_path):
    root = tmp_path / "plugin-data" / "chroma-memory"
    root.mkdir(parents=True)
    (root / "real.json").write_text("{}", encoding="utf-8")
    (root / "config.json").symlink_to(root / "real.json")

    def factory(target):
        raise AssertionError("client")

    store = _store(tmp_path, embedder=Toy(), client_factory=factory)
    result = store.recall({"query": "hi"})
    assert result["error_type"] == "path_refused"


def test_save_config_drops_secrets(tmp_path):
    save_profile_config(
        tmp_path,
        {"collection": "notes", "path": "/data/chroma", "api_key": "sekret-value"},
    )
    text = (tmp_path / "plugin-data" / "chroma-memory" / "config.json").read_text(encoding="utf-8")
    assert "sekret-value" not in text
    assert "notes" in text


def test_unscoped_secret_propagates(tmp_path):
    def read(name):
        raise service.UnscopedSecretError(name, "no scope")

    store = Store(tmp_path, read, embedder=Toy())
    with pytest.raises(service.UnscopedSecretError):
        store.recall({"query": "hi"})


def test_empty_turn_skips_the_client(tmp_path):
    def factory(target):
        raise AssertionError("client")

    store = _store(tmp_path, embedder=Toy(), client_factory=factory)
    assert store.add_turn("", "", "s") is None
    assert store.add_turn("\x00", " \x00 ", "s") is None


def test_member_rules_and_hash_before_extract(tmp_path):
    good = tarfile.TarInfo("onnx/tokenizer.json")
    good.type = tarfile.REGTYPE
    good.size = 1
    assert fetch.member_ok(good)
    parent = tarfile.TarInfo("onnx/../../etc/passwd")
    parent.type = tarfile.REGTYPE
    assert fetch.member_ok(parent) is False
    absolute = tarfile.TarInfo("/onnx/model.onnx")
    absolute.type = tarfile.REGTYPE
    assert fetch.member_ok(absolute) is False
    link = tarfile.TarInfo("onnx/model.onnx")
    link.type = tarfile.SYMTYPE
    assert fetch.member_ok(link) is False
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as archive:
        payload = b"{}"
        info = tarfile.TarInfo("onnx/../../etc/passwd")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(fetch.FetchRefused) as raised:
        fetch.extract_members(raw.getvalue(), tmp_path / "model")
    assert raised.value.kind == "hash"
    assert list(tmp_path.iterdir()) == []


def test_read_body_refuses_over_cap():
    class Chunks:
        def __init__(self):
            self.left = 3

        def read(self, _n):
            if self.left <= 0:
                return b""
            self.left -= 1
            return b"x" * 8

    with pytest.raises(fetch.FetchRefused) as raised:
        fetch.read_body(Chunks(), cap=10)
    assert raised.value.kind == "oversize"


def test_cross_host_redirect_is_refused():
    handler = fetch._SameHost()
    req = Request(fetch.ARCHIVE_URL)
    with pytest.raises(fetch.FetchRefused):
        handler.redirect_request(req, None, 302, "Found", EmailMessage(), "https://evil.example/onnx.tar.gz")
    same = handler.redirect_request(req, None, 302, "Found", EmailMessage(), fetch.ARCHIVE_URL)
    assert same.get_full_url().startswith("https://" + fetch.ARCHIVE_HOST)


@pytest.mark.parametrize(
    "name",
    [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
        "ftp_proxy",
    ],
)
def test_archive_opener_has_an_empty_proxy(monkeypatch, name):
    import urllib.request

    monkeypatch.setenv(name, "http://127.0.0.1:9")
    opener = fetch.build_opener()
    ours = [getattr(handler, "proxies", {}) for handler in opener.handlers]
    assert all(not mapping for mapping in ours)
    default = urllib.request.build_opener()
    env_proxies = [getattr(handler, "proxies", {}) for handler in default.handlers]
    assert any(mapping for mapping in env_proxies)


def test_pinned_constants_match_the_known_archive():
    assert fetch.ARCHIVE_SHA256 == "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
    assert fetch.ARCHIVE_BYTES == 83178821
    path = Path("/tmp/chroma-model/onnx.tar.gz")
    if not path.is_file():
        return
    assert path.stat().st_size == fetch.ARCHIVE_BYTES
    assert fetch.sha256_file(path) == fetch.ARCHIVE_SHA256
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            assert fetch.member_ok(member)


def test_remote_call_returns_while_the_worker_is_still_busy(tmp_path):
    import time

    def factory(target):
        time.sleep(6)
        raise RuntimeError("late")

    store = _store(
        tmp_path,
        {"CHROMA_SERVER_URL": "http://127.0.0.1:9"},
        embedder=Toy(),
        client_factory=factory,
    )
    started = time.monotonic()
    first = store.recall({"query": "hi"})
    assert first["error_type"] == "chroma_broken"
    second = store.recall({"query": "hi"})
    assert second["error_type"] == "chroma_busy"
    assert time.monotonic() - started < 6


def _bag():
    pytest.importorskip("chromadb")
    import numpy as np
    from chromadb.api.types import Documents, EmbeddingFunction

    class Bag(EmbeddingFunction[Documents]):
        def __init__(self) -> None:
            return None

        def __call__(self, input):
            rows = []
            for text in input:
                vec = [0.0] * 32
                for token in str(text).lower().split():
                    digest = hashlib.sha256(token.encode("utf-8")).digest()
                    vec[digest[0] % 32] += 1.0 if digest[1] % 2 == 0 else -1.0
                norm = sum(value * value for value in vec) ** 0.5 or 1.0
                rows.append(np.array([value / norm for value in vec], dtype=np.float32))
            return rows

        @staticmethod
        def name():
            return "default"

        def get_config(self):
            return NotImplemented

        @staticmethod
        def build_from_config(config):
            return NotImplemented

    class Holder:
        dimension = 32
        allowed_names = frozenset({"default"})
        marker = None

        def chroma(self):
            return Bag()

    return Holder()


def test_write_then_recall_and_profiles_stay_apart(tmp_path):
    bag = _bag()
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()
    store_a = Store(home_a, _secrets({}), embedder=bag)
    store_b = Store(home_b, _secrets({}), embedder=bag)
    store_a.add_turn("authentication decision alpha", "use token rotation", "s1")
    found = store_a.recall({"query": "authentication decision"})
    assert found["ok"] is True
    assert found["count"] >= 1
    assert "token rotation" in found["text"]
    other = store_b.recall({"query": "authentication decision"})
    assert "token rotation" not in other.get("text", "")
    before = store_a.stored_documents()
    again = store_a.recall({"query": "authentication decision"})
    assert store_a.stored_documents() == before
    assert again["ok"] is True
    store_a.close()
    store_b.close()


def test_turn_and_passage_are_clipped(tmp_path):
    bag = _bag()
    store = Store(tmp_path, _secrets({}), embedder=bag)
    store.add_turn("hi", "A" * 9000, "s")
    _ident, text = store.stored_documents()[0]
    assert "A" * 8000 in text
    assert "A" * 8001 not in text
    found = store.recall({"query": "hi"})
    assert found["passages"][0]["clipped"] is True
    assert len(found["passages"][0]["text"]) == 600
    store.close()


def test_outside_path_keeps_its_own_documents(tmp_path):
    bag = _bag()
    outside = tmp_path / "outside"
    outside.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    store = Store(home, _secrets({"CHROMA_PATH": str(outside)}), embedder=bag)
    store.add_turn("outside sentence", "kept apart", "s")
    assert (outside / "chroma.sqlite3").is_file() or any(outside.iterdir())
    assert not (home / "plugin-data" / "chroma-memory" / "store").exists()
    found = store.recall({"query": "outside sentence"})
    assert "kept apart" in found["text"]
    store.close()


def test_create_leaves_the_default_embedding_registration(tmp_path):
    pytest.importorskip("chromadb")
    from chromadb.api.types import DefaultEmbeddingFunction
    from chromadb.utils.embedding_functions import known_embedding_functions

    before = known_embedding_functions["default"]
    store = Store(tmp_path, _secrets({}), embedder=_bag())
    store.add_turn("hello", "world", "s")
    assert known_embedding_functions["default"] is before
    assert known_embedding_functions["default"] is DefaultEmbeddingFunction
    store.close()


def test_named_function_on_a_real_collection_is_left_in_place(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    import numpy as np
    from chromadb.api.types import Documents, EmbeddingFunction
    from chromadb.config import Settings
    from chromadb.utils.embedding_functions import known_embedding_functions

    class Other(EmbeddingFunction[Documents]):
        def __init__(self) -> None:
            return None

        def __call__(self, input):
            rows = []
            for _text in input:
                rows.append(np.ones(32, dtype=np.float32))
            return rows

        @staticmethod
        def name():
            return "other-model"

        def get_config(self):
            return {"kind": "other"}

        @staticmethod
        def build_from_config(config):
            return Other()

    path = tmp_path / "store"
    path.mkdir()
    client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.create_collection("hermes", embedding_function=Other())
        collection.add(ids=["doc-1"], documents=["keep this sentence"])
        stored_name = (collection.configuration_json.get("embedding_function") or {}).get("name")
        store = Store(tmp_path, _secrets({"CHROMA_PATH": str(path)}), embedder=_bag())
        result = store.recall({"query": "keep this sentence"})
        again = collection.get(include=["documents"])
        assert again["documents"] == ["keep this sentence"]
        if stored_name not in (None, "default", "onnx_mini_lm_l6_v2"):
            assert result["error_type"] == "collection_incompatible"
        store.close()
    finally:
        known_embedding_functions.pop("other-model", None)


def test_download_path_class_attribute_stays_on_the_cache(tmp_path):
    pytest.importorskip("chromadb")
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
    from embed import MiniLM

    before = ONNXMiniLM_L6_V2.DOWNLOAD_PATH
    embedder = MiniLM(tmp_path / "model").chroma()
    assert ".cache" in str(ONNXMiniLM_L6_V2.DOWNLOAD_PATH)
    assert ONNXMiniLM_L6_V2.DOWNLOAD_PATH == before
    assert embedder.DOWNLOAD_PATH == tmp_path / "model"
    assert embedder._download is not ONNXMiniLM_L6_V2._download


def test_refused_collection_name_does_not_open_a_client(tmp_path):
    def factory(target):
        raise AssertionError("client")

    for name in ("José", 'a/b?"', "a..b", "1.2.3.4"):
        store = _store(
            tmp_path,
            {"CHROMA_COLLECTION": name},
            embedder=Toy(),
            client_factory=factory,
        )
        result = store.recall({"query": "José"})
        assert result["error_type"] == "collection_name"
        assert result["ok"] is False


def test_collection_name_is_not_scrubbed(tmp_path):
    key = "sekret-value-zz"

    def factory(target):
        raise RuntimeError(f"Collection notes does not exist {key}")

    store = _store(
        tmp_path,
        {
            "CHROMA_COLLECTION": "notes",
            "CHROMA_API_KEY": key,
            "CHROMA_TENANT": "tenant-example",
            "CHROMA_DATABASE": "db-example",
        },
        embedder=Toy(),
        client_factory=factory,
    )
    result = store.recall({"query": "hi"})
    assert result["error_type"] == "chroma_broken"
    assert "notes" in result["error"]
    assert key not in result["error"]
    assert "[redacted]" in result["error"]


def test_neighbor_collection_is_not_hidden_by_an_empty_default(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    from chromadb.config import Settings

    path = tmp_path / "store"
    path.mkdir()
    client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    embedder = _bag()
    notes = client.create_collection("notes", embedding_function=embedder.chroma())
    document = "user: authentication decision\nassistant: use token rotation"
    notes.add(ids=["doc-1"], documents=[document])
    home = tmp_path / "home"
    home.mkdir()
    store = Store(home, _secrets({"CHROMA_PATH": str(path)}), embedder=embedder)
    result = store.recall({"query": "authentication"})
    assert result["ok"] is False
    assert result["error_type"] == "collection_missing"
    assert "notes" in result["error"]
    assert "hermes" in result["error"]
    assert "token rotation" not in result.get("text", "")
    written = store.add_turn("authentication decision", "should not land", "s")
    assert written["error_type"] == "collection_missing"
    assert [item.name for item in client.list_collections()] == ["notes"]
    assert notes.get(include=["documents"])["documents"] == [document]
    named = Store(
        home,
        _secrets({"CHROMA_PATH": str(path), "CHROMA_COLLECTION": "notes"}),
        embedder=embedder,
    )
    found = named.recall({"query": "authentication"})
    assert found["ok"] is True
    assert "token rotation" in found["text"]
    missing = named.recall({"query": "authentication"})
    store.close()
    named.close()
    client.close() if hasattr(client, "close") else None
    other = Store(
        home,
        _secrets({"CHROMA_PATH": str(path), "CHROMA_COLLECTION": "absent-name"}),
        embedder=embedder,
    )
    absent = other.recall({"query": "authentication"})
    assert absent["error_type"] == "collection_missing"
    assert "notes" in absent["error"]
    assert [item.name for item in chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    ).list_collections()] == ["notes"]
    other.close()
    assert missing["ok"] is True


def test_empty_store_read_does_not_create_a_collection(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    from chromadb.config import Settings

    store = Store(tmp_path, _secrets({}), embedder=_bag())
    result = store.recall({"query": "hello"})
    assert result == {"ok": True, "count": 0, "passages": [], "text": ""}
    store.close()
    path = tmp_path / "plugin-data" / "chroma-memory" / "store"
    client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    assert [item.name for item in client.list_collections()] == []


def test_empty_existing_collection_is_a_real_zero(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    from chromadb.config import Settings

    path = tmp_path / "store"
    path.mkdir()
    client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    client.create_collection("hermes", embedding_function=_bag().chroma())
    home = tmp_path / "home"
    home.mkdir()
    store = Store(home, _secrets({"CHROMA_PATH": str(path)}), embedder=_bag())
    result = store.recall({"query": "authentication"})
    assert result["ok"] is True
    assert result["count"] == 0
    assert [item.name for item in client.list_collections()] == ["hermes"]
    store.close()


def test_config_collection_selects_the_existing_name(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    from chromadb.config import Settings

    path = tmp_path / "store"
    path.mkdir()
    client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
    embedder = _bag()
    notes = client.create_collection("notes", embedding_function=embedder.chroma())
    notes.add(ids=["doc-1"], documents=["user: authentication\nassistant: from notes"])
    home = tmp_path / "home"
    home.mkdir()
    save_profile_config(home, {"collection": "notes", "path": str(path), "api_key": "sekret-value"})
    store = Store(home, _secrets({}), embedder=embedder)
    result = store.recall({"query": "authentication"})
    assert result["ok"] is True
    assert "from notes" in result["text"]
    store.close()


def test_query_edges_stay_on_the_named_collection(tmp_path):
    store = Store(tmp_path, _secrets({}), embedder=_bag())
    store.add_turn("José kept the accent", "accent row stays", "s")
    store.add_turn("authentication decision alpha", "use token rotation", "s")
    queries = (
        "authentication;",
        "authentication # comment",
        "認証",
        "ＡＵＴＨ",
        "AUTHENTICATION",
        "  authentication  ",
        "https://example.com/authentication",
        "José",
        "Jose",
        'a/b?"',
        "what did we decide about authentication",
    )
    for query in queries:
        result = store.recall({"query": query})
        assert result["ok"] is True, query
        assert result["count"] >= 1
    jose = store.recall({"query": "José"})
    assert "accent row stays" in jose["text"]
    store.close()


def test_persistent_client_does_not_open_a_socket(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    import socket

    def boom(self, *args, **kwargs):
        raise AssertionError("connect")

    monkeypatch.setattr(socket.socket, "connect", boom)
    target = resolve_target(tmp_path, {}, {})
    Path(target.path).mkdir(parents=True, exist_ok=True)
    service.default_client(target)
