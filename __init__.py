"""Use a Chroma collection you already keep as Hermes memory."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from agent.memory_provider import MemoryProvider, RecallStatus, spawn_context_thread
    from agent.secret_scope import UnscopedSecretError, get_secret
    from hermes_constants import get_hermes_home
except ImportError:
    class MemoryProvider:  # noqa: D401
        def is_available(self) -> bool:
            return False

        def initialize(self, session_id: str, **kwargs) -> None:
            return None

        def get_tool_schemas(self) -> list:
            return []

    class RecallStatus:
        def __init__(self, provider_label: str, count: int, glyph: str = ""):
            self.provider_label = provider_label
            self.count = count
            self.glyph = glyph

    class UnscopedSecretError(RuntimeError):
        pass

    def get_secret(name: str, default=None):
        return default

    def get_hermes_home() -> str:
        import os

        return os.environ.get("HERMES_HOME", "")

    def spawn_context_thread(target, *, name: str, daemon: bool = True, args=(), kwargs=None):
        import threading

        return threading.Thread(target=target, name=name, daemon=daemon, args=args, kwargs=kwargs or {})

try:
    from . import schemas
    from .service import Store, save_profile_config
except ImportError:
    import schemas
    from service import Store, save_profile_config

_SKILL = Path(__file__).resolve().parent / "skills" / "chroma-memory" / "SKILL.md"
_SKILL_TEXT = (
    "Recall a past decision from a Chroma collection you already keep. "
    "Passages are prefetched before the next reply."
)


def _secret(name: str):
    return get_secret(name, None)


def _tool(args, **kwargs) -> str:
    store = Store(get_hermes_home(), _secret)
    try:
        result = store.recall(args, allow_download=True, source="tool")
    finally:
        store.close()
    return json.dumps(result)


class ChromaMemoryProvider(MemoryProvider):
    def __init__(self, embedder=None):
        self._embedder = embedder
        self._agent_context = "primary"
        self._pending = None
        self._status = None
        self._store = None
        self._home = None
        self._session_id = ""

    @property
    def name(self) -> str:
        return "chroma-memory"

    def is_available(self) -> bool:
        try:
            import chromadb  # noqa: F401
        except ImportError:
            return False
        return True

    def unavailable_reason(self) -> str:
        if self.is_available():
            return ""
        return (
            "chromadb is not installed in this Hermes environment. "
            "Enable the plugin again so the declared dependency can be installed."
        )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id or ""
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._status = None

    def system_prompt_block(self) -> str:
        return ""

    def get_tool_schemas(self) -> list:
        return [schemas.CHROMA_RECALL]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name != "chroma_recall":
            raise NotImplementedError(tool_name)
        return _tool(args, **kwargs)

    def recall_status(self):
        return self._status

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, rewound: bool = False, **kwargs) -> None:
        self._session_id = new_session_id or ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._pending is not None:
            self._pending.join(3.0)
        try:
            result = self._store_for().recall(
                {"query": query or ""}, allow_download=False, source="prefetch"
            )
        except UnscopedSecretError:
            raise
        except Exception:
            self._status = None
            return "Chroma did not answer. The conversation can continue. Try again."
        if result.get("ok") and int(result.get("count") or 0) > 0:
            self._status = RecallStatus("Chroma", int(result["count"]))
            return str(result.get("text") or "")
        self._status = None
        if not result.get("ok"):
            return str(result.get("error") or "")
        return ""

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages=None,
        turn_author=None,
    ) -> None:
        if self._agent_context != "primary":
            return
        user_text = user_content or ""
        assistant_text = assistant_content or ""
        if not user_text.strip() and not assistant_text.strip():
            return
        bound_session = session_id or self._session_id

        def write() -> None:
            try:
                self._store_for().add_turn(user_text, assistant_text, bound_session)
            except UnscopedSecretError:
                raise

        thread = spawn_context_thread(write, name="chroma-memory-sync")
        self._pending = thread
        thread.start()

    def shutdown(self) -> None:
        if self._pending is not None:
            self._pending.join(3.0)
            self._pending = None
        if self._store is not None:
            self._store.close()
            self._store = None

    def get_config_schema(self) -> list:
        return [
            {
                "key": "collection",
                "description": "Collection name in the Chroma you already keep.",
                "env_var": "CHROMA_COLLECTION",
                "default": "hermes",
            },
            {
                "key": "path",
                "description": "Absolute path of an existing embedded Chroma directory.",
                "env_var": "CHROMA_PATH",
            },
            {
                "key": "server_url",
                "description": "Origin of a Chroma server you already run. Plain http sends the key unencrypted.",
                "secret": True,
                "env_var": "CHROMA_SERVER_URL",
            },
            {
                "key": "api_key",
                "description": "Key for that server or for Chroma Cloud.",
                "secret": True,
                "env_var": "CHROMA_API_KEY",
            },
            {
                "key": "tenant",
                "description": "Chroma Cloud tenant.",
                "secret": True,
                "env_var": "CHROMA_TENANT",
            },
            {
                "key": "database",
                "description": "Chroma Cloud database.",
                "secret": True,
                "env_var": "CHROMA_DATABASE",
            },
            {
                "key": "model_download",
                "description": "Set to 1 to let chroma_recall fetch the pinned embedding archive.",
                "secret": True,
                "env_var": "CHROMA_MODEL_DOWNLOAD",
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        save_profile_config(hermes_home, values or {})

    def backup_paths(self) -> list:
        return []

    def _store_for(self) -> Store:
        home = Path(get_hermes_home())
        if self._store is None or self._home != home:
            if self._store is not None:
                self._store.close()
            self._home = home
            self._store = Store(home, _secret, embedder=self._embedder)
        return self._store


def register(ctx) -> None:
    ctx.register_memory_provider(ChromaMemoryProvider())
    ctx.register_tool(
        name=schemas.CHROMA_RECALL["name"],
        toolset="chroma-memory",
        schema=schemas.CHROMA_RECALL,
        handler=_tool,
    )
    ctx.register_skill("chroma-memory", _SKILL, description=_SKILL_TEXT)
