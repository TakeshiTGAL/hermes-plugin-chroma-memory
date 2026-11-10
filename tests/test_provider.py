"""Provider lifecycle against the public memory contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _plugin():
    spec = importlib.util.spec_from_file_location(
        "chroma_memory_plugin",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Ctx:
    def __init__(self):
        self.tools = []
        self.provider = None
        self.skills = []
        self.hooks = []

    def register_memory_provider(self, provider):
        self.provider = provider

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)

    def register_skill(self, *args, **kwargs):
        self.skills.append((args, kwargs))

    def register_hook(self, *args, **kwargs):
        self.hooks.append((args, kwargs))


def test_register_records_one_tool_one_provider_and_one_skill(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    module = _plugin()
    ctx = Ctx()
    module.register(ctx)
    assert ctx.provider is not None
    assert ctx.provider.name == "chroma-memory"
    assert [row["name"] for row in ctx.tools] == ["chroma_recall"]
    assert ctx.hooks == []
    assert ctx.skills
    assert ctx.provider.system_prompt_block() == ""
    assert ctx.provider.get_tool_schemas()[0]["name"] == "chroma_recall"
    assert ctx.provider.backup_paths() == []
    text = ctx.tools[0]["handler"]({"query": "what did we decide about authentication"})
    assert isinstance(text, str)
    payload = json.loads(text)
    assert payload["error_type"] in {"model_missing", "chroma_missing"}
    assert payload["ok"] is False


def test_profiles_and_contexts(monkeypatch, tmp_path):
    pytest.importorskip("agent.memory_provider")
    from test_service import _bag

    module = _plugin()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    provider = module.ChromaMemoryProvider(embedder=_bag())
    provider.initialize("s1", hermes_home=str(home), platform="cli", agent_context="primary")
    messages = [{"role": "tool", "content": "tool-result-stay"}]
    snapshot = json.loads(json.dumps(messages))
    provider.sync_turn(
        "authentication decision alpha",
        "use token rotation",
        session_id="s1",
        messages=messages,
    )
    text = provider.prefetch("authentication decision", session_id="s1")
    assert "token rotation" in text
    assert provider.recall_status() is not None
    assert provider.recall_status().count >= 1
    assert messages == snapshot
    provider.shutdown()

    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(other))
    second = module.ChromaMemoryProvider(embedder=_bag())
    second.initialize("s2", hermes_home=str(other), platform="cli", agent_context="primary")
    quiet = second.prefetch("authentication decision", session_id="s2")
    assert "token rotation" not in quiet
    second.shutdown()


def test_subagent_and_cron_skip_writes(monkeypatch, tmp_path):
    pytest.importorskip("agent.memory_provider")
    from test_service import _bag, _secrets
    from service import Store

    module = _plugin()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    bag = _bag()
    for context in ("subagent", "cron"):
        provider = module.ChromaMemoryProvider(embedder=bag)
        provider.initialize("s", hermes_home=str(home), platform="cli", agent_context=context)
        provider.sync_turn("secret phrase gamma", "should stay out", session_id="s")
        provider.prefetch("secret phrase gamma", session_id="s")
        provider.shutdown()
    store = Store(home, _secrets({}), embedder=bag)
    assert store.stored_documents() == []
    store.close()


def test_empty_turn_and_prompt_block(monkeypatch, tmp_path):
    pytest.importorskip("agent.memory_provider")
    from test_service import _bag, _secrets
    from service import Store

    module = _plugin()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    provider = module.ChromaMemoryProvider(embedder=_bag())
    provider.initialize("s", hermes_home=str(home), platform="cli")
    assert provider._agent_context == "primary"
    provider.sync_turn("  ", "\n", session_id="s")
    provider.prefetch("hello", session_id="s")
    store = Store(home, _secrets({}), embedder=provider._embedder)
    assert store.stored_documents() == []
    assert provider.system_prompt_block() == ""
    values = {"collection": "notes", "api_key": "sekret-value", "path": ""}
    provider.save_config(values, str(home))
    text = (home / "plugin-data" / "chroma-memory" / "config.json").read_text(encoding="utf-8")
    assert "sekret-value" not in text
    schema = provider.get_config_schema()
    secrets = [row["env_var"] for row in schema if row.get("secret")]
    assert "CHROMA_API_KEY" in secrets
    assert "CHROMA_MODEL_DOWNLOAD" in secrets
    provider.shutdown()
    store.close()
