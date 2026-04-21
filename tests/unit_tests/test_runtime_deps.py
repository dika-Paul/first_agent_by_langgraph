from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel

import agent.runtime_deps as runtime_deps


class DummyToolInput(BaseModel):
    value: str


class DummyTool(BaseTool):
    name: str = "dummy"
    description: str = "Dummy tool for cache tests."
    args_schema: type[BaseModel] = DummyToolInput

    def _run(self, value: str) -> str:
        return value

    async def _arun(self, value: str) -> str:
        return value


def test_get_model_is_cached(monkeypatch) -> None:
    runtime_deps.get_model.cache_clear()
    calls: list[tuple[str, str, str | None, str | None]] = []

    class DummyModel:
        pass

    def fake_init_chat_model(
        *,
        model: str,
        model_provider: str,
        api_key: str | None,
        base_url: str | None,
    ) -> Any:
        calls.append((model, model_provider, api_key, base_url))
        return DummyModel()

    monkeypatch.setattr(runtime_deps, "init_chat_model", fake_init_chat_model)

    first = runtime_deps.get_model("fake-model", "fake-provider")
    second = runtime_deps.get_model("fake-model", "fake-provider")

    assert first is second
    assert calls == [("fake-model", "fake-provider", None, None)]


def test_get_tool_dict_accepts_json_style_list_and_is_cached(monkeypatch) -> None:
    runtime_deps.get_tool_dict_by_cache.cache_clear()
    monkeypatch.setattr(runtime_deps, "TOOL_DICT", {"dummy": DummyTool})

    first = runtime_deps.get_tool_dict(["dummy"])
    second = runtime_deps.get_tool_dict(["dummy"])

    assert first is second
    assert set(first) == {"dummy"}
    assert isinstance(first["dummy"], DummyTool)
