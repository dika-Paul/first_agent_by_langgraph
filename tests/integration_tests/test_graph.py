from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from agent import graph

pytestmark = pytest.mark.anyio


def make_tool_call(name: str, args: dict[str, Any], tool_call_id: str = "call-1") -> dict[str, Any]:
    return {
        "name": name,
        "args": args,
        "id": tool_call_id,
        "type": "tool_call",
    }


class FakeChatModel(BaseChatModel):
    responses: list[AIMessage] = Field(default_factory=list)
    _calls: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def calls(self) -> list[list[BaseMessage]]:
        return self._calls

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def bind_tools(self, tools, *, tool_choice: str | None = None, **kwargs: Any):
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        self._calls.append(list(messages))
        message = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=message)])


class EchoToolInput(BaseModel):
    text: str


class EchoTool(BaseTool):
    name: str = "echo"
    description: str = "Return a prefixed echo result."
    args_schema: type[BaseModel] = EchoToolInput

    def _run(self, text: str) -> str:
        return f"echo:{text}"

    async def _arun(self, text: str) -> str:
        return f"echo:{text}"


class FailingTool(BaseTool):
    name: str = "failing"
    description: str = "Raise an error for testing."
    args_schema: type[BaseModel] = EchoToolInput

    def _run(self, text: str) -> str:
        raise RuntimeError(f"boom:{text}")

    async def _arun(self, text: str) -> str:
        raise RuntimeError(f"boom:{text}")


def build_context(chat_model: BaseChatModel, tools: dict[str, BaseTool] | None = None) -> dict[str, Any]:
    return {
        "chat_model": chat_model,
        "SYSTEM_MSG": SystemMessage(content="system prompt"),
        "tools": tools or {},
    }


def test_graph_invoke_returns_direct_ai_response() -> None:
    chat_model = FakeChatModel(responses=[AIMessage(content="final answer")])

    result = graph.invoke({"query": "hello"}, context=build_context(chat_model))

    messages = result["messages"]
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "hello"
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content == "final answer"
    assert len(chat_model.calls) == 1
    assert isinstance(chat_model.calls[0][0], SystemMessage)
    assert isinstance(chat_model.calls[0][1], HumanMessage)


async def test_graph_returns_direct_ai_response() -> None:
    chat_model = FakeChatModel(responses=[AIMessage(content="final answer")])

    result = await graph.ainvoke({"query": "hello"}, context=build_context(chat_model))

    messages = result["messages"]
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "hello"
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content == "final answer"
    assert len(chat_model.calls) == 1
    assert isinstance(chat_model.calls[0][0], SystemMessage)
    assert isinstance(chat_model.calls[0][1], HumanMessage)


async def test_graph_completes_tool_call_loop() -> None:
    chat_model = FakeChatModel(
        responses=[
            AIMessage(content="", tool_calls=[make_tool_call("echo", {"text": "ping"})]),
            AIMessage(content="tool handled"),
        ]
    )

    result = await graph.ainvoke(
        {"query": "call the tool"},
        context=build_context(chat_model, {"echo": EchoTool()}),
    )

    messages = result["messages"]
    tool_message = messages[2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.status == "success"
    assert tool_message.content == "echo:ping"
    assert isinstance(messages[-1], AIMessage)
    assert messages[-1].content == "tool handled"
    assert len(chat_model.calls) == 2
    assert isinstance(chat_model.calls[1][-1], ToolMessage)
    assert chat_model.calls[1][-1].content == "echo:ping"


async def test_graph_returns_error_tool_message_when_tool_is_missing() -> None:
    chat_model = FakeChatModel(
        responses=[
            AIMessage(content="", tool_calls=[make_tool_call("missing_tool", {"text": "ping"})]),
            AIMessage(content="missing tool handled"),
        ]
    )

    result = await graph.ainvoke({"query": "missing tool"}, context=build_context(chat_model))

    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.status == "error"
    assert tool_message.name == "missing_tool"
    assert tool_message.tool_call_id == "call-1"
    assert result["messages"][-1].content == "missing tool handled"


async def test_graph_returns_error_tool_message_when_tool_raises() -> None:
    chat_model = FakeChatModel(
        responses=[
            AIMessage(content="", tool_calls=[make_tool_call("failing", {"text": "ping"})]),
            AIMessage(content="tool error handled"),
        ]
    )

    result = await graph.ainvoke(
        {"query": "tool raises"},
        context=build_context(chat_model, {"failing": FailingTool()}),
    )

    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.status == "error"
    assert tool_message.name == "failing"
    assert tool_message.tool_call_id == "call-1"
    assert result["messages"][-1].content == "tool error handled"
