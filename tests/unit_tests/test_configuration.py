from langchain_core.messages import AIMessage, ToolMessage
from langgraph.pregel import Pregel

from agent.graph import chat_model_route, graph
from agent.graph_state import GraphState


def test_graph_is_compiled() -> None:
    assert isinstance(graph, Pregel)


def test_chat_model_route_goes_to_tool_calling() -> None:
    state = GraphState(
        messages=[
            AIMessage(
                content="",
                tool_calls=[{"name": "echo", "args": {"text": "ping"}, "id": "call-1", "type": "tool_call"}],
            )
        ]
    )

    assert chat_model_route(state) == "tool_calling"


def test_chat_model_route_goes_to_end_for_final_ai_message() -> None:
    state = GraphState(messages=[AIMessage(content="final answer")])

    assert chat_model_route(state) == "end"


def test_chat_model_route_returns_chat_model_after_tool_message() -> None:
    state = GraphState(messages=[ToolMessage(content="done", tool_call_id="call-1")])

    assert chat_model_route(state) == "chat_model"
