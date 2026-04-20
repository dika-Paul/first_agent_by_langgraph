from typing import Annotated, TypedDict, Any
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages


def add_queries(old: Any, new: Any) -> list:
    if old is None:
        old = []
    if new is None:
        new = []
    if not isinstance(old, list):
        old = [old]
    if not isinstance(new, list):
        new = [new]
    return old + new


@dataclass
class GraphState:
    queries: Annotated[list[str], add_queries] = field(default_factory=list)
    messages: Annotated[list[BaseMessage], add_messages] = field(default_factory=list)


class GraphContext(TypedDict):
    chat_model: BaseChatModel
    SYSTEM_MSG: SystemMessage
    tools: dict[str, BaseTool]