from typing import Annotated, TypedDict
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


@dataclass
class GraphState:
    query: str = field(default_factory=str)
    messages: Annotated[list[BaseMessage], add_messages] = field(default_factory=list)


class GraphContext(TypedDict):
    model_name: str
    model_provider: str
    SYSTEM_MSG: str
    tools: list[str]