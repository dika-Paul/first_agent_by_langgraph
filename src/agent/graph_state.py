from typing import Annotated, TypedDict, Any
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph.message import add_messages


@dataclass
class GraphState:
    query: str = field(default_factory=str)
    messages: Annotated[list[BaseMessage], add_messages] = field(default_factory=list)


class GraphContext(TypedDict):
    chat_model: BaseChatModel
    SYSTEM_MSG: SystemMessage
    tools: dict[str, BaseTool]