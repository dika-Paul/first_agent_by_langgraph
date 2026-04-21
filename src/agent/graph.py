"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.constants import START, END

from .graph_state import GraphState, GraphContext
from .graph_node_factory import *


add_query = AddQueryNodeFactory.graph_node()
chat_model = ChatModelNodeFactory.graph_node()
tool_calling = ToolCallNodeFactory.graph_node()

def chat_model_route(state: GraphState) -> Literal['tool_calling', 'chat_model', 'end']:
    message = state.messages[-1]

    if isinstance(message, AIMessage):
        tool_calls = message.tool_calls
        if tool_calls:
            return 'tool_calling'

        return 'end'

    return 'chat_model'


# Define the graph
graph_builder = StateGraph(GraphState, context_schema=GraphContext)

graph_builder.add_node('add_query', add_query)
graph_builder.add_node('chat_model', chat_model)
graph_builder.add_node('tool_calling', tool_calling)

graph_builder.add_edge(START, 'add_query')
graph_builder.add_edge('add_query', 'chat_model')
graph_builder.add_conditional_edges(
    source='chat_model',
    path=chat_model_route,
    path_map={
        'chat_model': 'chat_model',
        'tool_calling': 'tool_calling',
        'end': END
    }
)
graph_builder.add_edge('tool_calling', 'chat_model')

graph = graph_builder.compile()
