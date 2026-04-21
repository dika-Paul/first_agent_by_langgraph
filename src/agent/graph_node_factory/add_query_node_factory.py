from typing import Any

from langchain_core.messages import HumanMessage

from ..graph_state import GraphState
from .base_node_factory import BaseSingleCallingNodeFactory


class AddQueryNodeFactory(BaseSingleCallingNodeFactory):

    @staticmethod
    def return_dict(content: Any) -> dict:
        return {
            'messages': HumanMessage(content=content),
        }


    @classmethod
    def function(cls):

        def func(state: GraphState) -> dict:
            query = state.query
            return cls.return_dict(query)

        return func