from typing import Any

from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime

from agent import GraphState, GraphContext
from base_node_factory import BaseMultiCallingNodeFactory


class ChatModelNodeFactory(BaseMultiCallingNodeFactory):

    @staticmethod
    def get_llm_config(runtime: Runtime[GraphContext]) -> tuple[Any, SystemMessage]:
        system_msg = runtime.context.get('SYSTEM_MSG')
        chat_model = runtime.context.get('chat_model')
        tools = runtime.context.get('tools')

        return (
            chat_model.bind_tools([tool for tool in tools.values()]),
            system_msg
        )

    @staticmethod
    def return_dict(context: Any) -> dict:
        return {
            'messages': context,
        }


    @classmethod
    def sync_function(cls):

        def func(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
            chat_model, system_msg = cls.get_llm_config(runtime)
            input_msgs = [system_msg] + state.messages

            resp = chat_model.invoke(input_msgs)

            return cls.return_dict(resp)

        return func

    @classmethod
    def async_function(cls):

        async def func(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
            chat_model, system_msg = cls.get_llm_config(runtime)
            input_msgs = [system_msg] + state.messages

            resp = await chat_model.ainvoke(input_msgs)

            return cls.return_dict(resp)

        return func