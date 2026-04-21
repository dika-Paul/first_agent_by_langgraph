import asyncio
from typing import Any

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime, get_runtime

from ..graph_state import GraphContext, GraphState
from .base_node_factory import BaseMultiCallingNodeFactory


class ToolCallNodeFactory(BaseMultiCallingNodeFactory):

    @staticmethod
    def get_tool_calls(state: GraphState) -> list[ToolCall]:
        ai_message = state.messages[-1]
        if not isinstance(ai_message, AIMessage):
            raise ValueError(f"错误的消息类型，期望：AIMessage，实际：{type(ai_message)}")

        tool_calls = ai_message.tool_calls
        return tool_calls

    @staticmethod
    def get_tool(
            runtime: Runtime[GraphContext],
            tool_call: ToolCall
    ) -> tuple[BaseTool | None, str, dict[str, Any], str | None]:
        tool_name = tool_call.get('name')
        tool_args = tool_call.get('args')
        tool_id = tool_call.get('id')

        tool = runtime.context.get('tools').get(tool_name)

        return tool, tool_name, tool_args, tool_id

    @staticmethod
    def return_dict(content: Any) -> dict:
        return {
            'messages': content,
        }


    @classmethod
    def sync_function(cls):

        def func(state: GraphState) -> dict:
            runtime = get_runtime(GraphContext)
            tool_calls  = cls.get_tool_calls(state)

            if not tool_calls:
                return {}

            tool_messages = []
            for tool_call in tool_calls:
                tool, tool_name,tool_args, tool_id = cls.get_tool(runtime, tool_call)

                if not tool:
                    tool_message = ToolMessage(
                            content=f'工具不存在：{tool_name}',
                            tool_call_id=tool_id,
                            name=tool_name,
                            status='error',
                        )
                else:
                    try:
                        tool_message = ToolMessage(
                            content=str(tool.invoke(tool_args)),
                            tool_call_id=tool_id,
                            name=tool_name,
                            status='success',
                        )
                    except Exception as e:
                        tool_message = ToolMessage(
                            content=f'捕获到异常{e}',
                            tool_call_id=tool_id or '未知工具请求',
                            name=tool_name,
                            status='error',
                        )

                tool_messages.append(tool_message)

            return cls.return_dict(tool_messages)

        return func

    @classmethod
    def async_function(cls):

        async def func(state: GraphState) -> dict:
            runtime = get_runtime(GraphContext)
            tool_calls = cls.get_tool_calls(state)

            if not tool_calls:
                return {}

            semaphore = asyncio.Semaphore(6)

            async def build_tool_message(tool_call: ToolCall):
                tool, tool_name,tool_args, tool_id = cls.get_tool(runtime, tool_call)
                if not tool:
                    return ToolMessage(
                        content=f'工具不存在：{tool_name}',
                        tool_call_id=tool_id,
                        name=tool_name,
                        status='error',
                    )

                async with semaphore:
                    try:
                        resp = await tool.ainvoke(tool_args)
                        tool_message = ToolMessage(
                            content=str(resp),
                            tool_call_id=tool_id,
                            name=tool_name,
                            status='success',
                        )
                    except Exception as e:
                        tool_message = ToolMessage(
                            content=f'捕获到异常{e}',
                            tool_call_id=tool_id,
                            name=tool_name,
                            status='error',
                        )

                return tool_message

            result = await asyncio.gather(
                *(build_tool_message(tool_call) for tool_call in tool_calls),
            )

            tool_messages = [tool_message for tool_message in result if tool_message]
            return cls.return_dict(tool_messages)

        return func
