import os
from functools import lru_cache

from langchain.chat_models import init_chat_model, BaseChatModel
from langchain_core.tools import BaseTool

from .agent_tool import IPLocateByGaoDe, SearchQueryByBoCha


def get_tool_name(tool_cls: type[BaseTool]) -> str:
    field = tool_cls.model_fields.get("name")
    if field is None or field.default is None:
        raise ValueError(f"{tool_cls.__name__} has no default tool name")
    return str(field.default)


TOOL_CLASSES: tuple[type[BaseTool], ...] = (
    IPLocateByGaoDe,
    SearchQueryByBoCha,
)

TOOL_DICT: dict[str, type[BaseTool]] = {
    get_tool_name(tool_cls): tool_cls
    for tool_cls in TOOL_CLASSES
}

@lru_cache(maxsize=5)
def get_model(model_name: str, model_provider: str) -> BaseChatModel:
    return init_chat_model(
        model=model_name,
        model_provider=model_provider,
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )


@lru_cache(maxsize=3)
def get_tool_dict_by_cache(tools: tuple[str, ...]) -> dict[str, BaseTool]:
    tool_dict = {}
    for tool in tools:
        if tool not in TOOL_DICT:
            raise ValueError(f"tool {tool} not found")
        tool_dict[tool] = TOOL_DICT[tool]()
    return tool_dict


def get_tool_dict(tools: list[str] | tuple[str, ...]) -> dict[str, BaseTool]:
    return get_tool_dict_by_cache(tuple(tools))
