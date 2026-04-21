from abc import ABC, abstractmethod
from typing import Any

from langchain_core.runnables import RunnableLambda


class BaseNodeFactory(ABC):
    @staticmethod
    @abstractmethod
    def return_dict(content: Any) -> dict:
        pass


    @classmethod
    @abstractmethod
    def graph_node(cls) -> RunnableLambda:
        pass



class BaseSingleCallingNodeFactory(BaseNodeFactory):
    @staticmethod
    @abstractmethod
    def return_dict(content: Any) -> dict:
        pass


    @classmethod
    @abstractmethod
    def function(cls):
        pass


    @classmethod
    def graph_node(cls) -> RunnableLambda:
        return RunnableLambda(cls.function())



class BaseMultiCallingNodeFactory(BaseNodeFactory):
    @staticmethod
    @abstractmethod
    def return_dict(content: Any) -> dict:
        pass

    @classmethod
    @abstractmethod
    def sync_function(cls):
        pass


    @classmethod
    @abstractmethod
    def async_function(cls):
        pass


    @classmethod
    def graph_node(cls) -> RunnableLambda:
        return RunnableLambda(
            func=cls.sync_function(),
            afunc=cls.async_function()
        )
