from abc import ABC, abstractmethod
from typing import Any
from collections.abc import AsyncIterator
from src.llm.config import ProviderConfig


class BaseLLMProvider(ABC):
    @abstractmethod
    def __init__(self, config: ProviderConfig) -> None:
        pass

    @property
    @abstractmethod
    def name(self) -> str:  # type:ignore
        pass

    @abstractmethod
    async def generate_stream(self, messages: list[dict], **kwargs) -> AsyncIterator[Any]:  # type:ignore
        pass

    @abstractmethod
    async def generate_structured(self, messages: list[dict], schema: type, **kwargs):  # type:ignore
        pass

    @abstractmethod
    async def is_available(self) -> bool:  # quick health ping
        pass

    @property
    @abstractmethod
    def config(self) -> ProviderConfig:  # type:ignore
        pass
