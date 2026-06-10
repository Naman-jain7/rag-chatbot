from abc import ABC, abstractmethod
from typing import AsyncIterator, Any
from src.llm.config import ProviderConfig


class BaseLLMProvider(ABC):
    def __init__(self, config: ProviderConfig) -> None:
        pass

    @property
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
    def config(self) -> ProviderConfig:  # type:ignore
        pass
