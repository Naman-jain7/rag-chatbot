from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel

from configs.core import settings
from src.llm.config import _convert_messages
from src.utils.exception import ExternalServiceError
from src.utils.logger import LLM_LOGGER

OPENROUTER_CONFIG = {
    "model": settings.llm.OPENROUTER_DEFAULT_MODEL,
    "api_key": settings.llm.OPENROUTER_API_KEY,
    "priority": settings.llm.OPENROUTER_PRIORITY,
    "timeout": settings.llm.OPENROUTER_TIMEOUT,
    "max_completion_tokens": settings.llm.DEFAULT_MAX_TOKENS,
    "base_url": "https://openrouter.ai/api/v1",
}


class OpenRouterProvider:
    """Client for OpenRouter generation."""

    def __init__(
        self,
        temperature: float = 0.4,
        max_completion_tokens: int = OPENROUTER_CONFIG["max_completion_tokens"],
        **default_kwargs: Any,
    ) -> None:

        self._validate_connection()
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.default_kwargs = default_kwargs

        self.llm = ChatOpenAI(
            model=OPENROUTER_CONFIG["model"],
            api_key=OPENROUTER_CONFIG["api_key"],  # type: ignore
            base_url=OPENROUTER_CONFIG["base_url"],
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            timeout=OPENROUTER_CONFIG["timeout"],
            **self.default_kwargs,
        )

        LLM_LOGGER.info(
            "OpenRouter provider initialized: model=%s",
            OPENROUTER_CONFIG["model"],
        )

    def _validate_connection(self) -> None:
        """Ping the OpenRouter endpoint to verify it is reachable."""
        import urllib.request

        ping_url = "https://openrouter.ai/api/v1/models"
        try:
            with urllib.request.urlopen(ping_url, timeout=5.0) as response:
                if response.status != 200:
                    raise Exception(f"Received status code {response.status}")
        except Exception as e:
            raise ExternalServiceError(
                message="OpenRouter service is unreachable.",
                payload={"url": ping_url, "original_error": str(e)},
            ) from e

    @traceable(run_type="llm", name="OpenRouter_generate")
    def generate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Generate a synchronous plain-text answer."""
        formatted_messages = _convert_messages(messages)
        try:
            response = self.llm.invoke(formatted_messages)
            return response.content  # type: ignore
        except Exception as e:
            raise ExternalServiceError(
                message=f"OpenRouter generation failed: {str(e)}",
                payload={"model": OPENROUTER_CONFIG["model"]},
            ) from e

    @traceable(run_type="llm", name="OpenRouter_generate_structured")
    def generate_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
    ) -> BaseModel:
        """Generate a synchronous structured (Pydantic) answer."""
        formatted_messages = _convert_messages(messages)
        try:
            structured_llm = self.llm.with_structured_output(schema)
            return structured_llm.invoke(formatted_messages)  # type: ignore
        except Exception as e:
            raise ExternalServiceError(
                message=f"OpenRouter structured generation failed: {str(e)}",
                payload={"model": OPENROUTER_CONFIG["model"]},
            ) from e

    @traceable(run_type="llm", name="OpenRouter_agenerate")
    async def agenerate(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        """Generate an asynchronous plain-text answer."""
        formatted_messages = _convert_messages(messages)
        try:
            response = await self.llm.ainvoke(formatted_messages)
            return response.content  # type: ignore
        except Exception as e:
            raise ExternalServiceError(
                message=f"OpenRouter async generation failed: {str(e)}",
                payload={"model": OPENROUTER_CONFIG["model"]},
            ) from e

    @traceable(run_type="llm", name="OpenRouter_agenerate_structured")
    async def agenerate_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
    ) -> BaseModel:
        """Generate an asynchronous structured (Pydantic) answer."""
        formatted_messages = _convert_messages(messages)
        try:
            structured_llm = self.llm.with_structured_output(schema)
            return await structured_llm.ainvoke(formatted_messages)  # type: ignore
        except Exception as e:
            raise ExternalServiceError(
                message=f"OpenRouter async structured generation failed: {str(e)}",
                payload={"model": OPENROUTER_CONFIG["model"]},
            ) from e

    @traceable(run_type="llm", name="OpenRouter_stream")
    def stream(
        self,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        """Stream response chunks."""
        formatted_messages = _convert_messages(messages)
        try:
            for chunk in self.llm.stream(formatted_messages):
                if chunk.content:
                    yield chunk.content  # type: ignore
        except Exception as e:
            raise ExternalServiceError(
                message=f"OpenRouter streaming interrupted: {str(e)}",
                payload={"model": OPENROUTER_CONFIG["model"]},
            ) from e

    @traceable(run_type="llm", name="OpenRouter_astream")
    async def astream(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        """Asynchronously stream response chunks."""
        formatted_messages = _convert_messages(messages)
        try:
            async for chunk in self.llm.astream(formatted_messages):
                if chunk.content:
                    yield chunk.content  # type: ignore
        except Exception as e:
            raise ExternalServiceError(
                message=f"OpenRouter async streaming interrupted: {str(e)}",
                payload={"model": OPENROUTER_CONFIG["model"]},
            ) from e
