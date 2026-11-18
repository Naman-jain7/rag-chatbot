from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_openai import ChatOpenAI
from langsmith import traceable
from pydantic import BaseModel

from configs.core import settings
from src.llm.config import _convert_messages
from src.utils.exception import ExternalServiceError
from src.utils.logger import LLM_LOGGER

OLLAMA_CLOUD_CONFIG = {
    "model": settings.llm.OLLAMA_CLOUD_DEFAULT_MODEL,
    "api_key": settings.llm.OLLAMA_CLOUD_API_KEY,
    "priority": settings.llm.OLLAMA_CLOUD_PRIORITY,
    "timeout": settings.llm.OLLAMA_CLOUD_TIMEOUT,
    "max_completion_tokens": settings.llm.DEFAULT_MAX_TOKENS,
    "base_url": "https://ollama.com/v1",
}


class OllamaCloudProvider:
    """Client for Ollama Cloud generation (OpenAI-compatible API)."""

    def __init__(
        self,
        temperature: float = 0.4,
        max_completion_tokens: int = OLLAMA_CLOUD_CONFIG["max_completion_tokens"],
        **default_kwargs: Any,
    ) -> None:

        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.default_kwargs = default_kwargs

        self.llm = ChatOpenAI(
            model=OLLAMA_CLOUD_CONFIG["model"],
            api_key=OLLAMA_CLOUD_CONFIG["api_key"],  # type: ignore
            base_url=OLLAMA_CLOUD_CONFIG["base_url"],
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            timeout=OLLAMA_CLOUD_CONFIG["timeout"],
            **self.default_kwargs,
        )

        LLM_LOGGER.info(
            "Ollama Cloud provider initialized: model=%s",
            OLLAMA_CLOUD_CONFIG["model"],
        )

    @traceable(run_type="llm", name="OllamaCloud_generate")
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
                message=f"Ollama Cloud generation failed: {str(e)}",
                payload={"model": OLLAMA_CLOUD_CONFIG["model"]},
            ) from e

    @traceable(run_type="llm", name="OllamaCloud_generate_structured")
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
                message=f"Ollama Cloud structured generation failed: {str(e)}",
                payload={"model": OLLAMA_CLOUD_CONFIG["model"]},
            ) from e

    @traceable(run_type="llm", name="OllamaCloud_agenerate")
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
                message=f"Ollama Cloud async generation failed: {str(e)}",
                payload={"model": OLLAMA_CLOUD_CONFIG["model"]},
            ) from e

    @traceable(run_type="llm", name="OllamaCloud_agenerate_structured")
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
                message=f"Ollama Cloud async structured generation failed: {str(e)}",
                payload={"model": OLLAMA_CLOUD_CONFIG["model"]},
            ) from e

    @traceable(run_type="llm", name="OllamaCloud_stream")
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
                message=f"Ollama Cloud streaming interrupted: {str(e)}",
                payload={"model": OLLAMA_CLOUD_CONFIG["model"]},
            ) from e

    @traceable(run_type="llm", name="OllamaCloud_astream")
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
                message=f"Ollama Cloud async streaming interrupted: {str(e)}",
                payload={"model": OLLAMA_CLOUD_CONFIG["model"]},
            ) from e