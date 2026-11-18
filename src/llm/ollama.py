from collections.abc import AsyncIterator, Iterator
from typing import Any

from langchain_ollama import ChatOllama
from langsmith import traceable
from pydantic import BaseModel

from configs.core import settings
from src.llm.config import _convert_messages
from src.utils.exception import ExternalServiceError
from src.utils.logger import LLM_LOGGER

OLLAMA_CONFIG = {
    "model": settings.llm.OLLAMA_DEFAULT_MODEL,
    "base_url": settings.llm.OLLAMA_BASE_URL,
    "priority": settings.llm.OLLAMA_PRIORITY,
    "num_predict": settings.llm.DEFAULT_MAX_TOKENS,
}


class OllamaProvider:
    """Client for local Ollama generation."""

    def __init__(
        self,
        temperature: float = 0.4,
        num_predict: int = OLLAMA_CONFIG["num_predict"],
        think: bool = False,
        **default_kwargs: Any,
    ) -> None:

        self._validate_connection()

        self.temperature = temperature
        self.num_predict = num_predict
        self.default_kwargs = default_kwargs

        self.llm = ChatOllama(
            model=OLLAMA_CONFIG["model"],
            base_url=OLLAMA_CONFIG["base_url"],
            temperature=self.temperature,
            num_predict=self.num_predict,
            validate_model_on_init=True,
            **self.default_kwargs,
        )

        LLM_LOGGER.info(
            "Ollama Local provider initialized: model=%s",
            OLLAMA_CONFIG["model"],
        )

    def _validate_connection(self) -> None:
        """Ping the local Ollama endpoint to verify it is reachable."""
        import urllib.request

        base_url = settings.llm.OLLAMA_BASE_URL.rstrip("/")
        try:
            with urllib.request.urlopen(base_url, timeout=2.0) as response:
                if response.status != 200:
                    raise Exception(f"Received status code {response.status}")
        except Exception as e:
            raise ExternalServiceError(
                message="Local Ollama service is not running or unreachable.",
                payload={"base_url": base_url, "original_error": str(e)},
            ) from e

    @traceable(run_type="llm", name="Ollama_generate")
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
                message=f"Local Ollama generation failed: {str(e)}",
                payload={"model": OLLAMA_CONFIG["model"]},
            ) from e

    @traceable(run_type="llm", name="Ollama_generate_structured")
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
                message=f"Local Ollama structured generation failed: {str(e)}",
                payload={"model": OLLAMA_CONFIG["model"]},
            ) from e

    @traceable(run_type="llm", name="Ollama_agenerate")
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
                message=f"Local Ollama async generation failed: {str(e)}",
                payload={"model": OLLAMA_CONFIG["model"]},
            ) from e

    @traceable(run_type="llm", name="Ollama_agenerate_structured")
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
                message=f"Local Ollama async structured generation failed: {str(e)}",
                payload={"model": OLLAMA_CONFIG["model"]},
            ) from e

    @traceable(run_type="llm", name="Ollama_stream")
    def stream(
        self,
        messages: list[dict[str, str]],
    ) -> Iterator[str]:
        """Stream response chunks in real-time."""
        formatted_messages = _convert_messages(messages)
        try:
            for chunk in self.llm.stream(formatted_messages):
                if chunk.content:
                    yield chunk.content  # type: ignore
        except Exception as e:
            raise ExternalServiceError(
                message=f"Local Ollama streaming interrupted: {str(e)}",
                payload={"model": OLLAMA_CONFIG["model"]},
            ) from e

    @traceable(run_type="llm", name="Ollama_astream")
    async def astream(
        self,
        messages: list[dict[str, str]],
    ) -> AsyncIterator[str]:
        """Asynchronously stream response chunks in real-time."""
        formatted_messages = _convert_messages(messages)
        try:
            async for chunk in self.llm.astream(formatted_messages):
                if chunk.content:
                    yield chunk.content  # type: ignore
        except Exception as e:
            raise ExternalServiceError(
                message=f"Local Ollama async streaming interrupted: {str(e)}",
                payload={"model": OLLAMA_CONFIG["model"]},
            ) from e
