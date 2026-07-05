from typing import Any, AsyncIterator, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langsmith import traceable

from app.core.config import LLM_PROVIDERS, settings
from src.graphs.tools import tools
from src.llm.base import BaseLLMProvider
from src.llm.config import ProviderConfig
from src.utils.exception import ExternalServiceError
from src.utils.logger import LLM_LOGGER


def _convert_messages(messages: List[Any]) -> List[Any]:
    lc_messages = []
    for m in messages:
        if isinstance(m, BaseMessage):
            lc_messages.append(m)
        elif isinstance(m, dict):
            role = m.get("role", "user").lower()
            content = m.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=str(m)))
    return lc_messages

class OpenRouterProvider(BaseLLMProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._config = config
        self.endpoint = "https://openrouter.ai/api/v1"
        
        self.llm = ChatOpenAI(
            model=self._config.model,
            api_key=self._config.api_key,  # type: ignore
            base_url=self.endpoint,
            timeout=settings.llm.TIMEOUT,
            max_retries=settings.llm.MAX_RETRIES,
        ).bind_tools(tools)
        
        LLM_LOGGER.info(
            "OpenRouter provider initialized: model=%s, endpoint=%s",
            self._config.model,
            self.endpoint,
        )

    @traceable(run_type="llm", name="OpenRouter")
    async def generate_stream(self, messages: List[Dict], **kwargs) -> AsyncIterator[Any]:
        lc_messages = _convert_messages(messages)
        try:
            async for chunk in self.llm.astream(lc_messages, **kwargs):
                yield chunk
        except Exception as e:
            LLM_LOGGER.error(
                "OpenRouter error during generate_stream: %s", e, exc_info=True
            )
            raise ExternalServiceError(f"OpenRouter generation failed: {str(e)}")

    @traceable(run_type="llm", name="OpenRouter_Structured")
    async def generate_structured(self, messages: List[Dict], schema: type, **kwargs):
        lc_messages = _convert_messages(messages)
        parser = PydanticOutputParser(pydantic_object=schema)
        
        if lc_messages and isinstance(lc_messages[-1], HumanMessage):
            lc_messages[-1].content = f"{lc_messages[-1].content}\n\n{parser.get_format_instructions()}"
        else:
            lc_messages.append(HumanMessage(content=parser.get_format_instructions()))
            
        try:
            response = await self.llm.ainvoke(lc_messages, **kwargs)
            return parser.invoke(response)
        except Exception as e:
            LLM_LOGGER.error("OpenRouter error during generate_structured: %s", e, exc_info=True)
            raise ExternalServiceError(f"OpenRouter structured generation failed: {str(e)}")

    async def is_available(self) -> bool:
        try:
            async for _ in self.generate_stream([{"role": "user", "content": "ping"}], max_tokens=1):
                break
            LLM_LOGGER.info("OpenRouter is available")
            return True
        except Exception as e:
            LLM_LOGGER.warning("OpenRouter unavailable: %s", e)
            return False

class GeminiProvider(BaseLLMProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._config = config
        self.llm = ChatGoogleGenerativeAI(
            model=self._config.model,
            google_api_key=self._config.api_key,
            timeout=settings.llm.TIMEOUT,
            max_retries=settings.llm.MAX_RETRIES,
        ).bind_tools(tools)
       
        LLM_LOGGER.info("Gemini provider initialized: model=%s", self._config.model)

    @traceable(run_type="llm", name="Gemini")
    async def generate_stream(self, messages: List[Dict], **kwargs) -> AsyncIterator[Any]:
        lc_messages = _convert_messages(messages)
        try:
            async for chunk in self.llm.astream(lc_messages, **kwargs):
                yield chunk
        except Exception as e:
            LLM_LOGGER.error("Gemini error during generate_stream: %s", e, exc_info=True)
            raise ExternalServiceError(f"Gemini generation failed: {str(e)}")

    @traceable(run_type="llm", name="Gemini_Structured")
    async def generate_structured(self, messages: List[Dict], schema: type, **kwargs):
        lc_messages = _convert_messages(messages)
        try:
            structured_llm = self.llm.with_structured_output(schema) # type: ignore
            return await structured_llm.ainvoke(lc_messages, **kwargs)
        except Exception as e:
            LLM_LOGGER.error("Gemini error during generate_structured: %s", e, exc_info=True)
            raise ExternalServiceError(f"Gemini structured generation failed: {str(e)}")

    async def is_available(self) -> bool:
        try:
            async for _ in self.generate_stream([{"role": "user", "content": "ping"}], max_tokens=1):
                break
            LLM_LOGGER.info("Gemini is available")
            return True
        except Exception as e:
            LLM_LOGGER.warning("Gemini unavailable: %s", e)
            return False

class OllamaProvider(BaseLLMProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._config = config
        self.endpoint = "https://ollama.com/v1"
        self.llm = ChatOpenAI(
            model=self._config.model,
            api_key=self._config.api_key or "ollama", # type: ignore
            base_url=self.endpoint,
            timeout=settings.llm.TIMEOUT,
            max_retries=settings.llm.MAX_RETRIES,
        ).bind_tools(tools)
        
        LLM_LOGGER.info(
            "Ollama provider initialized (ChatOpenAI wrapper): model=%s, endpoint=%s",
            self._config.model,
            self.endpoint,
        )

    @traceable(run_type="llm", name="Ollama")
    async def generate_stream(self, messages: List[Dict], **kwargs) -> AsyncIterator[Any]:
        lc_messages = _convert_messages(messages)
        
        try:
            async for chunk in self.llm.astream(lc_messages, **kwargs):
                yield chunk
        except Exception as e:
            LLM_LOGGER.error("Ollama error during generate_stream: %s", e, exc_info=True)
            raise ExternalServiceError(f"Ollama generation failed: {str(e)}")

    @traceable(run_type="llm", name="Ollama_Structured")
    async def generate_structured(self, messages: List[Dict], schema: type, **kwargs):
        lc_messages = _convert_messages(messages)
        parser = PydanticOutputParser(pydantic_object=schema)
        
        if lc_messages and isinstance(lc_messages[-1], HumanMessage):
            lc_messages[-1].content = f"{lc_messages[-1].content}\n\n{parser.get_format_instructions()}"
        else:
            lc_messages.append(HumanMessage(content=parser.get_format_instructions()))
            
        try:
            response = await self.llm.ainvoke(lc_messages, **kwargs)
            return parser.invoke(response)
        except Exception as e:
            LLM_LOGGER.error("Ollama error during generate_structured: %s", e, exc_info=True)
            raise ExternalServiceError(f"Ollama structured generation failed: {str(e)}")

    async def is_available(self) -> bool:
        try:
            async for _ in self.generate_stream([{"role": "user", "content": "ping"}]):
                break
            LLM_LOGGER.info("Ollama is available")
            return True
        except Exception as e:
            LLM_LOGGER.warning("Ollama unavailable: %s", e)
            return False

class OllamaLocalProvider(BaseLLMProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._config = config
        self.llm = ChatOllama(
            model=self._config.model,
            base_url=self._config.base_url,
            temperature=settings.llm.TEMPERATURE,
            num_predict=settings.llm.MAX_TOKENS,
            client_kwargs={"timeout": settings.llm.TIMEOUT},
        ).bind_tools(tools)

        LLM_LOGGER.info(
            "Local Ollama provider initialized: model=%s, base_url=%s",
            self._config.model,
            self._config.base_url,
        )

    @traceable(run_type="llm", name="Ollama_Local")
    async def generate_stream(self, messages: List[Dict], **kwargs) -> AsyncIterator[Any]:
        lc_messages = _convert_messages(messages)

        try:
            async for chunk in self.llm.astream(lc_messages, **kwargs):
                yield chunk
        except Exception as e:
            LLM_LOGGER.error("Local Ollama error during generate_stream: %s", e, exc_info=True)
            raise ExternalServiceError(f"Local Ollama generation failed: {str(e)}")

    @traceable(run_type="llm", name="Ollama_Local_Structured")
    async def generate_structured(self, messages: List[Dict], schema: type, **kwargs):
        lc_messages = _convert_messages(messages)

        try:
            structured_llm = self.llm.with_structured_output(schema)  # type: ignore
            return await structured_llm.ainvoke(lc_messages, **kwargs)
        except Exception as e:
            LLM_LOGGER.error("Local Ollama error during generate_structured: %s", e, exc_info=True)
            raise ExternalServiceError(f"Local Ollama structured generation failed: {str(e)}")

    async def is_available(self) -> bool:
        try:
            async for _ in self.generate_stream([{"role": "user", "content": "ping"}]):
                break
            LLM_LOGGER.info("Local Ollama is available")
            return True
        except Exception as e:
            LLM_LOGGER.warning("Local Ollama unavailable: %s", e)
            return False


def _create_provider(p_dict: dict):
    cfg = ProviderConfig(
        name=p_dict["name"],
        model=p_dict["model"],
        api_key=p_dict.get("api_key"),
        base_url=p_dict.get("base_url"),
        priority=p_dict.get("priority", 99),
    )

    provider_name = cfg.name.strip().lower().replace("_", " ")

    if provider_name == "openrouter":
        return OpenRouterProvider(cfg)
    elif provider_name == "gemini":
        return GeminiProvider(cfg)
    elif provider_name == "ollama":
        return OllamaProvider(cfg)
    elif provider_name == "ollama local":
        return OllamaLocalProvider(cfg)
    else:
        raise ValueError(f"Unknown provider name: {cfg.name}")


sorted_provider_dicts = sorted(LLM_PROVIDERS, key=lambda x: x.get("priority", 99))
providers_list = [_create_provider(p) for p in sorted_provider_dicts]

fast_providers = providers_list
slow_providers = providers_list
