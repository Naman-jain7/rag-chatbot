import pybreaker
from tenacity import AsyncRetrying, wait_exponential, stop_after_attempt
from typing import List, Dict, AsyncIterator, Type, Any
from pydantic import BaseModel

from src.llm.base import BaseLLMProvider
from src.llm.config import RetryConfig, CircuitBreakerConfig
from src.utils.logger import LLM_LOGGER
from src.utils.exception import AllProvidersExhaustedError

class ResilientLLMManager:
    """
    An asynchronous resilience wrapper that sequentially executes a tasks-fallback chain 
    across multiple LLM providers until one succeeds.
    """
    
    def __init__(self, providers: List[BaseLLMProvider]) -> None:
        """
        Initializes the routing manager and sets up individual circuit breakers
        for each LLM provider based on their respective threshold and cooldown configurations.

        Args:
            providers: A list of configured LLM provider instances.

        Raises:
            ValueError: If the providers list is empty.
        """
        if not providers:
            raise ValueError("At least one provider must be specified.")
            
        self.providers = providers
        self.breakers: Dict[str, pybreaker.CircuitBreaker] = {}
        
        cb_config = CircuitBreakerConfig()
        
        for provider in self.providers:
            p_config = getattr(provider, '_config', None)
            
            fail_max = getattr(p_config, 'circuit_threshold', cb_config.fail_threshold)
            reset_timeout = getattr(p_config, 'circuit_cooldown', cb_config.cooldown)
            
            provider_name = getattr(p_config, 'name', str(id(provider)))
            
            breaker = pybreaker.CircuitBreaker(fail_max=fail_max,reset_timeout=reset_timeout)
            self.breakers[provider_name] = breaker
            LLM_LOGGER.info("Initialized CircuitBreaker for provider '%s' (fail_max=%d, reset_timeout=%d)", 
                        provider_name, fail_max, reset_timeout)

    async def generate_stream(self, messages: List[Dict], **kwargs) -> AsyncIterator[Any]:
        """
        Streams responses from the highest-priority available LLM provider, employing
        exponential backoff retries and fallback routing if a provider fails or its circuit is open.

        Args:
            messages: A list of message dictionaries representing the conversation history.
            **kwargs: Additional provider-specific generation parameters.

        Yields:
            Any: Individual chunks of the generated stream from the successful provider.

        Raises:
            AllProvidersExhaustedError: If all configured providers fail or are skipped.
        """
        retry_config = RetryConfig()
        
        for provider in self.providers:
            provider_name = getattr(provider._config, 'name', str(id(provider))) # type: ignore
            breaker = self.breakers[provider_name]
            
            if breaker.current_state == "open": # type: ignore
                LLM_LOGGER.warning("Circuit is OPEN for provider '%s'. Skipping to next provider.", provider_name)
                continue
                
            try:
                # We encapsulate the async generator consumption in an async function
                async def _call():
                    chunks = []
                    async for chunk in provider.generate_stream(messages, **kwargs): # type: ignore
                        chunks.append(chunk)
                    return chunks

                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(retry_config.max_attempts),
                    wait=wait_exponential(multiplier=retry_config.min_wait, max=retry_config.max_wait),
                    reraise=True
                ):
                    with attempt:
                        chunks = await _call()
                
                # If we get here, the execution was successful. Tell pybreaker.
                try:
                    breaker.call(lambda: None)
                except pybreaker.CircuitBreakerError:
                    pass
                
                for chunk in chunks:
                    yield chunk
                
                return 

            except Exception as e:
                # Tell pybreaker about the failure
                def dummy_fail():
                    raise e
                try:
                    breaker.call(dummy_fail)
                except Exception:
                    pass
                    
                LLM_LOGGER.error("Provider '%s' failed to generate_stream after all retries. Error: %s", provider_name, str(e))
                # Flow naturally continues to the next fallback provider

        LLM_LOGGER.error("Terminal Failure: All LLM providers exhausted for generate_stream.")
        raise AllProvidersExhaustedError("All available LLM providers failed or were skipped due to open circuits.")

    async def generate_structured(self, messages: List[Dict], schema: Type[BaseModel], **kwargs) -> Any:
        retry_config = RetryConfig()
        
        for provider in self.providers:
            provider_name = getattr(provider._config, 'name', str(id(provider))) # type: ignore
            breaker = self.breakers[provider_name]
            
            if breaker.current_state == "open": # type: ignore
                LLM_LOGGER.warning("Circuit is OPEN for provider '%s'. Skipping to next provider.", provider_name)
                continue
                
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(retry_config.max_attempts),
                    wait=wait_exponential(multiplier=retry_config.min_wait, max=retry_config.max_wait),
                    reraise=True
                ):
                    with attempt:
                        result = await provider.generate_structured(messages, schema, **kwargs)
                
                # Tell pybreaker it succeeded
                try:
                    breaker.call(lambda: None)
                except pybreaker.CircuitBreakerError:
                    pass
                    
                return result

            except Exception as e:
                # Tell pybreaker it failed
                def dummy_fail():
                    raise e
                try:
                    breaker.call(dummy_fail)
                except Exception:
                    pass
                    
                LLM_LOGGER.error("Provider '%s' failed to generate_structured after all retries. Error: %s", provider_name, str(e))
                # Flow naturally continues to the next fallback provider

        LLM_LOGGER.error("Terminal Failure: All LLM providers exhausted for generate_structured.")
        raise AllProvidersExhaustedError("All available LLM providers failed or were skipped due to open circuits.")