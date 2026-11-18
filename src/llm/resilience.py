# from collections.abc import AsyncIterator, Iterator
# from typing import Any

# from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
# from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI
# from langsmith import traceable
# from pydantic import BaseModel

# from configs.core import settings

# from src.utils.exception import ExternalServiceError
# from src.utils.logger import LLM_LOGGER

# class ResilientLLMManager:
#     """
#     Leverages LangChain's native runnable architecture to automatically
#     fallback across multiple underlying provider chains.
#     """

#     def __init__(self, primary_provider, backup_providers: list):
#         main_chain = primary_provider.llm
#         fallback_chains = [p.llm for p in backup_providers]

#         self.resilient_llm = main_chain.with_fallbacks(fallbacks=fallback_chains)

#     async def astream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
#         formatted_messages = _convert_messages(messages)

#         async for chunk in self.resilient_llm.astream(formatted_messages):
#             if chunk.content:
#                 yield chunk.content

# # ============================================================================================

# llm = ResilientLLMManager(
#     primary_provider=OllamaLocalProvider(),
#     backup_providers=[OllamaProvider(), OpenRouterProvider()],
# )
