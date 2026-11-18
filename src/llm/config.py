from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    name: str
    model: str
    api_key: str | None
    base_url: str | None = None
    priority: int       # lower = tried first
    timeout: float      # per-provider request timeout in seconds


def _convert_messages(messages: list[Any]) -> list[BaseMessage]:
    """Convert a list of dicts or LangChain messages into LangChain message objects."""
    formatted: list[BaseMessage] = []

    for msg in messages:
        if isinstance(msg, BaseMessage):
            formatted.append(msg)
        elif isinstance(msg, dict):
            role = msg.get("role", "user").lower()
            content = msg.get("content", "")
            if role == "system":
                formatted.append(SystemMessage(content=content))
            elif role == "assistant":
                formatted.append(AIMessage(content=content))
            else:
                formatted.append(HumanMessage(content=content))
        else:
            formatted.append(HumanMessage(content=str(msg)))

    return formatted
