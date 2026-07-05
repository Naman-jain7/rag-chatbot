from pydantic import BaseModel
from typing import Optional


class ProviderConfig(BaseModel):
    name: str
    model: str
    api_key: str | None
    base_url: Optional[str] = None
    priority: int           # lower = tried first


class RetryConfig:
    max_attempts: int = 3
    min_wait: float = 1.0  # seconds
    max_wait: float = 60.0
    jitter: float = 2.0


class CircuitBreakerConfig:
    """Fallback class if circuit config not provided for any llm"""
    fail_threshold: int = 5
    cooldown: float = 30.0  # seconds before half-open