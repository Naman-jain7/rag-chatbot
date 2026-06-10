from dotenv import load_dotenv
from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Any

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]


class LLMSettings(BaseSettings):
    """LLM and Provider Configurations."""

    OPENROUTER_PROVIDER_NAME: str = Field(..., validation_alias="OPENROUTER_PROVIDER_NAME")
    OPENROUTER_MODEL_NAME: str = Field(..., validation_alias="OPENROUTER_MODEL_NAME")
    OPENROUTER_API_KEY: Optional[str] = Field(..., validation_alias="OPENROUTER_API_KEY")
    OPENROUTER_PRIORITY: int = Field(..., validation_alias="OPENROUTER_PRIORITY")
    OPENROUTER_TIMEOUT: int = Field(..., validation_alias="OPENROUTER_TIMEOUT")
    OPENROUTER_MAX_ATTEMPTS: int = Field(..., validation_alias="OPENROUTER_MAX_ATTEMPTS")
    OPENROUTER_CIRCUIT_THRESHOLD: int = Field(..., validation_alias="OPENROUTER_CIRCUIT_THRESHOLD")
    OPENROUTER_CIRCUIT_COOLDOWN: int = Field(..., validation_alias="OPENROUTER_CIRCUIT_COOLDOWN")

    GEMINI_PROVIDER_NAME: str = Field(..., validation_alias="GEMINI_PROVIDER_NAME")
    GEMINI_MODEL_NAME: str = Field(..., validation_alias="GEMINI_MODEL_NAME")
    GOOGLE_API_KEY: Optional[str] = Field(None, validation_alias="GOOGLE_API_KEY")
    GEMINI_PRIORITY: int = Field(..., validation_alias="GEMINI_PRIORITY")
    GEMINI_TIMEOUT: int = Field(..., validation_alias="GEMINI_TIMEOUT")
    GEMINI_MAX_ATTEMPTS: int = Field(..., validation_alias="GEMINI_MAX_ATTEMPTS")
    GEMINI_CIRCUIT_THRESHOLD: int = Field(..., validation_alias="GEMINI_CIRCUIT_THRESHOLD")
    GEMINI_CIRCUIT_COOLDOWN: int = Field(..., validation_alias="GEMINI_CIRCUIT_COOLDOWN")

    OLLAMA_PROVIDER_NAME: str = Field(..., validation_alias="OLLAMA_PROVIDER_NAME")
    OLLAMA_MODEL_NAME: str = Field(..., validation_alias="OLLAMA_MODEL_NAME")
    OLLAMA_API_KEY: Optional[str] = Field(None, validation_alias="OLLAMA_API_KEY")
    OLLAMA_PRIORITY: int = Field(..., validation_alias="OLLAMA_PRIORITY")
    OLLAMA_TIMEOUT: int = Field(..., validation_alias="OLLAMA_TIMEOUT")
    OLLAMA_MAX_ATTEMPTS: int = Field(..., validation_alias="OLLAMA_MAX_ATTEMPTS")
    OLLAMA_CIRCUIT_THRESHOLD: int = Field(..., validation_alias="OLLAMA_CIRCUIT_THRESHOLD")
    OLLAMA_CIRCUIT_COOLDOWN: int = Field(..., validation_alias="OLLAMA_CIRCUIT_COOLDOWN")

    MAX_TOKENS: int = Field(..., validation_alias="MAX_TOKENS")
    TEMPERATURE: float = Field(..., validation_alias="TEMPERATURE")

    @model_validator(mode="after")
    def validate_providers(self)->"LLMSettings":
        """
        Ensures that if a provider is intended to be used, both its API Key
        and Model Name are provided. Also guarantees at least one valid provider exists.
        """

        active_providers = []

        if self.OPENROUTER_API_KEY or self.OPENROUTER_MODEL_NAME:
            if not (self.OPENROUTER_API_KEY and self.OPENROUTER_MODEL_NAME):
                raise ValueError("Both OPENROUTER_API_KEY and OPENROUTER_MODEL_NAME must be provided together.")
            active_providers.append("OpenRouter")

        # 2. Check Gemini
        if self.GOOGLE_API_KEY or self.GEMINI_MODEL_NAME:
            if not (self.GOOGLE_API_KEY and self.GEMINI_MODEL_NAME):
                raise ValueError("Both GOOGLE_API_KEY and GEMINI_MODEL_NAME must be provided together.")
            active_providers.append("Gemini")

        # 3. Check Ollama (API Key might genuinely be optional/None for local setups)
        if self.OLLAMA_MODEL_NAME:
            active_providers.append("Ollama")

        # 4. Global Check: Ensure at least one provider is configured
        if not active_providers:
            raise ValueError("At least one LLM provider (OpenRouter, Gemini, or Ollama) must be fully configured.")

        return self


class EmbeddingSettings(BaseSettings):

    HUGGINGFACEHUB_API_TOKEN: str = Field(..., validation_alias="HUGGINGFACEHUB_API_TOKEN")
    EMBEDDING_LLM: str = Field(..., validation_alias="EMBEDDING_LLM")
    EMBEDDING_LLM_DIMENSION: str = Field(..., validation_alias="EMBEDDING_LLM_DIMENSION")

class MemorySettings(BaseSettings):
    MEM0_API_KEY: str = Field(..., validation_alias="MEM0_API_KEY")
    MEM0_PROJECT: str = Field(..., validation_alias="MEM0_PROJECT")
    MEM0_ORG_ID: Optional[str] = Field(None, validation_alias="MEM0_ORG_ID")

class DatabaseConfig(BaseSettings):
    DB_DSN: Optional[str] = Field("", validation_alias="DB_DSN")

    @model_validator(mode="before")
    @classmethod
    def preprocess_empty_strings(cls, values: Any) -> Any:
        if isinstance(values, dict):
            return {
                k: (None if isinstance(v, str) and v.strip() == "" else v)
                for k, v in values.items()
            }
        return values


class ServicesSettings(BaseSettings):
    ALPHAVANTAGE_STOCK_API_KEY: str = Field(..., validation_alias="ALPHAVANTAGE_STOCK_API_KEY")

    CURRENCY_EXCHANGE_URL: str = Field(..., validation_alias="CURRENCY_EXCHANGE_URL")
    CURRENCY_EXCHANGE_API_KEY: str = Field(..., validation_alias="CURRENCY_EXCHANGE_API_KEY")


class AppConfig(BaseSettings):
    APP_NAME: str = Field(..., validation_alias="APP_NAME")
    APP_VERSION: str = Field(..., validation_alias="APP_VERSION")
    DB_PATH: str = Field(..., validation_alias="DB_PATH")
    SYS_PROMPTS_PATH: str = Field(..., validation_alias="SYS_PROMPTS_PATH")


class Settings(BaseSettings):
    """Master Settings Object."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm: LLMSettings = LLMSettings()  # type: ignore
    embedding: EmbeddingSettings = EmbeddingSettings()  # type: ignore
    memory: MemorySettings = MemorySettings() # type:ignore
    db: DatabaseConfig = DatabaseConfig()  # type: ignore
    services: ServicesSettings = ServicesSettings()  # type: ignore
    app_config: AppConfig = AppConfig()  # type: ignore


settings = Settings()

LLM_PROVIDERS = [
    {
        "name": settings.llm.GEMINI_PROVIDER_NAME,
        "model": settings.llm.GEMINI_MODEL_NAME,
        "api_key": settings.llm.GOOGLE_API_KEY,
        "priority": settings.llm.GEMINI_PRIORITY,
        "timeout": settings.llm.GEMINI_TIMEOUT,
        "max_attempts": settings.llm.GEMINI_MAX_ATTEMPTS,
        "circuit_threshold": settings.llm.GEMINI_CIRCUIT_THRESHOLD,
        "circuit_cooldown": settings.llm.GEMINI_CIRCUIT_COOLDOWN,
    },
    {
        "name": settings.llm.OPENROUTER_PROVIDER_NAME,
        "model": settings.llm.OPENROUTER_MODEL_NAME,
        "api_key": settings.llm.OPENROUTER_API_KEY,
        "priority": settings.llm.OPENROUTER_PRIORITY,
        "timeout": settings.llm.OPENROUTER_TIMEOUT,
        "max_attempts": settings.llm.OPENROUTER_MAX_ATTEMPTS,
        "circuit_threshold": settings.llm.OPENROUTER_CIRCUIT_THRESHOLD,
        "circuit_cooldown": settings.llm.OPENROUTER_CIRCUIT_COOLDOWN,
    },
    {
        "name": settings.llm.OLLAMA_PROVIDER_NAME,
        "model": settings.llm.OLLAMA_MODEL_NAME,
        "api_key": settings.llm.OLLAMA_API_KEY,
        "priority": settings.llm.OLLAMA_PRIORITY,
        "timeout": settings.llm.OLLAMA_TIMEOUT,
        "max_attempts": settings.llm.OLLAMA_MAX_ATTEMPTS,
        "circuit_threshold": settings.llm.OLLAMA_CIRCUIT_THRESHOLD,
        "circuit_cooldown": settings.llm.OLLAMA_CIRCUIT_COOLDOWN,
    },
]

RAW_DB_PATH = BASE_DIR / settings.app_config.DB_PATH
