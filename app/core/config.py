from dotenv import load_dotenv
from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Any, Annotated

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]


class LLMSettings(BaseSettings):
    """LLM and Provider Configurations."""

    OPENROUTER_PROVIDER_NAME: Annotated[str, Field(default='openrouter', validation_alias="OPENROUTER_PROVIDER_NAME")]
    OPENROUTER_MODEL_NAME: Annotated[str, Field(..., validation_alias="OPENROUTER_MODEL_NAME")]
    OPENROUTER_API_KEY: Annotated[Optional[str], Field(..., validation_alias="OPENROUTER_API_KEY")]
    OPENROUTER_PRIORITY: Annotated[int, Field(..., validation_alias="OPENROUTER_PRIORITY")]

    GEMINI_PROVIDER_NAME: Annotated[str, Field(default='gemini', validation_alias="GEMINI_PROVIDER_NAME")]
    GEMINI_MODEL_NAME: Annotated[str, Field(..., validation_alias="GEMINI_MODEL_NAME")]
    GOOGLE_API_KEY: Annotated[Optional[str], Field(..., validation_alias="GOOGLE_API_KEY")]
    GEMINI_PRIORITY: Annotated[int, Field(..., validation_alias="GEMINI_PRIORITY")]

    OLLAMA_PROVIDER_NAME: Annotated[str, Field(default='ollama', validation_alias="OLLAMA_PROVIDER_NAME")]
    OLLAMA_MODEL_NAME: Annotated[str, Field(..., validation_alias="OLLAMA_MODEL_NAME")]
    OLLAMA_API_KEY: Annotated[Optional[str], Field(..., validation_alias="OLLAMA_API_KEY")]
    OLLAMA_PRIORITY: Annotated[int, Field(..., validation_alias="OLLAMA_PRIORITY")]

    OLLAMA_LOCAL_PROVIDER_NAME: str = Field(..., validation_alias="OLLAMA_LOCAL_PROVIDER_NAME")
    OLLAMA_LOCAL_MODEL_NAME: str = Field(..., validation_alias="OLLAMA_LOCAL_MODEL_NAME")
    OLLAMA_LOCAL_BASE_URL: str = Field(default="http://localhost:11434", validation_alias="OLLAMA_LOCAL_BASE_URL")
    OLLAMA_LOCAL_PRIORITY: Annotated[int, Field(..., validation_alias="OLLAMA_LOCAL_PRIORITY")]

    TIMEOUT: Annotated[int, Field(default=1, validation_alias="TIMEOUT")]
    MAX_RETRIES: Annotated[int, Field(default=1, validation_alias="MAX_RETRIES")]
    CIRCUIT_BREAKER_THRESHOLD: Annotated[int, Field(default=3, validation_alias="CIRCUIT_BREAKER_THRESHOLD")]
    CIRCUIT_BREAKER_COOLDOWN: Annotated[int, Field(default=30, validation_alias="CIRCUIT_BREAKER_COOLDOWN")]
    MAX_TOKENS: Annotated[int, Field(default=1024, validation_alias="MAX_TOKENS")]
    TEMPERATURE: Annotated[float, Field(default=0.3, validation_alias="TEMPERATURE")]

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


class WhisperSettings(BaseSettings):
    """
    Configuration for the faster-whisper transcription model.

    WHISPER_MODEL_SIZE   – Model variant: tiny | base | small | medium | large-v3
                           Larger models are more accurate but slower and use more RAM.
    WHISPER_COMPUTE_TYPE – Quantisation: int8 (fast, CPU) | float16 (GPU) | float32
    WHISPER_DEVICE       – 'auto' resolves to 'cuda' if a GPU is available, else 'cpu'.
    """
    WHISPER_MODEL_SIZE:   str = Field(default="base", validation_alias="WHISPER_MODEL_SIZE")
    WHISPER_COMPUTE_TYPE: str = Field(default="int8", validation_alias="WHISPER_COMPUTE_TYPE")
    WHISPER_DEVICE:       str = Field(default="auto", validation_alias="WHISPER_DEVICE")


class EmbeddingSettings(BaseSettings):
    HUGGINGFACEHUB_API_TOKEN: str = Field(..., validation_alias="HUGGINGFACEHUB_API_TOKEN")
    EMBEDDING_MODEL: Annotated[str, Field(validation_alias="EMBEDDING_MODEL")]
    EMBEDDING_DIMENSION: Annotated[int, Field(validation_alias="EMBEDDING_DIMENSION")]
    EMBEDDINGS_TABLE_NAME:Annotated[str, Field(..., validation_alias='EMBEDDINGS_TABLE_NAME')]
    RERANKER_MODEL: str = Field(..., validation_alias="RERANKER_MODEL")

    CHUNK_SIZE: Annotated[int, Field(..., validation_alias="CHUNK_SIZE")]
    CHUNK_OVERLAP: Annotated[int, Field(..., validation_alias="CHUNK_OVERLAP")]

    @model_validator(mode="after")
    def validate_overlap(self) -> "EmbeddingSettings":
        if self.CHUNK_SIZE <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.CHUNK_OVERLAP < 0:
            raise ValueError("chunk_overlap cannot be < 0")
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("chunk_overlap must be < chunk_size.")
        return self

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
    SECRET_KEY: str = Field(..., validation_alias='SECRET_KEY')

    DB_PATH: str = Field(..., validation_alias="DB_PATH")
    VECTOR_DB_PATH: str = Field(..., validation_alias="VECTOR_DB_PATH")

    SYS_PROMPTS_PATH: str = Field(..., validation_alias="SYS_PROMPTS_PATH")

    LANGCHAIN_PROJECT: str = Field(..., validation_alias="LANGCHAIN_PROJECT")

class Settings(BaseSettings):
    """Master Settings Object."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm: LLMSettings = LLMSettings()  # type: ignore
    embedding: EmbeddingSettings = EmbeddingSettings()  # type: ignore
    db: DatabaseConfig = DatabaseConfig()  # type: ignore
    services: ServicesSettings = ServicesSettings()  # type: ignore
    app: AppConfig = AppConfig()  # type: ignore
    whisper: WhisperSettings = WhisperSettings()  # type: ignore


settings = Settings()

LLM_PROVIDERS = [
    {
        "name": settings.llm.GEMINI_PROVIDER_NAME,
        "model": settings.llm.GEMINI_MODEL_NAME,
        "api_key": settings.llm.GOOGLE_API_KEY,
        "priority": settings.llm.GEMINI_PRIORITY,
    },
    {
        "name": settings.llm.OPENROUTER_PROVIDER_NAME,
        "model": settings.llm.OPENROUTER_MODEL_NAME,
        "api_key": settings.llm.OPENROUTER_API_KEY,
        "priority": settings.llm.OPENROUTER_PRIORITY,
    },
    {
        "name": settings.llm.OLLAMA_PROVIDER_NAME,
        "model": settings.llm.OLLAMA_MODEL_NAME,
        "api_key": settings.llm.OLLAMA_API_KEY,
        "priority": settings.llm.OLLAMA_PRIORITY,
    },
    {
        "name": settings.llm.OLLAMA_LOCAL_PROVIDER_NAME,
        "model": settings.llm.OLLAMA_LOCAL_MODEL_NAME,
        "api_key": None,
        "base_url": settings.llm.OLLAMA_LOCAL_BASE_URL,
        "priority": settings.llm.OLLAMA_LOCAL_PRIORITY,
    },
]

RAW_DB_PATH = BASE_DIR / settings.app.DB_PATH
VECTOR_DB_PATH = BASE_DIR / settings.app.VECTOR_DB_PATH
