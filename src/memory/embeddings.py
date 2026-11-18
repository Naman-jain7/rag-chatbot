from __future__ import annotations

import asyncio

from langchain_huggingface import HuggingFaceEmbeddings

from configs.core import settings
from src.memory.redis_cache import redis_cache
from src.utils.logger import EMBEDDING_LOGGER, MEMORY_LOGGER

_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings_model() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        EMBEDDING_LOGGER.info(
            "Loading HuggingFaceEmbeddings model %s",
            settings.embedding.EMBEDDING_MODEL,
        )
        _embeddings = HuggingFaceEmbeddings(model_name=settings.embedding.EMBEDDING_MODEL)
    return _embeddings


async def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    cached = await redis_cache.get_embeddings(texts)
    missing_indexes = [i for i, value in enumerate(cached) if value is None]

    if missing_indexes:
        MEMORY_LOGGER.info(
            "Embedding cache miss for %s/%s texts",
            len(missing_indexes),
            len(texts),
        )
        model = await asyncio.to_thread(get_embeddings_model)
        missing_texts = [texts[i] for i in missing_indexes]
        new_embeddings = await asyncio.to_thread(model.embed_documents, missing_texts)
        await redis_cache.set_embeddings(missing_texts, new_embeddings)
        for index, embedding in zip(missing_indexes, new_embeddings, strict=True):
            cached[index] = embedding
    else:
        MEMORY_LOGGER.info("Embedding cache hit for all %s texts", len(texts))

    return [embedding if embedding is not None else [] for embedding in cached]


async def embed_query(text: str) -> list[float]:
    cached = await redis_cache.get_embedding(text)
    if cached is not None:
        MEMORY_LOGGER.info("Query embedding cache hit")
        return cached

    MEMORY_LOGGER.info("Query embedding cache miss")
    model = await asyncio.to_thread(get_embeddings_model)
    embedding = await asyncio.to_thread(model.embed_query, text)
    await redis_cache.set_embedding(text, embedding)
    return embedding
