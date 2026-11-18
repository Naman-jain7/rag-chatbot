from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from redis.asyncio import Redis

from configs.core import settings
from src.utils.logger import MEMORY_LOGGER


def _json_safe(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


class RedisCache:
    """
    Redis-backed cache for embedding vectors and hybrid search results.

    Cache misses and Redis outages are non-fatal: callers always fall through
    to the original embedding / retrieval path.
    """

    _EMBED_PREFIX = "emb"
    _SEARCH_PREFIX = "search"
    _SEARCH_INDEX_PREFIX = "search_idx"

    def __init__(self) -> None:
        self._client: Redis | None = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    async def connect(self) -> None:
        if not settings.redis.REDIS_ENABLED:
            MEMORY_LOGGER.info("Redis cache disabled via REDIS_ENABLED=false.")
            return

        if self._client is not None:
            MEMORY_LOGGER.warning("Redis client already initialised.")
            return

        try:
            client: Redis = Redis.from_url(
                settings.redis.redis_url,
                socket_timeout=settings.redis.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=settings.redis.REDIS_SOCKET_TIMEOUT,
                decode_responses=True,
            )
            await client.ping()
            self._client = client
            MEMORY_LOGGER.info("Redis cache connected at %s", settings.redis.redis_url)
        except Exception as exc:
            self._client = None
            MEMORY_LOGGER.warning(
                "Redis unavailable (%s). Embedding and search caching disabled.",
                exc,
            )

    async def disconnect(self) -> None:
        if self._client is None:
            return
        await self._client.aclose()
        self._client = None
        MEMORY_LOGGER.info("Redis cache connection closed.")

    def _digest(self, *parts: str) -> str:
        hasher = hashlib.sha256()
        for part in parts:
            hasher.update(part.encode("utf-8"))
            hasher.update(b"\x1f")
        return hasher.hexdigest()

    def _embedding_key(self, text: str) -> str:
        model = settings.embedding.EMBEDDING_MODEL
        return f"{self._EMBED_PREFIX}:{model}:{self._digest(text)}"

    def _search_key(self, user_id: int, kind: str, *parts: str) -> str:
        return f"{self._SEARCH_PREFIX}:{user_id}:{kind}:{self._digest(*parts)}"

    def _search_index_key(self, user_id: int) -> str:
        return f"{self._SEARCH_INDEX_PREFIX}:{user_id}"

    async def get_embedding(self, text: str) -> list[float] | None:
        if self._client is None:
            return None
        try:
            raw = await self._client.get(self._embedding_key(text))
            if raw is None:
                return None
            payload = json.loads(raw)
            if not isinstance(payload, list):
                return None
            return [float(v) for v in payload]
        except Exception as exc:
            MEMORY_LOGGER.warning("Failed to read embedding cache: %s", exc)
            return None

    async def get_embeddings(self, texts: Sequence[str]) -> list[list[float] | None]:
        if self._client is None or not texts:
            return [None] * len(texts)
        keys = [self._embedding_key(text) for text in texts]
        try:
            raw_values = await self._client.mget(keys)
        except Exception as extra:
            MEMORY_LOGGER.warning("Failed to bulk-read embedding cache: %s", extra)
            return [None] * len(texts)

        results: list[list[float] | None] = []
        for raw in raw_values:
            if raw is None:
                results.append(None)
                continue
            try:
                payload = json.loads(raw)
                results.append([float(v) for v in payload])
            except Exception:
                results.append(None)
        return results

    async def set_embedding(self, text: str, embedding: Sequence[float]) -> None:
        await self.set_embeddings([text], [list(embedding)])

    async def set_embeddings(
        self, texts: Sequence[str], embeddings: Sequence[Sequence[float]]
    ) -> None:
        if self._client is None or not texts:
            return
        ttl = settings.redis.EMBEDDING_CACHE_TTL_SECONDS
        try:
            pipe = self._client.pipeline(transaction=False)
            for text, embedding in zip(texts, embeddings, strict=True):
                pipe.set(
                    self._embedding_key(text),
                    json.dumps(list(embedding)),
                    ex=ttl,
                )
            await pipe.execute()
        except Exception as extra:
            MEMORY_LOGGER.warning("Failed to write embedding cache: %s", extra)

    async def get_search_results(
        self,
        user_id: int,
        kind: str,
        *parts: str,
    ) -> list[dict[str, Any]] | None:
        if self._client is None:
            return None
        try:
            raw = await self._client.get(self._search_key(user_id, kind, *parts))
            if raw is None:
                return None
            payload = json.loads(raw)
            if not isinstance(payload, list):
                return None
            return payload
        except Exception as extra:
            MEMORY_LOGGER.warning("Failed to read search cache: %s", extra)
            return None

    async def set_search_results(
        self,
        user_id: int,
        kind: str,
        results: list[dict[str, Any]],
        *parts: str,
    ) -> None:
        if self._client is None:
            return
        key = self._search_key(user_id, kind, *parts)
        ttl = settings.redis.SEARCH_CACHE_TTL_SECONDS
        try:
            pipe = self._client.pipeline(transaction=False)
            pipe.set(key, json.dumps(_json_safe(results)), ex=ttl)
            pipe.sadd(self._search_index_key(user_id), key)
            pipe.expire(self._search_index_key(user_id), ttl)
            await pipe.execute()
        except Exception as extra:
            MEMORY_LOGGER.warning("Failed to write search cache: %s", extra)

    async def invalidate_user_search(self, user_id: int) -> None:
        if self._client is None:
            return
        index_key = self._search_index_key(user_id)
        try:
            keys = await self._client.smembers(index_key)
            if keys:
                await self._client.delete(*keys, index_key)
            else:
                await self._client.delete(index_key)
            MEMORY_LOGGER.info("Invalidated search cache for user %s", user_id)
        except Exception as extra:
            MEMORY_LOGGER.warning("Failed to invalidate search cache: %s", extra)


redis_cache = RedisCache()
