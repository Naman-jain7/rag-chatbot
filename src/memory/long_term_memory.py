import asyncio
import uuid
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.db.manager import db_manager
from src.utils.logger import APP_LOGGER


class LongTermMemory:
    """
    Data Access Layer for long term memory. Handles embedding generation
    and lower-level transactional execution against PostgreSQL.
    """
    def __init__(self):
        self._embeddings: HuggingFaceEmbeddings | None = None

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        """Lazy-load the embeddings model so import time is fast."""
        if self._embeddings is None:
            APP_LOGGER.info("Loading HuggingFaceEmbeddings for memory encoding…")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding.EMBEDDING_MODEL
            )
        return self._embeddings

    async def get_all_raw_texts(self, user_id: int) -> List[str]:
        """Fetch plain text of all memory entries for a user."""
        try:
            sql = "SELECT memory_text FROM memories WHERE user_id = $1"
            rows = await db_manager.fetch_rows(sql, user_id)
            return [row["memory_text"] for row in rows] if rows else []
        except Exception as e:
            APP_LOGGER.error(f"Error fetching raw memories for user {user_id}: {e}")
            return []

    async def store_batch(self, user_id: int, text_memories: List[str]) -> None:
        """Embed and insert memories into the DB."""
        try:
            embeddings_model = await asyncio.to_thread(self._get_embeddings)
            embeddings = await asyncio.to_thread(
                embeddings_model.embed_documents, text_memories
            )

            for text, embedding in zip(text_memories, embeddings):
                memory_id = str(uuid.uuid4())

                memory_query = """
                    INSERT INTO memories (id, user_id, memory_text, memory_type)
                    VALUES ($1::uuid, $2, $3, $4)
                """
                await db_manager.execute_command(
                    memory_query, memory_id, user_id, text, "user_memory"
                )

                embedding_str = f"[{','.join(map(str, embedding))}]"
                embedding_query = """
                    INSERT INTO memory_embeddings (memory_id, embedding)
                    VALUES ($1::uuid, $2::vector)
                """
                await db_manager.execute_command(
                    embedding_query, memory_id, embedding_str
                )

            APP_LOGGER.info(
                f"Stored {len(text_memories)} memory records for user {user_id}."
            )
        except Exception as e:
            APP_LOGGER.error(f"Failed storing memory batch: {e}", exc_info=True)

    async def get_relevant(self, user_id: int, query: str) -> str:
        """
        Retrieve relevant memories for a user.

        Strategy:
        1. Vector similarity search via memory_embeddings JOIN.
        2. If no rows returned (embeddings missing) OR on exception, fall back to
           fetching all memories ordered by recency.
        """
        # ── 1. Vector search ──────────────────────────────────────────────────
        try:
            embeddings_model = await asyncio.to_thread(self._get_embeddings)
            query_embedding = await asyncio.to_thread(
                embeddings_model.embed_query, query
            )
            embedding_str = f"[{','.join(map(str, query_embedding))}]"

            sql = """
                SELECT m.memory_text
                FROM memories m
                JOIN memory_embeddings me ON m.id = me.memory_id
                WHERE m.user_id = $2
                ORDER BY me.embedding <=> $1::vector
                LIMIT 10
            """
            rows = await db_manager.fetch_rows(sql, embedding_str, user_id)
            if rows:
                return "\n".join([f"- {row['memory_text']}" for row in rows])
            APP_LOGGER.info(
                f"Vector search returned no memories for user {user_id}; "
                "trying plain fallback."
            )
        except Exception as e:
            APP_LOGGER.error(
                f"Vector memory retrieval failed for user {user_id}: {e}",
                exc_info=True,
            )

        # ── 2. Plain SQL fallback (works even without embeddings) ─────────────
        try:
            sql = (
                "SELECT memory_text FROM memories "
                "WHERE user_id = $1 ORDER BY created_at DESC LIMIT 10"
            )
            rows = await db_manager.fetch_rows(sql, user_id)
            return "\n".join([f"- {row['memory_text']}" for row in rows]) if rows else ""
        except Exception as ex:
            APP_LOGGER.error(
                f"Fallback memory retrieval failed for user {user_id}: {ex}",
                exc_info=True,
            )
            return ""


long_term_memory = LongTermMemory()
