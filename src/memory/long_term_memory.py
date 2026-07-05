import asyncio
import uuid
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.db.manager import db_manager
from src.llm.providers import fast_providers
from src.llm.resilience import ResilientLLMManager
from src.utils.logger import APP_LOGGER


class LongTermMemory:
    """
    Data Access Layer for long term memory. Handles embedding generations
    and lower-level transactional execution boundaries against PostgreSQL.
    """
    def __init__(self):
        self._embeddings = HuggingFaceEmbeddings(model_name=settings.embedding.EMBEDDING_MODEL)
        self._fast_llm = ResilientLLMManager(providers=fast_providers) # type: ignore

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        return self._embeddings

    def _get_fast_llm(self) -> ResilientLLMManager:
        return self._fast_llm
    
    async def get_all_raw_texts(self, user_id: int) -> List[str]:
        """Fetch plain text arrays of all historical entries."""
        try:
            sql = "SELECT memory_text FROM memories WHERE user_id = $1"
            rows = await db_manager.fetch_rows(sql, user_id)
            return [row["memory_text"] for row in rows] if rows else []
        except Exception as e:
            APP_LOGGER.error(f"Error fetching raw memories for user {user_id}: {e}")
            return []

    async def store_batch(self, user_id: int, text_memories: List[str]) -> None:
        """Generates vectors and inserts rows down into relational tables."""
        try:
            embeddings_model = await asyncio.to_thread(self._get_embeddings)
            embeddings = await asyncio.to_thread(embeddings_model.embed_documents, text_memories)
            
            for text, embedding in zip(text_memories, embeddings):
                memory_id = str(uuid.uuid4())
                
                memory_query = """
                    INSERT INTO memories (id, user_id, memory_text, memory_type)
                    VALUES ($1::uuid, $2, $3, $4)
                """
                await db_manager.execute_command(memory_query, memory_id, user_id, text, "user_memory")
                
                embedding_str = f"[{','.join(map(str, embedding))}]"
                embedding_query = """
                    INSERT INTO memory_embeddings (memory_id, embedding)
                    VALUES ($1::uuid, $2::vector)
                """
                await db_manager.execute_command(embedding_query, memory_id, embedding_str)
                
            APP_LOGGER.info(f"Stored {len(text_memories)} new database records for user {user_id}.")
        except Exception as e:
            APP_LOGGER.error(f"Failed storing memory batch to DB: {e}")

    async def get_relevant(self, user_id: int, query: str) -> str:
        """Retrieves relevant memories using vector search in Postgres."""
        try:
            embeddings_model = await asyncio.to_thread(self._get_embeddings)
            query_embedding = await asyncio.to_thread(embeddings_model.embed_query, query)
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            sql = """
                SELECT m.memory_text, 1 - (me.embedding <=> $1::vector) AS similarity
                FROM memories m
                JOIN memory_embeddings me ON m.id = me.memory_id
                WHERE m.user_id = $2
                ORDER BY me.embedding <=> $1::vector
                LIMIT 5
            """
            rows = await db_manager.fetch_rows(sql, embedding_str, user_id)
            if not rows:
                return ""
            
            return "\n".join([f"- {row['memory_text']}" for row in rows])
        except Exception as e:
            APP_LOGGER.error(f"Error retrieving memories via vector: {e}")
            # Fallback: retrieve recent memories
            try:
                sql = "SELECT memory_text FROM memories WHERE user_id = $1 ORDER BY created_at DESC LIMIT 10"
                rows = await db_manager.fetch_rows(sql, user_id)
                return "\n".join([f"- {row['memory_text']}" for row in rows]) if rows else ""
            except Exception as ex:
                APP_LOGGER.error(f"Fallback memory retrieval failed: {ex}")
                return ""

long_term_memory = LongTermMemory()
