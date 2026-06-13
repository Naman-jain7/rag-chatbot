import uuid
import os
import re
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

from app.db.manager import db_manager
from src.utils.logger import EMBEDDING_LOGGER
from app.core.config import settings

class VectorDBManager:
    """
    Handles Postgres hybrid retrieval (vector + BM25 equivalent),
    storing, and deleting user document chunks.
    Also performs reranking using BAAI/bge-reranker-v2-m3 cross-encoder.
    """
    def __init__(self):
        EMBEDDING_LOGGER.info(f'Loading {settings.embedding.RERANKER_MODEL}')
        self._reranker = CrossEncoder(settings.embedding.RERANKER_MODEL)

    def _get_reranker(self) -> CrossEncoder:
        return self._reranker

    async def store_chunks(self, user_id: int, doc_id: str, filename: str, chunks_data: List[Dict[str, Any]]) -> None:
        """
        Stores document chunks in batch.
        `chunks_data` should be a list of dictionaries, each containing:
        - chunk_text (str)
        - embedding (List[float])
        - user_name (Optional[str])
        - email (Optional[str])
        - file_hash (Optional[str])
        - page_number (Optional[int])
        - page_range (Optional[str])
        - version_id (Optional[str])
        """
        if not chunks_data:
            EMBEDDING_LOGGER.error('No chunks generated to store')
            return

        query = """
            INSERT INTO document_chunks (
                section_id, user_id, doc_id, filename, chunk_text, embedding, 
                user_name, email, file_hash, page_number, page_range, version_id
            )
            VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8, $9, $10, $11, $12)
        """
        
        args_list = []
        for chunk in chunks_data:
            section_id = str(uuid.uuid4())
            embedding_str = f"[{','.join(map(str, chunk['embedding']))}]"
            
            args = (
                section_id,
                user_id,
                doc_id,
                filename,
                chunk['chunk_text'],
                embedding_str,
                chunk.get('user_name'),
                chunk.get('email'),
                chunk.get('file_hash'),
                chunk.get('page_number'),
                chunk.get('page_range'),
                chunk.get('version_id')
            )
            args_list.append(args)

        EMBEDDING_LOGGER.info(f"store_chunks called with {len(chunks_data) if chunks_data else 0} chunks.")
        
        try:
            await db_manager.execute_many(query, args_list)
            EMBEDDING_LOGGER.info("Execute_many completed successfully.")
        except Exception as e:
            EMBEDDING_LOGGER.error(f"DATABASE INSERT FAILED: {str(e)}", exc_info=True)
            raise e

    async def delete_user_documents(self, user_id: int, doc_id: str) -> None:
        """Deletes all document chunks matching user_id and doc_id."""
        query = "DELETE FROM document_chunks WHERE user_id = $1 AND doc_id = $2"
        await db_manager.execute_command(query, user_id, doc_id)
        EMBEDDING_LOGGER.info(f"Deleted document chunks for user {user_id}, doc {doc_id}.")

    @staticmethod
    def _normalise_filename(value: str) -> str:
        stem = os.path.splitext(value)[0]
        return re.sub(r"[^a-z0-9]+", " ", stem.lower()).strip()

    @classmethod
    def _filename_aliases(cls, filename: str) -> set[str]:
        stem = os.path.splitext(filename)[0]
        without_parenthetical = re.sub(r"\s*\([^)]*\)\s*", " ", stem)
        return {
            alias
            for alias in {
                cls._normalise_filename(stem),
                cls._normalise_filename(without_parenthetical),
            }
            if alias
        }

    async def find_matching_filenames(self, user_id: int, query_text: str) -> List[str]:
        """Return filenames explicitly referenced by the query."""
        rows = await db_manager.fetch_rows(
            "SELECT filename FROM user_documents WHERE user_id = $1 ORDER BY created_at DESC",
            user_id,
        )
        normalised_query = re.sub(r"[^a-z0-9]+", " ", query_text.lower()).strip()
        return [
            row["filename"]
            for row in rows
            if any(alias in normalised_query for alias in self._filename_aliases(row["filename"]))
        ]

    async def get_document_chunks(self, user_id: int, filenames: List[str]) -> List[Dict[str, Any]]:
        """Load every chunk for explicitly named documents in source order."""
        if not filenames:
            return []
        
        filenames_lower = [f.lower() for f in filenames]

        return await db_manager.fetch_rows(
            """
            SELECT id, chunk_text, section_id, filename, page_number,
                   1.0::float AS vector_score, 0.0::float AS fts_score,
                   1.0::float AS rerank_score, 'filename' AS retrieval_method
            FROM document_chunks
            WHERE user_id = $1 AND filename = ANY($2::text[])
            ORDER BY filename, page_number NULLS FIRST, id
            """,
            user_id,
            filenames,
        )

    async def retrieve_and_rerank(self, user_id: int, query_text: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves chunks using hybrid search (Vector + BM25) and reranks using a cross-encoder.
        """
        # 1. Vector Search (Top 20)
        vector_query = """
            SELECT id, chunk_text, section_id, filename, page_number, 
                   1 - (embedding <=> $1::vector) AS vector_score
            FROM document_chunks
            WHERE user_id = $2
            ORDER BY embedding <=> $1::vector
            LIMIT 20
        """
        embedding_str = f"[{','.join(map(str, query_embedding))}]"
        vector_results = await db_manager.fetch_rows(vector_query, embedding_str, user_id)

        # 2. BM25 / Full Text Search (Top 20)
        fts_query = """
            SELECT id, chunk_text, section_id, filename, page_number,
                   ts_rank(fts, websearch_to_tsquery('english', $1)) AS fts_score
            FROM document_chunks
            WHERE user_id = $2 AND fts @@ websearch_to_tsquery('english', $1)
            ORDER BY fts_score DESC
            LIMIT 20
        """
        fts_results = await db_manager.fetch_rows(fts_query, query_text, user_id)

        # 3. Merge and deduplicate
        merged_docs = {}
        for row in vector_results:
            merged_docs[row['id']] = dict(row)
            merged_docs[row['id']]['retrieval_method'] = 'vector'
            merged_docs[row['id']]['fts_score'] = 0.0

        for row in fts_results:
            if row['id'] in merged_docs:
                merged_docs[row['id']]['fts_score'] = row['fts_score']
                merged_docs[row['id']]['retrieval_method'] = 'hybrid'
            else:
                merged_docs[row['id']] = dict(row)
                merged_docs[row['id']]['retrieval_method'] = 'bm25'
                merged_docs[row['id']]['vector_score'] = 0.0

        candidates = list(merged_docs.values())

        if not candidates:
            return []

        # 4. Rerank using CrossEncoder
        reranker = self._get_reranker()
        pairs = [[query_text, doc['chunk_text']] for doc in candidates]
        scores = reranker.predict(pairs)

        for doc, score in zip(candidates, scores):
            doc['rerank_score'] = float(score)

        # Sort by rerank score
        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

        return candidates[:top_k]

vector_db_manager = VectorDBManager()
