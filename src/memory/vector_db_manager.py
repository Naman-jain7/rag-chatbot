import uuid
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

        await db_manager.execute_many(query, args_list)
        EMBEDDING_LOGGER.info(f"Successfully stored {len(chunks_data)} chunks for user {user_id}, doc {doc_id}.")

    async def delete_user_documents(self, user_id: int, doc_id: str) -> None:
        """Deletes all document chunks matching user_id and doc_id."""
        query = "DELETE FROM document_chunks WHERE user_id = $1 AND doc_id = $2"
        await db_manager.execute_command(query, user_id, doc_id)
        EMBEDDING_LOGGER.info(f"Deleted document chunks for user {user_id}, doc {doc_id}.")

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
