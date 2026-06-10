from typing import List, Dict, Any, Optional
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.logger import EMBEDDING_LOGGER


class ChromaManager:
    """
    A unified Data Access Object (DAO) for ChromaDB vector operations.
    Handles lifecycle initialization, document upserts, deletions, and semantic search.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_dir: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.model_name = model_name

        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            self.client = chromadb.PersistentClient(path=self.persist_dir)  # type: ignore

            self._vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,  # type: ignore
                embedding_function=self.embeddings,
            )
            EMBEDDING_LOGGER.info(
                f"ChromaManager initialized for collection: '{self.collection_name}'"
            )
        except Exception as e:
            EMBEDDING_LOGGER.critical(f"ChromaManager failed initialization: {str(e)}", exc_info=True)
            raise RuntimeError(f"Vector database initialization failed: {e}") from e

    def _get_vectorstore(self, collection_name: Optional[str] = None) -> Chroma:
        """Helper method to dynamically switch collections safely if requested."""
        if collection_name and collection_name != self.collection_name:
            return Chroma(client=self.client,collection_name=collection_name,embedding_function=self.embeddings,)
        return self._vectorstore

    def insert_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> List[str]:
        """
        Embeds and inserts text documents into the vector database.
        Returns a list of generated IDs for the stored documents.
        """
        if not texts:
            EMBEDDING_LOGGER.warning("Empty text batch provided for insertion.")
            return []

        try:
            v_store = self._get_vectorstore(collection_name)
            # LangChain's Chroma wrapper handles chunks -> embedding generation -> storage implicitly
            ids = v_store.add_texts(texts=texts, metadatas=metadatas)
            EMBEDDING_LOGGER.info(f"Successfully inserted {len(texts)} documents into collection.")
            return ids
        except Exception as e:  # noqa: F841
            EMBEDDING_LOGGER.error("Failed to insert documents into ChromaDB.", exc_info=True)
            raise

    def retrieve(self, query: str, top_k: int = 5, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Executes a semantic similarity search query against the collection."""
        if not query or not query.strip():
            return []

        try:
            v_store = self._get_vectorstore(collection_name)
            docs = v_store.similarity_search(query, k=top_k)
            return [
                {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
            ]
        except Exception as e:  # noqa: F841
            EMBEDDING_LOGGER.error(f"Vector retrieval failure for query: '{query}'", exc_info=True)
            return []

    def retrieve_and_rerank(
        self,
        query: str,
        initial_k: int = 15,
        final_top_k: int = 3,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves an expanded candidate pool using semantic search,
        then refines and trims the results using the `rank_docs` function.
        """
        if not query or not query.strip():
            return []
        
        if final_top_k > initial_k:
            EMBEDDING_LOGGER.warning(
                f"final_top_k ({final_top_k}) cannot exceed initial_k ({initial_k}). "
                f"Adjusting initial_k to match final_top_k."
            )
            initial_k=final_top_k

        EMBEDDING_LOGGER.info(f"Executing retrieve_and_rerank for query: '{query}'")

        candidates = self.retrieve(query=query, top_k=initial_k, collection_name=collection_name)

        if not candidates:
            return []
        
        try:
            reranked_docs = self._rank_docs(query, docs=candidates, top_k=final_top_k)
            EMBEDDING_LOGGER.info(f"Reranked {len(candidates)} candidates down to {len(reranked_docs)} documents.")
            return reranked_docs
        except Exception:
            EMBEDDING_LOGGER.error("Error occurred during document reranking phase.", exc_info=True)
            return candidates[:final_top_k]

    def _rank_docs(self, query: str, docs: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        """Ranks documents based on how many words they have in common with the query"""
        for doc in docs:
            doc["score"] = len(set(query.split()) & set(doc["content"].split()))

        sorted_docs = sorted(docs, key=lambda x: x["score"], reverse=True)
        return sorted_docs[:top_k]

    def delete_by_ids(self, ids: List[str], collection_name: Optional[str] = None) -> bool:
        """Removes specific records from the vector store using their IDs."""
        if not ids:
            return False

        try:
            v_store = self._get_vectorstore(collection_name)
            v_store.delete(ids=ids)
            EMBEDDING_LOGGER.info(f"Successfully deleted {len(ids)} documents from vector store.")
            return True
        except Exception as e:  # noqa: F841
            EMBEDDING_LOGGER.error(f"Failed to delete vector IDs: {ids}", exc_info=True)
            return False
