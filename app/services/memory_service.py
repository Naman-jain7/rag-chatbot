from typing import List, Dict, Any, Optional
from mem0 import MemoryClient
from app.core.config import settings
from src.utils.logger import MEMORY_LOGGER

class MemoryService:
    def __init__(self):
        try:
            self.client = MemoryClient(api_key=settings.memory.MEM0_API_KEY)
            self.project = settings.memory.MEM0_PROJECT
            self.org_id = settings.memory.MEM0_ORG_ID
            MEMORY_LOGGER.info("MemoryService successfully initialized with mem0 MemoryClient.")
        except Exception as e:
            MEMORY_LOGGER.error(f"Failed to initialize MemoryService: {e}")
            self.client = None

    def store(self, user_id: int, query: str, response: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Save Q&A pair to mem0 after generation.
        """
        if not self.client:
            return

        try:
            # We add both query and response as context
            context = f"User: {query}\nAssistant: {response}"
            
            # Using memory client to add the interaction
            kwargs = {"user_id": str(user_id)}
            if metadata:
                kwargs["metadata"] = metadata # type: ignore

            self.client.add(context, **kwargs) # type: ignore
            MEMORY_LOGGER.info(f"Successfully stored memory for user {user_id}.")
        except Exception as e:
            MEMORY_LOGGER.error(f"Error storing memory for user {user_id}: {e}")

    def get_relevant(self, user_id: int, query: str) -> str:
        """
        Searches mem0 for memories relevant to the current query.
        Returns a formatted string of the retrieved memories.
        """
        if not self.client:
            return ""

        try:
            # Note: user_id must be a string for mem0
            results = self.client.search(query, filters={"user_id": str(user_id)})
            
            if not results:
                return ""
                
            memories = []
            for r in results:
                if isinstance(r, dict):
                    memory_text = r.get("memory", r.get("text", str(r)))
                else:
                    memory_text = getattr(r, "memory", getattr(r, "text", str(r)))
                memories.append(f"- {memory_text}")
                
            return "\n".join(memories)
        except Exception as e:
            MEMORY_LOGGER.error(f"Error retrieving memories for user {user_id}: {e}")
            return ""

    def get_all(self, user_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve full memory history for a user.
        """
        if not self.client:
            return []

        try:
            results = self.client.get_all(user_id=str(user_id))
            return results if isinstance(results, list) else [results]
        except Exception as e:
            MEMORY_LOGGER.error(f"Error retrieving all memories for user {user_id}: {e}")
            return []

memory_service = MemoryService()
