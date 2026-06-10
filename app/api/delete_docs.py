from typing import List

from fastapi import APIRouter, Body, HTTPException

from app.core.config import RAW_DB_PATH, settings
from app.db.manager import db_manager
from app.vector_db.chroma_db import ChromaManager
from src.utils.logger import APP_LOGGER

router = APIRouter()

@router.post("/delete")
async def delete_docs(user_id: int, filenames: List[str] = Body(...)):
    """
    Deletes documents from ChromaDB index matching the filename in metadata, 
    and also removes records from PostgreSQL.
    """
    if not filenames:
        raise HTTPException(status_code=400, detail="filenames list cannot be empty")

    try:
        # 1. Delete from PostgreSQL
        deleted_status = await db_manager.execute_command(
            "DELETE FROM user_documents WHERE user_id = $1 AND filename = ANY($2)",
            user_id, filenames
        )
        APP_LOGGER.info(f"Postgres DELETE for user {user_id}: {deleted_status}")

        # 2. Delete from ChromaDB
        chroma_manager = ChromaManager(
            collection_name=f"user_{user_id}",
            persist_dir=str(RAW_DB_PATH),
            model_name=settings.embedding.EMBEDDING_LLM
        )
        
        collection = chroma_manager._vectorstore._collection
        if collection:
            # Note: ChromaDB $in requires a list of strings
            collection.delete(where={"filename": {"$in": filenames}}) # type: ignore
            APP_LOGGER.info(f"ChromaDB DELETE for user {user_id}, files: {filenames}")
            
        return {"message": "Documents deleted successfully", "filenames": filenames}
        
    except Exception as e:
        APP_LOGGER.error(f"Error during document deletion for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
