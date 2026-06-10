from fastapi import APIRouter, HTTPException

from app.db.manager import db_manager
from src.utils.logger import APP_LOGGER

router = APIRouter()

@router.get("/documents/{user_id}")
async def list_documents(user_id: int):
    """
    Returns a list of documents that the user has uploaded and are stored in PostgreSQL.
    """
    try:
        docs = await db_manager.fetch_rows(
            "SELECT doc_id, filename, created_at FROM user_documents WHERE user_id = $1 ORDER BY created_at DESC",
            user_id
        )
        # asyncpg records return datetime objects which might need string formatting if not handled by FastAPI automatically. 
        # But FastAPI json encoder handles datetime implicitly.
        return {"user_id": user_id, "documents": docs}
    except Exception as e:
        APP_LOGGER.error(f"Failed to fetch documents for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch documents")