from typing import List

from fastapi import APIRouter, Body, HTTPException

from app.db.manager import db_manager
from src.memory.vector_db_manager import vector_db_manager
from src.utils.logger import APP_LOGGER

router = APIRouter()


@router.post("/delete")
async def delete_docs(user_id: int, filenames: List[str] = Body(...)):
    """
    Deletes documents from Postgres (document_chunks and user_documents tables)
    matching the given filenames for the specified user.
    """
    if not filenames:
        raise HTTPException(status_code=400, detail="filenames list cannot be empty")

    try:
        # Look up doc_ids for the given filenames so we can delete their chunks
        rows = await db_manager.fetch_rows(
            "SELECT doc_id FROM user_documents WHERE user_id = $1 AND filename = ANY($2)", user_id, filenames,
        )
        doc_ids = [row["doc_id"] for row in rows]

        if not doc_ids:
            raise HTTPException(status_code=404, detail="No matching documents found for this user")

        # Delete chunks from document_chunks (pgvector table)
        for doc_id in doc_ids:
            await vector_db_manager.delete_user_documents(user_id, doc_id)

        # Delete records from user_documents metadata table
        deleted_status = await db_manager.execute_command(
            "DELETE FROM user_documents WHERE user_id = $1 AND filename = ANY($2)", user_id, filenames,
        )
        APP_LOGGER.info(
            f"Deleted {len(doc_ids)} document(s) for user {user_id} "
            f"(filenames: {filenames}): {deleted_status}"
        )

        return {"message": "Documents deleted successfully", "filenames": filenames}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        APP_LOGGER.error(f"Error during document deletion for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
