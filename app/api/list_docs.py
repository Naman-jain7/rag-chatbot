from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.db.manager import db_manager
from src.utils.logger import APP_LOGGER

router = APIRouter()


@router.get("/documents/{user_id}")
async def list_documents(
    user_id: int,
    type: Optional[str] = Query(
        default=None,
        description="Filter by source type: 'document' or 'audio'. Omit for all.",
    ),
):
    """
    Returns a list of files that the user has uploaded and are stored in PostgreSQL.

    Query parameters
    ----------------
    type : str, optional
        ``document``  – text/PDF uploads only
        ``audio``     – MP3/WAV/M4A/etc uploads only
        *(omitted)*   – all files, sorted by upload time descending
    """
    try:
        if type is not None and type not in ("document", "audio"):
            raise HTTPException(
                status_code=400,
                detail="Invalid type filter. Use 'document' or 'audio'.",
            )

        if type:
            docs = await db_manager.fetch_rows(
                """
                SELECT doc_id, filename, source_type, extra_metadata, created_at
                FROM user_documents
                WHERE user_id = $1 AND source_type = $2
                ORDER BY created_at DESC
                """,
                user_id,
                type,
            )
        else:
            docs = await db_manager.fetch_rows(
                """
                SELECT doc_id, filename, source_type, extra_metadata, created_at
                FROM user_documents
                WHERE user_id = $1
                ORDER BY created_at DESC
                """,
                user_id,
            )

        return {"user_id": user_id, "documents": docs}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        APP_LOGGER.error(f"Failed to fetch documents for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch documents")
