"""
app/api/ingest_audio.py
-----------------------
FastAPI endpoint for audio file upload and ingestion.

Flow
----
1.  Validate extension (.mp3 / .wav / .m4a / .ogg / .flac)
2.  Write upload to a temp file
3.  SHA-256 duplicate check (same logic as ingest_docs.py)
4.  Run faster-whisper transcription in thread pool
5.  Chunk transcript using segment-grouping AudioChunker
6.  Embed chunks in thread pool (HuggingFace embeddings — same model as docs)
7.  Store chunks via vector_db_manager.store_chunks() — zero schema changes
8.  Insert user_documents row with source_type='audio' and extra_metadata JSONB
9.  Clean up temp file

Endpoint: POST /upload-audio
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings
from app.db.manager import db_manager
from src.audio.chunker import audio_chunker
from src.audio.transcriber import audio_transcriber
from src.memory.vector_db_manager import vector_db_manager
from src.utils.logger import APP_LOGGER

router = APIRouter()

# Lazy singleton — shares embedding model lifecycle with ingest_docs.py
_embeddings: HuggingFaceEmbeddings | None = None

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        APP_LOGGER.info("Loading HuggingFaceEmbeddings for audio ingestion...")
        _embeddings = HuggingFaceEmbeddings(model_name=settings.embedding.EMBEDDING_MODEL)
    return _embeddings


@router.post("/upload-audio")
async def upload_audio(
    user_id: int = Form(...),
    file: UploadFile = File(...),
    version_id: str = Form("1.0"),
):
    """
    Transcribes an audio file, splits the transcript into chunks,
    generates embeddings, and stores everything in PostgreSQL
    (document_chunks via pgvector + user_documents metadata).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Accepted: {', '.join(sorted(AUDIO_EXTENSIONS))}",
        )

    doc_id = str(uuid.uuid4())

    # ── Save upload to temp file ──────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        temp_path = tmp.name

    try:
        file_hash = hashlib.sha256(content).hexdigest()

        # ── Duplicate check ───────────────────────────────────────────────────
        existing = await db_manager.fetch_rows(
            "SELECT id FROM user_documents WHERE user_id = $1 AND file_hash = $2",
            user_id, file_hash,
        )
        if existing:
            raise HTTPException(status_code=409, detail="Audio file already uploaded by this user")

        # ── Fetch user info ───────────────────────────────────────────────────
        user_record = await db_manager.fetch_rows(
            "SELECT full_name, email FROM users WHERE id = $1", user_id
        )
        user_name  = user_record[0]["full_name"] if user_record else "Unknown"
        user_email = user_record[0]["email"]     if user_record else "Unknown"

        # ── Transcription (blocking — run in thread pool) ─────────────────────
        APP_LOGGER.info(f"Transcribing '{file.filename}' for user {user_id} …")
        transcript = await asyncio.to_thread(audio_transcriber.transcribe, temp_path)

        timestamps = transcript["metadata"]["timestamps"]
        duration   = transcript["metadata"]["duration"]

        if not timestamps:
            raise HTTPException(status_code=422, detail="Audio produced no transcript segments")

        # ── Segment-grouping chunking ─────────────────────────────────────────
        raw_chunks = audio_chunker.chunk(timestamps)

        if not raw_chunks:
            raise HTTPException(status_code=422, detail="Transcript chunking produced no chunks")

        # ── Embed chunks (blocking — run in thread pool) ──────────────────────
        texts            = [c["chunk_text"] for c in raw_chunks]
        embeddings_model = await asyncio.to_thread(_get_embeddings)
        embeddings       = await asyncio.to_thread(embeddings_model.embed_documents, texts)

        # ── Build chunks_data for VectorDBManager ─────────────────────────────
        chunks_data = []
        for chunk, embedding in zip(raw_chunks, embeddings):
            chunks_data.append({
                "chunk_text":  chunk["chunk_text"],
                "embedding":   embedding,
                "user_name":   user_name,
                "email":       user_email,
                "file_hash":   file_hash,
                "page_number": chunk["chunk_index"],
                "page_range":  chunk["page_range"],   # "H:MM:SS.mmm–H:MM:SS.mmm"
                "version_id":  version_id,
            })

        # ── Persist chunks to Postgres (reuses existing store_chunks) ─────────
        await vector_db_manager.store_chunks(
            user_id=user_id,
            doc_id=doc_id,
            filename=file.filename,
            chunks_data=chunks_data,
        )
        APP_LOGGER.info(
            f"Ingested {len(chunks_data)} audio chunks into Postgres for user {user_id}"
        )

        # ── Register in user_documents with audio metadata ────────────────────
        extra_metadata = json.dumps({
            "duration":   duration,
            "timestamps": timestamps,
        })

        await db_manager.execute_command(
            """
            INSERT INTO user_documents
                (user_id, filename, doc_id, file_hash, source_type, extra_metadata)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            """,
            user_id, file.filename, doc_id, file_hash, "audio", extra_metadata,
        )

        return {
            "message":  "Audio ingested successfully",
            "doc_id":   doc_id,
            "chunks":   len(chunks_data),
            "duration": duration,
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        APP_LOGGER.error(f"Error during audio ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
