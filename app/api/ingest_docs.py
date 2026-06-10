import os
import tempfile
import uuid
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings, RAW_DB_PATH
from app.vector_db.chroma_db import ChromaManager
from app.db.manager import db_manager
from src.utils.logger import APP_LOGGER

router = APIRouter()

@router.post("/upload")
async def upload_document(
    user_id: int = Form(...),
    file: UploadFile = File(...),
    version_id: str = Form("1.0")
):
    """
    Ingests a document (PDF/TXT), processes it into chunks with rich metadata,
    stores in ChromaDB, and records the upload in PostgreSQL.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".txt"]:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")

    doc_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    
    # Save uploaded file to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
        else:
            loader = TextLoader(temp_path, encoding="utf-8")
            docs = loader.load()
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        # Prepare for ChromaDB
        texts = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            section_id = str(uuid.uuid4())
            page_number = chunk.metadata.get("page", 0)  # PyPDFLoader usually puts page number here
            
            meta = {
                "user_id": user_id,
                "filename": file.filename,
                "doc_id": doc_id,
                "section_id": section_id,
                "page_number": page_number,
                "version_id": version_id,
                "timestamp": timestamp,
            }
            
            texts.append(chunk.page_content)
            metadatas.append(meta)
            ids.append(section_id)
            
        # Add to ChromaManager
        chroma_manager = ChromaManager(
            collection_name=f"user_{user_id}",
            persist_dir=str(RAW_DB_PATH),
            model_name=settings.embedding.EMBEDDING_LLM
        )
        
        chroma_manager.insert_documents(texts=texts, metadatas=metadatas, collection_name=chroma_manager.collection_name)
        APP_LOGGER.info(f"Ingested {len(texts)} chunks into ChromaDB for user {user_id}")
        
        # Add record to Postgres
        await db_manager.execute_command(
            "INSERT INTO user_documents (user_id, filename, doc_id) VALUES ($1, $2, $3)",
            user_id, file.filename, doc_id
        )
        
        return {"message": "Document ingested successfully", "doc_id": doc_id, "chunks": len(texts)}
        
    except Exception as e:
        APP_LOGGER.error(f"Error during ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
