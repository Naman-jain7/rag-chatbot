import hashlib
import os
import tempfile
import uuid
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    Language,
)

from app.core.config import RAW_DB_PATH, settings
from app.db.manager import db_manager
from app.vector_db.chroma_db import ChromaManager
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
    valid_exts = [
        ".pdf", ".txt", ".md", ".html", 
        ".py", ".js", ".ts", ".java", ".cpp", ".go", ".rb", ".php", ".rs"
    ]
    if ext not in valid_exts:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    doc_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()
    
    # Save uploaded file to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        file_hash = hashlib.sha256(content).hexdigest()
        
        existing_doc = await db_manager.fetch_rows(
            "SELECT id FROM user_documents WHERE user_id = $1 AND file_hash = $2",
            user_id, file_hash
        )
        if existing_doc:
            raise HTTPException(status_code=409, detail="File already exists for this user")
            
        user_record = await db_manager.fetch_rows("SELECT full_name, email FROM users WHERE id = $1", user_id)
        user_name = user_record[0]["full_name"] if user_record else "Unknown"
        user_email = user_record[0]["email"] if user_record else "Unknown"

        ext_to_lang = {
            ".py": Language.PYTHON,
            ".js": Language.JS,
            ".ts": Language.TS,
            ".java": Language.JAVA,
            ".cpp": Language.CPP,
            ".go": Language.GO,
            ".rb": Language.RUBY,
            ".php": Language.PHP,
            ".rs": Language.RUST,
        }

        if ext == ".pdf":
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
        elif ext == ".md":
            loader = TextLoader(temp_path, encoding="utf-8")
            docs = loader.load()
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            md_docs = markdown_splitter.split_text(docs[0].page_content)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(md_docs)
        elif ext == ".html":
            loader = TextLoader(temp_path, encoding="utf-8")
            docs = loader.load()
            headers_to_split_on = [
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
            ]
            html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            html_docs = html_splitter.split_text(docs[0].page_content)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(html_docs)
        elif ext in ext_to_lang:
            loader = TextLoader(temp_path, encoding="utf-8")
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter.from_language(
                language=ext_to_lang[ext], chunk_size=1000, chunk_overlap=200
            )
            chunks = text_splitter.split_documents(docs)
        else:
            # fallback for .txt and others
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
                "user_name": user_name,
                "email": user_email,
                "filename": file.filename,
                "doc_id": doc_id,
                "file_hash": file_hash,
                "section_id": section_id,
                "page_number": page_number,
                "page_range": f"{page_number}-{page_number}",
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
            "INSERT INTO user_documents (user_id, filename, doc_id, file_hash) VALUES ($1, $2, $3, $4)",
            user_id, file.filename, doc_id, file_hash
        )
        
        return {"message": "Document ingested successfully", "doc_id": doc_id, "chunks": len(texts)}
        
    except Exception as e:
        APP_LOGGER.error(f"Error during ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
