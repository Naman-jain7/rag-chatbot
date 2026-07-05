import asyncio
import hashlib
import os
import tempfile
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import HTMLHeaderTextSplitter, Language, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from app.core.config import settings
from app.db.manager import db_manager
from src.memory.vector_db_manager import vector_db_manager
from src.utils.logger import APP_LOGGER

router = APIRouter()

_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        APP_LOGGER.info("Loading HuggingFaceEmbeddings for document ingestion...")
        _embeddings = HuggingFaceEmbeddings(model_name=settings.embedding.EMBEDDING_MODEL)
    return _embeddings


@router.post("/upload")
async def upload_document(
    user_id: int = Form(...),
    file: UploadFile = File(...),
    version_id: str = Form("1.0"),
):
    """
    Ingests a document, splits it into chunks, generates embeddings,
    and stores everything in PostgreSQL (document_chunks via pgvector).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing")

    ext = os.path.splitext(file.filename)[1].lower()
    valid_exts = [
        ".pdf", ".txt", ".md", ".html",
        ".py", ".js", ".ts", ".java", ".cpp", ".go", ".rb", ".php", ".rs",
    ]
    if ext not in valid_exts:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    doc_id = str(uuid.uuid4())

    # Save uploaded file to a temp path so loaders can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        file_hash = hashlib.sha256(content).hexdigest()

        # Duplicate check — same file hash for this user
        existing_doc = await db_manager.fetch_rows(
            "SELECT id FROM user_documents WHERE user_id = $1 AND file_hash = $2",
            user_id, file_hash,
        )
        if existing_doc:
            raise HTTPException(status_code=409, detail="File already exists for this user")

        user_record = await db_manager.fetch_rows(
            "SELECT full_name, email FROM users WHERE id = $1", user_id
        )
        user_name = user_record[0]["full_name"] if user_record else "Unknown"
        user_email = user_record[0]["email"] if user_record else "Unknown"

        # ── Splitting ────────────────────────────────────────────────────────
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
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            ).split_documents(docs)

        elif ext == ".md":
            loader = TextLoader(temp_path, encoding="utf-8")
            docs = loader.load()
            md_docs = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
            ).split_text(docs[0].page_content)
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            ).split_documents(md_docs)

        elif ext == ".html":
            loader = TextLoader(temp_path, encoding="utf-8")
            docs = loader.load()
            html_docs = HTMLHeaderTextSplitter(
                headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")]
            ).split_text(docs[0].page_content)
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            ).split_documents(html_docs)

        elif ext in ext_to_lang:
            loader = TextLoader(temp_path, encoding="utf-8")
            docs = loader.load()
            chunks = RecursiveCharacterTextSplitter.from_language(
                language=ext_to_lang[ext], chunk_size=1000, chunk_overlap=200
            ).split_documents(docs)

        else:  # .txt and any other text files
            loader = TextLoader(temp_path, encoding="utf-8")
            docs = loader.load()
            chunks = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            ).split_documents(docs)

        if not chunks:
            raise HTTPException(status_code=422, detail="Document produced no text chunks")

        # ── Embedding ────────────────────────────────────────────────────────
        texts = [chunk.page_content for chunk in chunks]
        embeddings_model = await asyncio.to_thread(_get_embeddings)
        embeddings = await asyncio.to_thread(embeddings_model.embed_documents, texts)

        # ── Build chunks_data for VectorDBManager ────────────────────────────
        chunks_data = []
        for chunk, embedding in zip(chunks, embeddings):
            page_number = chunk.metadata.get("page", 0)
            chunks_data.append({
                "chunk_text": chunk.page_content,
                "embedding": embedding,
                "user_name": user_name,
                "email": user_email,
                "file_hash": file_hash,
                "page_number": page_number,
                "page_range": f"{page_number}-{page_number}",
                "version_id": version_id,
            })

        # ── Persist to Postgres ───────────────────────────────────────────────
        await vector_db_manager.store_chunks(
            user_id=user_id,
            doc_id=doc_id,
            filename=file.filename,
            chunks_data=chunks_data,
        )
        APP_LOGGER.info(f"Ingested {len(chunks_data)} chunks into Postgres for user {user_id}")

        # Register document in user_documents table
        await db_manager.execute_command(
            "INSERT INTO user_documents (user_id, filename, doc_id, file_hash) VALUES ($1, $2, $3, $4)",
            user_id, file.filename, doc_id, file_hash,
        )

        return {"message": "Document ingested successfully", "doc_id": doc_id, "chunks": len(chunks_data)}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        APP_LOGGER.error(f"Error during ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
