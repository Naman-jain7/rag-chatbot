from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import settings
from app.db.manager import db_manager
from src.utils.logger import APP_LOGGER, EMBEDDING_LOGGER


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: connect to DB and ensure tables exist
    await db_manager.connect()
    
    # Create tables
    try:
        await db_manager.execute_command("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                full_name VARCHAR(100) NOT NULL,
                age INT,
                email VARCHAR(255) UNIQUE NOT NULL,
                hashed_password VARCHAR(255) NOT NULL,
                preferences JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await db_manager.execute_command("""
            CREATE TABLE IF NOT EXISTS user_documents (
                id SERIAL PRIMARY KEY,
                user_id INT REFERENCES users(id) ON DELETE CASCADE,
                filename VARCHAR(255) NOT NULL,
                doc_id VARCHAR(255) NOT NULL,
                file_hash VARCHAR(64),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await db_manager.execute_command("""
            ALTER TABLE user_documents ADD COLUMN IF NOT EXISTS file_hash VARCHAR(64);
        """)
        APP_LOGGER.info("Database tables verified.")
    except Exception as e:
        APP_LOGGER.error(f"Failed to create database tables: {e}")

    # Pre-load embedding model to avoid slow first-upload
    try:
        import asyncio

        from langchain_huggingface import HuggingFaceEmbeddings
        await asyncio.to_thread(HuggingFaceEmbeddings, model_name=settings.embedding.EMBEDDING_LLM)
        EMBEDDING_LOGGER.info("Embedding model pre-loaded successfully.")
    except Exception as e:
        EMBEDDING_LOGGER.warning(f"Could not pre-load embedding model: {e}")

    yield
    
    # Shutdown
    await db_manager.disconnect()


app = FastAPI(
    title=settings.app_config.APP_NAME,
    version=settings.app_config.APP_VERSION,
    lifespan=lifespan
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first = errors[0] if errors else {}
    field = ".".join(str(loc) for loc in first.get("loc", [])) if first.get("loc") else "body"
    msg = first.get("msg", "Invalid input")
    return JSONResponse(
        status_code=422,
        content={"detail": f"'{field}': {msg}"}
    )

app.include_router(api_router, prefix="/api/v1")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

