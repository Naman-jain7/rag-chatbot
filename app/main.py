from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import settings
from app.db.manager import db_manager
from src.utils.logger import APP_LOGGER, EMBEDDING_LOGGER


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 1. asyncpg pool (app-level queries) ──────────────────────────────────
    await db_manager.connect()

    # ── 2. Base SQL tables (no pgvector required) ─────────────────────────────
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
            CREATE TABLE IF NOT EXISTS chat_conversations (
                user_id INT REFERENCES users(id) ON DELETE CASCADE,
                chat_id VARCHAR(255) NOT NULL,
                title VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, chat_id)
            );
        """)
        await db_manager.execute_command("""
            CREATE TABLE IF NOT EXISTS memories (
                id UUID PRIMARY KEY,
                user_id INT REFERENCES users(id) ON DELETE CASCADE,
                memory_text TEXT NOT NULL,
                memory_type VARCHAR(50),
                importance_score FLOAT DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                metadata JSONB
            );
        """)
        APP_LOGGER.info("Base database tables verified.")
    except Exception as e:
        APP_LOGGER.error(f"Failed to create base tables: {e}")

    # ── 3. pgvector tables ────────────────────────────────────────────────────
    try:
        await db_manager.execute_command("CREATE EXTENSION IF NOT EXISTS vector;")
        await db_manager.execute_command(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                section_id UUID UNIQUE NOT NULL,
                user_id INT REFERENCES users(id) ON DELETE CASCADE,
                user_name VARCHAR(100),
                email VARCHAR(255),
                filename VARCHAR(255) NOT NULL,
                doc_id VARCHAR(255) NOT NULL,
                file_hash VARCHAR(64),
                page_number INT,
                page_range VARCHAR(50),
                version_id VARCHAR(50),
                chunk_text TEXT NOT NULL,
                embedding VECTOR({settings.embedding.EMBEDDING_LLM_DIMENSION}),
                fts tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await db_manager.execute_command("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_doc_chunks_fts
            ON document_chunks USING gin (fts);
        """)
        await db_manager.execute_command(f"""
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id UUID PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                embedding VECTOR({settings.embedding.EMBEDDING_LLM_DIMENSION})
            );
        """)
        await db_manager.execute_command("""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_doc_chunks_embedding_hnsw
            ON document_chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 200);
        """)
        APP_LOGGER.info("Vector database tables verified.")
    except Exception as e:
        APP_LOGGER.warning(f"Could not initialize vector tables (pgvector may be missing): {e}")

    # ── 4. AsyncPostgresSaver (LangGraph short-term / checkpointer memory) ────
    # AsyncPostgresSaver requires its own psycopg3 connection pool (it cannot
    # share the asyncpg pool used by db_manager).
    cp_pool = None
    try:
        from psycopg_pool import AsyncConnectionPool
        from src.memory.short_term_memory import build_checkpointer
        from src.graphs.chains import workflow  # import the uncompiled workflow

        cp_pool = AsyncConnectionPool(
            conninfo=settings.db.DB_DSN, # type: ignore
            min_size=2,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False,  # open manually inside the async context
        )
        await cp_pool.open()

        checkpointer = await build_checkpointer(cp_pool) # type: ignore
        app.state.checkpointer = checkpointer

        # Recompile the graph with the real async checkpointer
        import src.graphs.chains as chains_module
        chains_module.graph = workflow.compile(checkpointer=checkpointer)
        app.state.graph = chains_module.graph
        APP_LOGGER.info("AsyncPostgresSaver checkpointer initialised and graph recompiled.")
    except Exception as e:
        APP_LOGGER.error(f"Failed to initialise checkpointer: {e}. Running without persistent memory.")
        # Fall back: graph was already compiled without a checkpointer in chains.py
        import src.graphs.chains as chains_module
        app.state.graph = chains_module.graph

    # ── 5. Pre-warm embedding model ───────────────────────────────────────────
    try:
        import asyncio
        from langchain_huggingface import HuggingFaceEmbeddings
        await asyncio.to_thread(HuggingFaceEmbeddings, model_name=settings.embedding.EMBEDDING_LLM)
        EMBEDDING_LOGGER.info("Embedding model pre-loaded successfully.")
    except Exception as e:
        EMBEDDING_LOGGER.warning(f"Could not pre-load embedding model: {e}")

    yield  # ── application runs ───────────────────────────────────────────────

    # ── 6. Shutdown ───────────────────────────────────────────────────────────
    if cp_pool is not None:
        await cp_pool.close()
    await db_manager.disconnect()


app = FastAPI(
    title=settings.app.APP_NAME,
    version=settings.app.APP_VERSION,
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first = errors[0] if errors else {}
    field = ".".join(str(loc) for loc in first.get("loc", [])) if first.get("loc") else "body"
    msg = first.get("msg", "Invalid input")
    return JSONResponse(
        status_code=422,
        content={"detail": f"'{field}': {msg}"},
    )


app.include_router(api_router, prefix="/api/v1")
