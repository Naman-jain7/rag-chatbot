from fastapi import APIRouter
from app.api import ingest_docs, list_docs, delete_docs, auth, chat

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
api_router.include_router(ingest_docs.router, tags=["Ingestion"])
api_router.include_router(list_docs.router, tags=["Documents"])
api_router.include_router(delete_docs.router, tags=["Documents"])
