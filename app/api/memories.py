from fastapi import APIRouter
from pydantic import BaseModel

from app.db.manager import db_manager
from src.memory.long_term_memory import long_term_memory as memory_service

router = APIRouter()

class MemoryCreate(BaseModel):
    memory_text: str

@router.post("/memories/{user_id}")
async def add_memory(user_id: int, request: MemoryCreate):
    await memory_service.store_batch(user_id, [request.memory_text])
    return {"message": "Memory added successfully"}

@router.get("/memories/{user_id}")
async def list_memories(user_id: int):
    rows = await db_manager.fetch_rows(
        "SELECT id::text, memory_text, memory_type, created_at::text FROM memories WHERE user_id = $1 ORDER BY created_at DESC",
        user_id,
    )
    return {"memories": [dict(r) for r in rows]}


@router.delete("/memories/{user_id}/{memory_id}")
async def delete_memory(user_id: int, memory_id: str):
    await db_manager.execute_command(
        "DELETE FROM memories WHERE id = $1::uuid AND user_id = $2",
        memory_id,
        user_id,
    )
    return {"message": "Memory deleted"}


@router.delete("/memories/{user_id}")
async def delete_all_memories(user_id: int):
    await db_manager.execute_command(
        "DELETE FROM memories WHERE user_id = $1",
        user_id,
    )
    return {"message": "All memories deleted"}
