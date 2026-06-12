from fastapi import APIRouter
from app.db.manager import db_manager

router = APIRouter()

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
