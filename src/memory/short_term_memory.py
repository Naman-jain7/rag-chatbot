# Short-term (checkpointer) memory.
# AsyncPostgresSaver is created in app/main.py lifespan using an AsyncConnectionPool
# and stored on app.state.checkpointer.  This module only exposes the helper.

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from app.core.config import settings


async def build_checkpointer(pool: AsyncConnectionPool) -> AsyncPostgresSaver:
    """Create and set-up an AsyncPostgresSaver backed by *pool*."""
    checkpointer = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
    await checkpointer.setup()
    return checkpointer