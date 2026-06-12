from src.utils.logger import APP_LOGGER
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any, List, Dict, Optional
import asyncpg


class DatabaseManager:
    """
    Thread-safe production database manager handling connection pooling.
    Should be initialized once at application startup.
    """

    def __init__(self, dsn: str, min_size: int = 4, max_size: int = 6) -> None:
        self.dsn = dsn  # Data Source Name
        self.min_size = min_size  # minimum number of db connections the pool keeps open and ready at all times, even when idle
        self.max_size = max_size
        self._pool: Optional[asyncpg.Pool] = None   # cached collection of pre-warmed database connections

    async def connect(self) -> None:
        """Initializes the connection pool."""

        if self._pool is not None:
            APP_LOGGER.warning("Database pool already initialized")
            return

        try:
            self._pool = await asyncpg.create_pool(
                dsn=self.dsn,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=30.0,
            )
            APP_LOGGER.info("Database connection pool established successfully.")
        except Exception as e:
            APP_LOGGER.critical(f"Failed to create database pool: {e}", exc_info=True)
            raise

    async def disconnect(self) -> None:
        """Closes the connection pool safely during shutdown."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            APP_LOGGER.info("Database connection pool closed.")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:  # type:ignore
        """
        Context manager to lease a connection from the pool.
        Automatically handles transaction rollbacks on failure.
        """

        if self._pool is None:
            raise RuntimeError("Database pool is not initialized. Call connect() first.")

        connection = None
        try:
            connection = await self._pool.acquire()
            yield connection  # type: ignore
        except Exception as e:
            APP_LOGGER.error(f"Database operation failed: {e}", exc_info=True)
            raise
        finally:
            if connection and self._pool:
                await self._pool.release(connection)

    async def fetch_rows(self, query: str, *args: Any) -> List[Dict[str, Any]]:
        """Executes a SELECT query and returns results as standard dicts."""

        async with self.get_connection() as conn:
            records = await conn.fetch(query, *args)
            return [dict(record) for record in records]

    async def execute_command(self, query: str, *args: Any) -> str:
        """Executes an INSERT, UPDATE, or DELETE query."""
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)

    async def execute_many(self, query: str, args_list: List[tuple]) -> None:
        """Executes a batch INSERT, UPDATE, or DELETE query."""
        async with self.get_connection() as conn:
            await conn.executemany(query, args_list)

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Fetches a single user record by their unique email."""
        query = "SELECT id, full_name, age, email, hashed_password, preferences FROM users WHERE email = $1"
        rows = await self.fetch_rows(query, email)
        return rows[0] if rows else None

    async def create_user(self, full_name: str, age: Optional[int], email: str, hashed_password: str, preferences: str) -> int:
        """Inserts a new user and returns their ID."""
        query = """
            INSERT INTO users (full_name, age, email, hashed_password, preferences)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            RETURNING id
        """
        async with self.get_connection() as conn:
            return await conn.fetchval(query, full_name, age, email, hashed_password, preferences) # type: ignore