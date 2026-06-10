from app.db.postgres_db import DatabaseManager
from app.core.config import settings

db_manager = DatabaseManager(dsn=settings.db.DB_DSN) # type: ignore
