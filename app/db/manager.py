from app.db.postgres_db import DatabaseManager
from configs.core import settings

db_manager = DatabaseManager(dsn=settings.db.DB_DSN) # type: ignore
