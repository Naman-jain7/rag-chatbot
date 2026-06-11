import logging
from pathlib import Path

log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def setup_logger(name: str, log_file: Path, level=logging.INFO) -> logging.Logger:
    """Helper function to create and configure a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if script runs multiple times
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

APP_LOGGER = setup_logger("app", log_dir / "app.log")
LLM_LOGGER = setup_logger("llm", log_dir / "llm.log")
EMBEDDING_LOGGER = setup_logger("embeddings", log_dir / "embeddings.log")
MEMORY_LOGGER = setup_logger('memory', log_dir / "memory.log")