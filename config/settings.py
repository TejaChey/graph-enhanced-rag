from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    STORAGE_DIR: Path = PROJECT_ROOT / "storage"
    VECTORSTORE_DIR: Path = STORAGE_DIR / "vectorstore"
    GRAPH_DIR: Path = STORAGE_DIR / "graph"
    LOG_DIR: Path = PROJECT_ROOT / "logs"

    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Embedding Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    # EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DEVICE: str = "cpu"  # or "cuda"

    # Vector Store
    VECTORSTORE_COLLECTION_NAME: str = "documentation"

    # LLM Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 512

    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = 4
    RETRIEVAL_SCORE_THRESHOLD: Optional[float] = None  # Minimum similarity score

    # Graph Configuration
    GRAPH_ENABLED: bool = False
    GRAPH_BACKEND: str = "networkx"  # or "neo4j"
    NEO4J_URI: Optional[str] = None
    NEO4J_USER: Optional[str] = None
    NEO4J_PASSWORD: Optional[str] = None

    # Entity Extraction
    SPACY_MODEL: str = "en_core_web_sm"
    ENTITY_TYPES: list = ["PERSON", "ORG", "PRODUCT", "GPE", "EVENT", "TECH"]

    # Logging
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def setup_directories():
    directories = [
        settings.RAW_DATA_DIR,
        settings.PROCESSED_DATA_DIR,
        settings.VECTORSTORE_DIR,
        settings.GRAPH_DIR,
        settings.LOG_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    setup_directories()
    print("Directories created")
    print("Configuration loaded")
    print(f" - Chunk size: {settings.CHUNK_SIZE}")
    print(f" - Embedding model: {settings.EMBEDDING_MODEL}")
    print(f" - LLM model: {settings.OLLAMA_MODEL}")
