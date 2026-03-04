import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

_ = load_dotenv()

class Settings(BaseSettings):
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    STORAGE_DIR: Path = PROJECT_ROOT / "storage"
    VECTORSTORE_DIR: Path = STORAGE_DIR / "vectorstore"
    GRAPH_DIR: Path = STORAGE_DIR / "graph"

    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Embedding Configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"  # or "cuda"

    # Vector Store
    VECTORSTORE_COLLECTION_NAME: str = "documentation"

    # LLM Configuration
    # HF_MODEL_NAME: str = "HuggingFaceH4/zephyr-7b-beta"
    HF_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct"
    HF_API_TOKEN: str | None = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 512

    # Retrieval Configuration
    RETRIEVAL_TOP_K: int = 4
    RETRIEVAL_SCORE_THRESHOLD: float | None = None

    # Graph Configuration
    GRAPH_ENABLED: bool = False
    GRAPH_BACKEND: str = "networkx"  # or "neo4j"
    NEO4J_URI: str | None = None
    NEO4J_USER: str | None = None
    NEO4J_PASSWORD: str | None = None

    # Entity Extraction
    SPACY_MODEL: str = "en_core_web_sm"
    ENTITY_TYPES: list[str] = ["PERSON", "ORG", "PRODUCT", "GPE", "EVENT", "TECH"]

    class Config:
        env_file: str = ".env"
        case_sensitive: bool = True
        extra: str = "ignore"


settings = Settings()


def setup_directories():
    directories = [
        settings.RAW_DATA_DIR,
        settings.PROCESSED_DATA_DIR,
        settings.VECTORSTORE_DIR,
        settings.GRAPH_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
