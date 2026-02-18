from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings

from config import settings
from src.utils import setup_logger

logger = setup_logger(__name__)


class EmbeddingModel:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None
    ):
        self.model_name: str = model_name or settings.EMBEDDING_MODEL
        self.device: str = device or settings.EMBEDDING_DEVICE

        logger.info(f"Initializing embedding model: {self.model_name} on {self.device}")

        self.embeddings: HuggingFaceEmbeddings | None = None

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        if self.embeddings is None:
            logger.info("Loading HuggingFace embeddings model...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embeddings model loaded successfully")
        return self.embeddings

    def embed_query(self, text: str) -> list[float]:
        logger.info("Embedding single query...")
        embeddings = self.embeddings.embed_query(text)
        logger.info("Query embedded successfully")
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        logger.info(f"Embedding {len(texts)} documents...")
        embeddings = self.embeddings.embed_documents(texts)
        logger.info("Documents embedded successfully")
        return embeddings


if __name__ == "__main__":
    # Test the embedding model
    model = EmbeddingModel()
    print(f"Model: {model.model_name}")
    print(f"Device: {model.device}")
