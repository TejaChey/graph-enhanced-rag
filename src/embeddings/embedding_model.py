from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings

from config import settings
from src.utils import setup_logger

logger = setup_logger(__name__)


class EmbeddingModel:
    def __init__(
        self,
        model_name: str,
        device: str
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE

        logger.info(f"Initializing embedding model: {self.model_name} on {self.device}")

        # TODO: Initialize HuggingFaceEmbeddings
        self.embeddings = None

    def get_embeddings(self) -> HuggingFaceEmbeddings:
        # TODO: Implement lazy loading
        # if self.embeddings is None:
        #     self.embeddings = HuggingFaceEmbeddings(
        #         model_name=self.model_name,
        #         model_kwargs={'device': self.device}
        #     )
        # return self.embeddings

        raise NotImplementedError("TODO: Implement embedding model initialization")

    def embed_query(self, text: str) -> List[float]:
        # TODO: Use self.get_embeddings().embed_query(text)
        raise NotImplementedError("TODO: Implement query embedding")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # TODO: Use self.get_embeddings().embed_documents(texts)
        raise NotImplementedError("TODO: Implement document embedding")


if __name__ == "__main__":
    # Test the embedding model
    model = EmbeddingModel()
    print(f"Model: {model.model_name}")
    print(f"Device: {model.device}")
