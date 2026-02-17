from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from src.embeddings import EmbeddingModel
from src.utils import setup_logger

logger = setup_logger(__name__)


class BaseRetriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
        vectorstore_dir: Path | None = None,
        top_k: int | None = None
    ):
        self.vectorstore_dir: Path = vectorstore_dir or settings.VECTORSTORE_DIR
        self.embedding_model: EmbeddingModel = embedding_model or EmbeddingModel()
        self.top_k: int = top_k or settings.RETRIEVAL_TOP_K
        self.vectorstore: Chroma | None = None

        logger.info(f"BaseRetriever initialized with top_k={self.top_k}")

    def create_vectorstore(self, documents: list[Document]) -> Chroma:
        logger.info(f"Creating vector store from {len(documents)} documents...")

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            persist_directory=str(self.vectorstore_dir),
            embedding_function=self.embedding_model.get_embeddings(),
            collection_name=settings.VECTORSTORE_COLLECTION_NAME,
        )

        logger.info(f"Vector store created and persisted to {self.vectorstore_dir}")
        return self.vectorstore

    def load_vectorstore(self) -> Chroma:
        logger.info(f"Loading vector store from {self.vectorstore_dir}...")

        if not self.vectorstore_dir.exists():
            raise FileNotFoundError(f"Vector store not found at {self.vectorstore_dir}")

        self.vectorstore = Chroma(
            persist_directory=str(self.vectorstore_dir),
            embedding_function=self.embedding_model.get_embeddings(),
            collection_name=settings.VECTORSTORE_COLLECTION_NAME
        )

        logger.info("Vector store loaded successfully")
        return self.vectorstore

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        k = top_k or self.top_k
        logger.info(f"Retrieving top {k} documents...")

        if self.vectorstore is None:
            _ = self.load_vectorstore()

        results = self.vectorstore.similarity_search(query, k=k)
        logger.info(f"Retrieved {len(results)} docs")
        return results

    def retrieve_with_scores(self, query: str, top_k: int | None = None) -> list[tuple[Document, float]]:
        k = top_k or self.top_k
        logger.info(f"Retrieving top {k} documents with scores...")

        if self.vectorstore is None:
            _ = self.load_vectorstore()

        results = self.vectorstore.similarity_search_with_score(query, k=k)
        logger.info(f"Retrieved {len(results)} docs")
        return results


if __name__ == "__main__":
    embedding_model = EmbeddingModel()
    retriever = BaseRetriever(embedding_model=embedding_model)
    print(f"Vector store directory: {retriever.vectorstore_dir}")
    print(f"Top K: {retriever.top_k}")
