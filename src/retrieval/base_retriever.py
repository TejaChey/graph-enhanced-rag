from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from config import settings
from src.embeddings import EmbeddingModel
from src.utils import setup_logger

logger = setup_logger(__name__)


class BaseRetriever:
    def __init__(
        self,
        vectorstore_dir: Path,
        embedding_model: EmbeddingModel,
        top_k: int
    ):
        self.vectorstore_dir = vectorstore_dir or settings.VECTORSTORE_DIR
        self.embedding_model = embedding_model or EmbeddingModel()
        self.top_k = top_k or settings.RETRIEVAL_TOP_K
        self.vectorstore = None

        logger.info(f"BaseRetriever initialized with top_k={self.top_k}")

    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        logger.info(f"Creating vector store from {len(documents)} documents...")

        # TODO: Implement vector store creation
        # Example structure:
        # self.vectorstore = Chroma.from_documents(
        #     documents=documents,
        #     embedding=self.embedding_model.get_embeddings(),
        #     persist_directory=str(self.vectorstore_dir),
        #     collection_name=settings.VECTORSTORE_COLLECTION_NAME
        # )
        # logger.info(f"Vector store created and persisted to {self.vectorstore_dir}")
        # return self.vectorstore

        raise NotImplementedError("TODO: Implement vector store creation")

    def load_vectorstore(self) -> Chroma:
        logger.info(f"Loading vector store from {self.vectorstore_dir}...")

        # TODO: Implement vector store loading
        # Example:
        # if not self.vectorstore_dir.exists():
        #     raise FileNotFoundError(f"Vector store not found at {self.vectorstore_dir}")
        # self.vectorstore = Chroma(
        #     persist_directory=str(self.vectorstore_dir),
        #     embedding_function=self.embedding_model.get_embeddings(),
        #     collection_name=settings.VECTORSTORE_COLLECTION_NAME
        # )
        # logger.info("Vector store loaded successfully")
        # return self.vectorstore

        raise NotImplementedError("TODO: Implement vector store loading")

    def retrieve(self, query: str, top_k: int) -> List[Document]:
        k = top_k or self.top_k
        logger.info(f"Retrieving top {k} documents for query: {query[:50]}...")

        # TODO: Implement retrieval
        # Make sure to load vectorstore first if needed

        raise NotImplementedError("TODO: Implement document retrieval")

    def retrieve_with_scores(self, query: str, top_k: int) -> List[tuple]:
        k = top_k or self.top_k
        logger.info(f"Retrieving top {k} documents with scores...")

        # TODO: Use similarity_search_with_score
        raise NotImplementedError("TODO: Implement retrieval with scores")


if __name__ == "__main__":
    # Test the retriever
    retriever = BaseRetriever()
    print(f"Vector store directory: {retriever.vectorstore_dir}")
    print(f"Top K: {retriever.top_k}")
