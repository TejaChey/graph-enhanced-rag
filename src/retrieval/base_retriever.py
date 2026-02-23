from langchain_chroma import Chroma

from config import settings
from src.embeddings import EmbeddingModel


class BaseRetriever:
    def __init__(self, embedding_model=None, vectorstore_dir=None, top_k=None):
        self.vectorstore_dir = vectorstore_dir or settings.VECTORSTORE_DIR
        self.embedding_model = embedding_model or EmbeddingModel()
        self.top_k = top_k or settings.RETRIEVAL_TOP_K
        self.vectorstore = None

    def create_vectorstore(self, documents):
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            persist_directory=str(self.vectorstore_dir),
            embedding=self.embedding_model.get_embeddings(),
            collection_name=settings.VECTORSTORE_COLLECTION_NAME,
        )
        return self.vectorstore

    def load_vectorstore(self):
        if not self.vectorstore_dir.exists():
            raise FileNotFoundError(f"Vector store not found at {self.vectorstore_dir}")
        self.vectorstore = Chroma(
            persist_directory=str(self.vectorstore_dir),
            embedding_function=self.embedding_model.get_embeddings(),
            collection_name=settings.VECTORSTORE_COLLECTION_NAME,
        )
        return self.vectorstore

    def as_retriever(self):
        """Return a LangChain-compatible retriever for use in chains."""
        if self.vectorstore is None:
            self.load_vectorstore()
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )

    def retrieve(self, query, top_k=None):
        k = top_k or self.top_k
        if self.vectorstore is None:
            self.load_vectorstore()
        return self.vectorstore.similarity_search(query, k=k)

    def retrieve_with_scores(self, query, top_k=None):
        k = top_k or self.top_k
        if self.vectorstore is None:
            self.load_vectorstore()
        return self.vectorstore.similarity_search_with_score(query, k=k)


if __name__ == "__main__":
    retriever = BaseRetriever()
    print(f"Vector store directory: {retriever.vectorstore_dir}")
    print(f"Top K: {retriever.top_k}")
