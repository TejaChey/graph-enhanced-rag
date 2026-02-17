from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from src.utils import setup_logger

logger = setup_logger(__name__)


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None
    ):
        self.chunk_size: int = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap: int = chunk_overlap or settings.CHUNK_OVERLAP

        logger.info(
            "DocumentChunker initialized: ",
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )

        self.text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        logger.info(f"Chunking {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def get_chunk_statistics(self, chunks: list[Document]) -> dict[str,int]:
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": round(sum(chunk_sizes) / len(chunk_sizes)),
            "min_chunk_size": max(chunk_sizes),
            "max_chunk_size": min(chunk_sizes),
        }


if __name__ == "__main__":
    chunker = DocumentChunker()
    print(f"Chunk size: {chunker.chunk_size}")
    print(f"Chunk overlap: {chunker.chunk_overlap}")
