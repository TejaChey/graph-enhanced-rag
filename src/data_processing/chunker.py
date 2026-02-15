from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import settings
from src.utils import setup_logger

logger = setup_logger(__name__)


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        logger.info(
            f"DocumentChunker initialized: "
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )

        # TODO: Initialize the text splitter
        self.text_splitter = None

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        logger.info(f"Chunking {len(documents)} documents...")

        # TODO: Implement chunking
        # Example structure:
        # if not self.text_splitter:
        #     self.text_splitter = RecursiveCharacterTextSplitter(
        #         chunk_size=self.chunk_size,
        #         chunk_overlap=self.chunk_overlap,
        #         length_function=len,
        #     )
        # chunks = self.text_splitter.split_documents(documents)
        # logger.info(f"Created {len(chunks)} chunks")
        # return chunks

        raise NotImplementedError("TODO: Implement document chunking")

    def get_chunk_statistics(self, chunks: List[Document]) -> dict:
        # TODO: Implement statistics calculation
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0
        }


if __name__ == "__main__":
    # Test the chunker
    chunker = DocumentChunker()
    print(f"Chunk size: {chunker.chunk_size}")
    print(f"Chunk overlap: {chunker.chunk_overlap}")
