from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings


class DocumentChunker:
    def __init__(self, chunk_size=None, chunk_overlap=None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk_documents(self, documents):
        return self.text_splitter.split_documents(documents)

    def get_chunk_statistics(self, chunks):
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": round(sum(chunk_sizes) / len(chunk_sizes)),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
        }


if __name__ == "__main__":
    chunker = DocumentChunker()
    print(f"Chunk size: {chunker.chunk_size}")
    print(f"Chunk overlap: {chunker.chunk_overlap}")
