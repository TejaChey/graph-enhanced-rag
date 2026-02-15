import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings, setup_directories
from src.data_processing import DocumentChunker, DocumentLoader, TextCleaner
from src.embeddings import EmbeddingModel
from src.retrieval import BaseRetriever
from src.utils import setup_logger

logger = setup_logger(__name__, log_file=settings.LOG_DIR / "ingest_baseline.log")


def main():
    logger.info("=" * 60)
    logger.info("Starting Baseline RAG Ingestion Pipeline")
    logger.info("=" * 60)

    # Setup
    setup_directories()
    logger.info(f"Data directory: {settings.RAW_DATA_DIR}")
    logger.info(f"Vector store directory: {settings.VECTORSTORE_DIR}")

    # TODO: Step 1 - Load documents
    # loader = DocumentLoader()
    # documents = loader.load_all_documents()
    # logger.info(f"Loaded {len(documents)} documents")

    logger.info("TODO: Implement document loading")

    # TODO: Step 2 - Clean documents (optional but recommended)
    # cleaner = TextCleaner()
    # cleaned_docs = cleaner.clean_documents(documents)

    logger.info("TODO: Implement document cleaning")

    # TODO: Step 3 - Chunk documents
    # chunker = DocumentChunker()
    # chunks = chunker.chunk_documents(cleaned_docs)
    # logger.info(f"Created {len(chunks)} chunks")
    # stats = chunker.get_chunk_statistics(chunks)
    # logger.info(f"Chunk statistics: {stats}")

    logger.info("TODO: Implement document chunking")

    # TODO: Step 4 - Create embeddings and vector store
    # embedding_model = EmbeddingModel()
    # retriever = BaseRetriever(embedding_model=embedding_model)
    # vectorstore = retriever.create_vectorstore(chunks)
    # logger.info("Vector store created and persisted")

    logger.info("TODO: Implement vector store creation")

    logger.info("=" * 60)
    logger.info("Ingestion Complete!")
    logger.info("=" * 60)
    logger.info("Next step: Run 'python pipelines/query_baseline.py' to test queries")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)
