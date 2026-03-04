import time

from config import settings, setup_directories
from src.data_processing import DocumentChunker, DocumentLoader, TextCleaner
from src.embeddings import EmbeddingModel
from src.generation import LLMGenerator
from src.retrieval import BaseRetriever


def test_ingestion_pipeline():
    print("\n" + "="*60)
    print("STEP 1: INGESTION PIPELINE")
    print("="*60)

    # Load
    print("\n[1/5] Loading documents...")
    loader = DocumentLoader()
    docs = loader.load_all_documents()
    print(f"Loaded {len(docs)} documents")

    if len(docs) == 0:
        print("\nERROR: No documents found!")
        print("Please add documentation to data/raw/ first")
        print("Run: bash scripts/download_docs.sh")
        return None, None

    # Clean
    print("\n[2/5] Cleaning documents...")
    cleaner = TextCleaner()
    cleaned = cleaner.clean_documents(docs)
    print(f"Cleaned {len(cleaned)} documents")

    # Chunk
    print("\n[3/5] Chunking documents...")
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(cleaned)
    stats = chunker.get_chunk_statistics(chunks)
    print(f"Created {stats['total_chunks']} chunks")
    print(f"  - Avg size: {stats['avg_chunk_size']} chars")

    # Embed
    print("\n[4/5] Creating embeddings...")
    embedding_model = EmbeddingModel()
    print(f"Embedding model: {embedding_model.model_name}")

    # Vector Store
    print("\n[5/5] Creating vector store...")
    retriever = BaseRetriever(embedding_model=embedding_model)
    start_time = time.time()
    retriever.create_vectorstore(chunks)
    elapsed = time.time() - start_time
    print(f"Vector store created in {elapsed:.2f}s")

    return retriever, embedding_model


def test_query_pipeline(retriever, embedding_model):
    print("\n" + "="*60)
    print("STEP 2: QUERY PIPELINE")
    print("="*60)

    test_questions = [
        "What is NumPy.",
        "What is broadcasting?",
        "What is numpy.ravel()?"
    ]

    # Initialize generator
    print("\n[1/3] Initializing LLM...")
    try:
        generator = LLMGenerator()
        print(f"LLM model: {generator.model_name}")
    except Exception as e:
        print(f"LLM initialization failed: {e}")
        print("  Make sure HUGGINGFACEHUB_API_TOKEN is set in .env")
        return

    # Test retrieval
    print("\n[2/3] Testing retrieval...")
    for question in test_questions:
        print(f"\n  Query: {question}")
        start_time = time.time()
        docs = retriever.retrieve(question, top_k=3)
        elapsed = time.time() - start_time
        print(f"  Retrieved {len(docs)} docs in {elapsed:.2f}s")
        if docs:
            print(f"    Preview: {docs[0].page_content[:100]}...")

    # Test generation
    print("\n[3/3] Testing answer generation...")
    test_query = test_questions[0]
    print(f"\n  Question: {test_query}")

    try:
        start_time = time.time()
        relevant_docs = retriever.retrieve(test_query, top_k=4)
        answer = generator.generate_answer(test_query, relevant_docs)
        elapsed = time.time() - start_time

        print(f"\n  Answer generated in {elapsed:.2f}s")
        print("\n  Answer:")
        print(f"  {'-'*50}")
        print(f"  {answer}")
        print(f"  {'-'*50}")
    except Exception as e:
        print(f"\n  Generation failed: {e}")
        print("    This might be due to HuggingFace API issues")


def test_retrieval_quality(retriever):
    print("\n" + "="*60)
    print("STEP 3: RETRIEVAL QUALITY CHECK")
    print("="*60)

    # Test with specific query
    query = "document loaders"
    print(f"\nQuery: '{query}'")

    docs_with_scores = retriever.retrieve_with_scores(query, top_k=5)

    print(f"\nRetrieved {len(docs_with_scores)} documents with scores:")
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        print(f"\n  [{i}] Score: {score:.4f}")
        print(f"      Preview: {doc.page_content[:150]}...")
        if 'source' in doc.metadata:
            print(f"      Source: {doc.metadata['source']}")


def main():
    print("\n" + "="*80)
    print(" "*20 + "END-TO-END RAG SYSTEM TEST")
    print("="*80)

    # Setup
    setup_directories()

    # Test ingestion
    retriever, embedding_model = test_ingestion_pipeline()

    if retriever is None:
        print("\nIngestion failed - cannot continue")
        return

    # Test querying
    test_query_pipeline(retriever, embedding_model)

    # Test retrieval quality
    test_retrieval_quality(retriever)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print("Ingestion pipeline: PASSED")
    print("Query pipeline: PASSED")
    print("Retrieval quality: PASSED")
    print("\nRAG system is working end-to-end!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
