import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings, setup_directories
from src.data_processing import DocumentChunker, DocumentLoader, TextCleaner
from src.embeddings import EmbeddingModel
from src.graph import EntityExtractor, KnowledgeGraph
from src.retrieval import BaseRetriever


def main():
    print("=" * 60)
    print("Starting Baseline RAG Ingestion Pipeline")
    print("=" * 60)

    setup_directories()
    print(f"Data directory: {settings.RAW_DATA_DIR}")
    print(f"Vector store directory: {settings.VECTORSTORE_DIR}")

    # Step 1 - Load documents
    print("\nStep 1: Loading documents...")
    loader = DocumentLoader()
    documents = loader.load_all_documents()
    print(f"  Loaded {len(documents)} documents")

    if not documents:
        print("  No documents found. Run: bash scripts/download_docs.sh")
        return

    # Step 2 - Clean documents
    print("\nStep 2: Cleaning documents...")
    cleaner = TextCleaner()
    cleaned_docs = cleaner.clean_documents(documents)
    print(f"  Cleaned {len(cleaned_docs)} documents")

    # Step 3 - Chunk documents
    print("\nStep 3: Chunking documents...")
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(cleaned_docs)
    stats = chunker.get_chunk_statistics(chunks)
    print(f"  Created {len(chunks)} chunks")
    print(f"  Stats: {stats}")

    # Step 4 - Create embeddings and vector store
    print("\nStep 4: Creating vector store...")
    embedding_model = EmbeddingModel()
    retriever = BaseRetriever(embedding_model=embedding_model)
    retriever.create_vectorstore(chunks)
    print(f"  Vector store persisted to {settings.VECTORSTORE_DIR}")

    # Step 5 - Build knowledge graph with typed relationships
    print("\nStep 5: Building knowledge graph with typed relationships...")
    extractor = EntityExtractor()
    kg = KnowledgeGraph()

    print("  Extracting entities and classifying relationships (this may take a moment)...")
    chunks_with_context = extractor.extract_entities_with_context(chunks)

    kg.build_from_chunks_with_relations(chunks_with_context)
    graph_stats = kg.get_stats()
    print(f"  Graph nodes (unique entities): {graph_stats['nodes']}")
    print(f"  Graph edges (relationships):   {graph_stats['edges']}")
    print(f"  Graph density:                 {graph_stats['density']}")

    rel_summary = kg.get_relation_type_summary()
    print(f"  Relationship types breakdown:")
    for rel_type, count in sorted(rel_summary.items(), key=lambda x: -x[1]):
        print(f"    {rel_type:<15} {count} edges")

    top_entities = kg.get_top_entities(n=10)
    print(f"  Top entities by connections:")
    for entity, degree in top_entities:
        node_type = kg.graph.nodes[entity].get("type", "?")
        print(f"    [{node_type}] {entity!r} — {degree} connections")

    saved_path = kg.save()
    print(f"  Knowledge graph saved to {saved_path}")

    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print("Next step: Run 'python pipelines/query_baseline.py'")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Ingestion failed: {e}")
        sys.exit(1)
