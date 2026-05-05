import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings, setup_directories
from src.data_processing import DocumentChunker, DocumentLoader, TextCleaner
from src.embeddings import EmbeddingModel
from src.graph import EntityExtractor, KnowledgeGraph
from src.retrieval import BaseRetriever


def resolve_data_dir(data_dir_arg: str | None) -> Path:
    if data_dir_arg is None:
        # Default: prefer data/sample
        default = settings.DATA_DIR / "sample"
        return default if default.exists() else settings.DATA_DIR

    candidate = Path(data_dir_arg)
    if not candidate.is_absolute():
        candidate = settings.DATA_DIR / data_dir_arg
    if not candidate.exists():
        print(f"ERROR: '{candidate}' does not exist.")
        sys.exit(1)
    return candidate


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into Baseline RAG + Knowledge Graph")
    parser.add_argument(
        "--data-dir", "-d",
        default=None,
        help=(
            "Directory (or subdirectory name under data/) to ingest. "
            "Defaults to data/sample if it exists, otherwise data/."
        ),
    )
    args = parser.parse_args()

    print("\nStarting Baseline RAG Ingestion Pipeline")
    print("-" * 40)

    setup_directories()
    data_dir = resolve_data_dir(args.data_dir)
    print(f"Data directory: {data_dir}")
    print(f"Vector store directory: {settings.VECTORSTORE_DIR}")

    print("\nLoading documents...")
    loader = DocumentLoader(data_dir=data_dir)
    documents = loader.load_all_documents()
    print(f"  Loaded {len(documents)} documents")

    if not documents:
        print("  No documents found. Add files to data/")
        return

    print("\nCleaning documents...")
    cleaner = TextCleaner()
    cleaned_docs = cleaner.clean_documents(documents)
    print(f"  Cleaned {len(cleaned_docs)} documents")

    print("\nChunking documents...")
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(cleaned_docs)
    stats = chunker.get_chunk_statistics(chunks)
    print(f"  Created {len(chunks)} chunks")
    print(f"  Stats: {stats}")

    print("\nCreating vector store...")
    embedding_model = EmbeddingModel()
    retriever = BaseRetriever(embedding_model=embedding_model)
    retriever.create_vectorstore(chunks)
    print(f"  Vector store persisted to {settings.VECTORSTORE_DIR}")

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

    print("\nIngestion complete! Ready for querying.")
    print("Next step: Run 'python pipelines/query_graph_rag.py'")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Ingestion failed: {e}")
        sys.exit(1)
