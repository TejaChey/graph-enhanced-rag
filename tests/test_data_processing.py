from config import setup_directories
from src.data_processing import DocumentChunker, DocumentLoader, TextCleaner


def test_loader():
    print("\n" + "="*60)
    print("Testing Document Loader")
    print("="*60)

    loader = DocumentLoader()
    docs = loader.load_all_documents()

    print(f"Loaded {len(docs)} documents")

    if docs:
        print(f"First doc preview: {docs[0].page_content[:100]}...")
        print(f"Metadata keys: {list(docs[0].metadata.keys())}")
    else:
        print("No documents found. Add files to data/")
        return docs


def test_cleaner(documents):
    print("\n" + "="*60)
    print("Testing Text Cleaner")
    print("="*60)

    if not documents:
        print("Skipping - no documents to clean")
        return documents

    cleaner = TextCleaner()

    # Test single document
    test_doc = documents[0]
    original_len = len(test_doc.page_content)
    cleaned = cleaner.clean_document(test_doc)
    new_len = len(cleaned.page_content)

    print(f"Original length: {original_len}")
    print(f"Cleaned length: {new_len}")
    print(f"Reduction: {original_len - new_len} chars")

    # Clean all
    all_cleaned = cleaner.clean_documents(documents)
    print(f"Cleaned {len(all_cleaned)} documents")

    return all_cleaned


def test_chunker(documents):
    print("\n" + "="*60)
    print("Testing Document Chunker")
    print("="*60)

    if not documents:
        print("Skipping - no documents to chunk")
        return []

    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)
    stats = chunker.get_chunk_statistics(chunks)

    print(f"Created {stats['total_chunks']} chunks")
    print(f"Average size: {stats['avg_chunk_size']} chars")
    print(f"Min size: {stats['min_chunk_size']} chars")
    print(f"Max size: {stats['max_chunk_size']} chars")

    if chunks:
        print(f"\nSample chunk:")
        print(f"  {chunks[0].page_content[:200]}...")

    return chunks


def main():
    print("\n" + "="*60)
    print("DATA PROCESSING TESTS")
    print("="*60)

    # Setup
    setup_directories()

    # Test pipeline
    docs = test_loader()
    cleaned_docs = test_cleaner(docs)
    chunks = test_chunker(cleaned_docs)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Documents loaded: {len(docs)}")
    print(f"Documents cleaned: {len(cleaned_docs)}")
    print(f"Chunks created: {len(chunks)}")
    print("\nData processing tests complete!")


if __name__ == "__main__":
    main()
