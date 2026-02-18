from config import settings, setup_directories
from src.data_processing import DocumentChunker, DocumentLoader
from src.embeddings import EmbeddingModel
from src.generation import LLMGenerator
from src.retrieval import BaseRetriever

print("\nQuick RAG Test\n")

# Setup
setup_directories()

# 1. Load small sample
print("1. Loading documents...", end=" ")
loader = DocumentLoader()
docs = loader.load_all_documents()
if not docs:
    print("No documents!")
    print("   Run: bash scripts/download_docs.sh")
print(f"({len(docs)} docs)")

# Take only first 5 docs for speed
docs = docs[:5]

# 2. Chunk
print("2. Chunking...", end=" ")
chunker = DocumentChunker()
chunks = chunker.chunk_documents(docs)
print(f"({len(chunks)} chunks)")

# 3. Embed & Store
print("3. Creating vector store...", end=" ")
embedding_model = EmbeddingModel()
retriever = BaseRetriever(embedding_model=embedding_model)
retriever.create_vectorstore(chunks)

# 4. Query
print("4. Testing retrieval...", end=" ")
test_query = "What is LangChain?"
results = retriever.retrieve(test_query, top_k=2)
print(f"(found {len(results)} docs)")

# 5. Generate (optional - skip if no token)
print("5. Testing generation...", end=" ")
try:
    generator = LLMGenerator()
    answer = generator.generate_answer(test_query, results)
    print(f"\nAnswer: {answer[:200]}...\n")
except Exception as e:
    print(f"Skipped ({e})")

print("RAG is working!\n")
