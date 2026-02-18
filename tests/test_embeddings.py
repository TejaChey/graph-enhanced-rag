from src.embeddings import EmbeddingModel


def test_embedding_initialization():
    print("\n" + "="*60)
    print("Testing Embedding Model Initialization")
    print("="*60)

    model = EmbeddingModel()

    print(f"Model name: {model.model_name}")
    print(f"Device: {model.device}")

    return model


def test_query_embedding(model):
    print("\n" + "="*60)
    print("Testing Query Embedding")
    print("="*60)

    test_query = "What is LangChain?"

    embedding = model.embed_query(test_query)

    print(f"Query: {test_query}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    return embedding


def test_document_embedding(model):
    print("\n" + "="*60)
    print("Testing Document Embeddings")
    print("="*60)

    test_docs = [
        "LangChain is a framework for developing applications with LLMs.",
        "RAG stands for Retrieval-Augmented Generation.",
        "Vector databases store embeddings for similarity search."
    ]

    embeddings = model.embed_documents(test_docs)

    print(f"Number of documents: {len(test_docs)}")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")

    return embeddings


def test_embedding_similarity(model):
    print("\n" + "="*60)
    print("Testing Embedding Similarity")
    print("="*60)

    # Similar texts
    text1 = "LangChain is a framework for LLMs"
    text2 = "LangChain helps build LLM applications"
    text3 = "The weather is sunny today"

    emb1 = model.embed_query(text1)
    emb2 = model.embed_query(text2)
    emb3 = model.embed_query(text3)

    # Simple cosine similarity
    import numpy as np

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_12 = cosine_similarity(emb1, emb2)
    sim_13 = cosine_similarity(emb1, emb3)

    print(f"Similarity (text1 vs text2): {sim_12:.4f}")
    print(f"Similarity (text1 vs text3): {sim_13:.4f}")

    if sim_12 > sim_13:
        print("Similar texts are more similar than different texts!")
    else:
        print("Warning: Expected similar texts to be more similar")


def main():
    print("\n" + "="*60)
    print("EMBEDDING MODEL TESTS")
    print("="*60)

    model = test_embedding_initialization()
    test_query_embedding(model)
    test_document_embedding(model)
    test_embedding_similarity(model)

    print("\nEmbedding tests complete!")


if __name__ == "__main__":
    main()
