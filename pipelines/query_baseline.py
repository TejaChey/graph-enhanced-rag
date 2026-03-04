import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings
from src.embeddings import EmbeddingModel
from src.generation import LLMGenerator
from src.retrieval import BaseRetriever


def build_rag(top_k=None):
    embedding_model = EmbeddingModel()
    retriever = BaseRetriever(embedding_model=embedding_model, top_k=top_k)
    retriever.load_vectorstore()
    lc_retriever = retriever.as_retriever()

    generator = LLMGenerator()
    chain = generator.build_rag_chain(lc_retriever)
    return chain


def query_rag(question, chain):
    result = chain.invoke({"input": question})
    return result["answer"]


def interactive_mode():
    print("=" * 60)
    print("Baseline RAG - Interactive Mode")
    print("=" * 60)

    print("Initialising RAG chain...")
    chain = build_rag()
    print("RAG system ready!\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            answer = query_rag(question, chain)
            print(f"\nAnswer: {answer}\n")
            print("-" * 60)
        except Exception as e:
            print(f"Error: {e}\n")


def single_query_mode(question):
    print("Initialising RAG chain...")
    chain = build_rag()

    answer = query_rag(question, chain)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Query the Baseline RAG system")
    parser.add_argument(
        "--question", "-q",
        help="Single question to answer (omit to enter interactive mode)",
    )
    args = parser.parse_args()

    if args.question:
        single_query_mode(args.question)
    else:
        interactive_mode()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Query pipeline failed: {e}")
        sys.exit(1)
