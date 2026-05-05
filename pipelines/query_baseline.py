import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings import EmbeddingModel
from src.generation import LLMGenerator
from src.retrieval import BaseRetriever


def build_rag(top_k=None, use_local=False):
    embedding_model = EmbeddingModel()
    retriever = BaseRetriever(embedding_model=embedding_model, top_k=top_k)
    retriever.load_vectorstore()
    lc_retriever = retriever.as_retriever()

    generator = LLMGenerator(use_local=use_local)
    chain = generator.build_rag_chain(lc_retriever)
    return chain


def query_rag(question, chain):
    result = chain.invoke({"input": question})
    return result["answer"]


def interactive_mode(use_local=False):
    chain = build_rag(use_local=use_local)

    while True:
        question = input("Question: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            break

        if not question:
            continue

        try:
            answer = query_rag(question, chain)
            print(f"\nAnswer: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def single_query_mode(question, use_local=False):
    chain = build_rag(use_local=use_local)
    answer = query_rag(question, chain)
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Query the Baseline RAG system")
    parser.add_argument(
        "--question", "-q",
        help="Single question to answer (omit to enter interactive mode)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local CPU model instead of the HuggingFace API",
    )
    args = parser.parse_args()

    if args.question:
        single_query_mode(args.question, use_local=args.local)
    else:
        interactive_mode(use_local=args.local)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Query pipeline failed: {e}")
        sys.exit(1)
