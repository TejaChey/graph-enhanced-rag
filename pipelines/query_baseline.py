import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings
from src.embeddings import EmbeddingModel
from src.generation import LLMGenerator
from src.retrieval import BaseRetriever
from src.utils import setup_logger

logger = setup_logger(__name__, log_file=settings.LOG_DIR / "query_baseline.log")


def query_rag(question: str, retriever: BaseRetriever, generator: LLMGenerator) -> str:
    logger.info(f"Query: {question}")

    # TODO: Step 1 - Retrieve relevant documents
    # relevant_docs = retriever.retrieve(question)
    # logger.info(f"Retrieved {len(relevant_docs)} relevant documents")

    logger.info("TODO: Implement document retrieval")

    # TODO: Step 2 - Generate answer
    # answer = generator.generate_answer(question, relevant_docs)
    # logger.info("Answer generated")

    logger.info("TODO: Implement answer generation")

    # Placeholder return
    return "TODO: Implement query_rag function"


def interactive_mode():
    logger.info("=" * 60)
    logger.info("Baseline RAG - Interactive Mode")
    logger.info("=" * 60)

    # TODO: Initialize components
    # embedding_model = EmbeddingModel()
    # retriever = BaseRetriever(embedding_model=embedding_model)
    # retriever.load_vectorstore()
    # generator = LLMGenerator()

    logger.info("TODO: Implement component initialization")

    print("\nRAG System Ready! Type your questions (or 'quit' to exit)\n")

    # TODO: Implement query loop
    # while True:
    #     question = input("Question: ").strip()
    #
    #     if question.lower() in ['quit', 'exit', 'q']:
    #         print("Goodbye!")
    #         break
    #
    #     if not question:
    #         continue
    #
    #     try:
    #         answer = query_rag(question, retriever, generator)
    #         print(f"\nAnswer: {answer}\n")
    #         print("-" * 60)
    #     except Exception as e:
    #         logger.error(f"Query failed: {e}", exc_info=True)
    #         print(f"Error: {e}\n")

    print("TODO: Implement interactive query loop")


def single_query_mode(question: str):
    logger.info("Single query mode")

    # TODO: Initialize and query
    # embedding_model = EmbeddingModel()
    # retriever = BaseRetriever(embedding_model=embedding_model)
    # retriever.load_vectorstore()
    # generator = LLMGenerator()
    #
    # answer = query_rag(question, retriever, generator)
    # print(f"\nQuestion: {question}")
    # print(f"Answer: {answer}\n")

    print("TODO: Implement single query mode")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Query the Baseline RAG system")
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Single question to answer (if not provided, enters interactive mode)"
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
        logger.error(f"Query pipeline failed: {e}", exc_info=True)
        sys.exit(1)
