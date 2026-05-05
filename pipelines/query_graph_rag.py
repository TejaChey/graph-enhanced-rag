import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings
from src.embeddings import EmbeddingModel
from src.generation import LLMGenerator
from src.retrieval import BaseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.utils import format_docs

GRAPH_RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about technical documentation.\n\n"
    "You have been given TWO sources of context:\n\n"
    "1. DOCUMENT CONTEXT — exact passages retrieved from the documentation.\n"
    "2. KNOWLEDGE GRAPH CONTEXT — relationships between key concepts/entities "
    "extracted from the documentation (e.g. 'NumPy is related to: arrays, Python, SciPy').\n\n"
    "Synthesise BOTH sources to give a complete, accurate answer.\n"
    "If the answer is not in either source, say so — do not make things up.\n\n"
    "Document Context:\n{context}\n\n"
    "Knowledge Graph Context:\n{graph_context}"
)


def build_graph_rag(top_k=None, use_local=False):
    # Vector retriever (ChromaDB)
    embedding_model = EmbeddingModel()
    base_retriever = BaseRetriever(embedding_model=embedding_model, top_k=top_k)
    base_retriever.load_vectorstore()

    # Graph retriever (NetworkX knowledge graph)
    graph_retriever = GraphRetriever()

    generator = LLMGenerator(use_local=use_local)
    return base_retriever, graph_retriever, generator


def query_graph_rag(question: str, base_retriever, graph_retriever, generator) -> dict:
    vector_docs = base_retriever.retrieve(question)
    graph_docs = graph_retriever.retrieve(question)

    vector_context = format_docs(vector_docs) if vector_docs else "No document context found."
    graph_context = (
        "\n".join(doc.page_content for doc in graph_docs)
        if graph_docs
        else "No graph relationships found for this query."
    )

    prompt_with_graph = GRAPH_RAG_SYSTEM_PROMPT.format(
        context=vector_context,
        graph_context=graph_context,
    )
    answer = generator.generate_answer(
        query=question,
        context_docs=vector_docs,
        prompt_template=prompt_with_graph,
    )

    return {
        "answer": answer,
        "vector_docs": vector_docs,
        "graph_docs": graph_docs,
    }


def interactive_mode(use_local=False):
    graph_path = settings.GRAPH_DIR / "knowledge_graph.graphml"
    if not graph_path.exists():
        print("Knowledge graph not found. Run 'python pipelines/ingest.py' first.")
        sys.exit(1)

    base_retriever, graph_retriever, generator = build_graph_rag(use_local=use_local)

    while True:
        question = input("Question: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            break

        if not question:
            continue

        try:
            result = query_graph_rag(question, base_retriever, graph_retriever, generator)
            print(f"\nAnswer: {result['answer']}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def single_query_mode(question: str, use_local=False):
    graph_path = settings.GRAPH_DIR / "knowledge_graph.graphml"
    if not graph_path.exists():
        print("Knowledge graph not found. Run ingest.py first.")
        sys.exit(1)

    base_retriever, graph_retriever, generator = build_graph_rag(use_local=use_local)
    result = query_graph_rag(question, base_retriever, graph_retriever, generator)
    print(f"\nQuestion: {question}")
    print(f"Answer: {result['answer']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Query using Graph-Enhanced RAG")
    parser.add_argument(
        "--question", "-q",
        help="Single question to answer (omit for interactive mode)",
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
        print(f"Graph RAG query failed: {e}")
        sys.exit(1)
