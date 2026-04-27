import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_json, save_json


class RAGEvaluator:
    def __init__(self, test_questions_path=None):
        self.test_questions_path = (
            test_questions_path
            or project_root / "evaluation" / "test_questions.json"
        )
        self.test_data = None

    def load_test_questions(self):
        self.test_data = load_json(self.test_questions_path)
        print(f"Loaded {len(self.test_data.get('questions', []))} test questions")

    def evaluate_retrieval(self, retriever, questions):
        times = []
        hits = 0

        for item in questions:
            query = item["question"]
            expected_keywords = item.get("keywords", [])

            start = time.perf_counter()
            docs = retriever.retrieve(query)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            retrieved_text = " ".join(d.page_content for d in docs).lower()
            if any(kw.lower() in retrieved_text for kw in expected_keywords):
                hits += 1

        n = len(questions) or 1
        return {
            "avg_retrieval_time": round(sum(times) / n, 4),
            "top_k_accuracy": round(hits / n, 4),
        }

    def evaluate_generation(self, chain, questions):
        times = []

        for item in questions:
            query = item["question"]
            start = time.perf_counter()
            chain.invoke({"input": query})
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        n = len(questions) or 1
        return {
            "avg_generation_time": round(sum(times) / n, 4),
        }

    def run_full_evaluation(self, retriever, chain):
        print("\nRunning Full RAG Evaluation")
        print("-" * 60)

        if self.test_data is None:
            self.load_test_questions()

        questions = self.test_data.get("questions", [])

        retrieval_metrics = self.evaluate_retrieval(retriever, questions)
        generation_metrics = self.evaluate_generation(chain, questions)

        results = {
            "retrieval": retrieval_metrics,
            "generation": generation_metrics,
            "overall": {
                "total_questions": len(questions),
            },
        }

        print("Evaluation complete:", results)
        return results

    def save_results(self, results, output_path):
        save_json(results, output_path)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument(
        "--mode",
        choices=["baseline", "graph"],
        default="baseline",
        help="Which RAG mode to evaluate",
    )
    args = parser.parse_args()

    print(f"Evaluating {args.mode} RAG...")

    if args.mode == "baseline":
        from src.embeddings import EmbeddingModel
        from src.generation import LLMGenerator
        from src.retrieval import BaseRetriever

        embedding_model = EmbeddingModel()
        retriever = BaseRetriever(embedding_model=embedding_model)
        retriever.load_vectorstore()
        lc_retriever = retriever.as_retriever()

        generator = LLMGenerator()
        chain = generator.build_rag_chain(lc_retriever)

        evaluator = RAGEvaluator()
        results = evaluator.run_full_evaluation(retriever, chain)
        evaluator.save_results(
            results,
            project_root / "evaluation" / "results_baseline.json",
        )
