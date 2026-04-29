"""
evaluation/evaluator.py
-----------------------
Head-to-head evaluation of Baseline RAG vs Graph-Enhanced RAG.

Metrics collected per question, per system:
  - retrieval_time_s   : seconds spent retrieving context
  - generation_time_s  : seconds spent generating the answer
  - total_time_s       : retrieval + generation
  - answer             : the generated answer text
  - answer_len         : word count of the answer
  - relevance_score    : LLM-as-judge score 1-5 (how well the answer addresses the question)
  - faithfulness_score : LLM-as-judge score 1-5 (does the answer stick to the retrieved context?)

Aggregate summary per system:
  - avg / std of all timing and score metrics
  - win-rate comparison (which system scores higher on relevance / faithfulness per question)

Usage:
    python evaluation/evaluator.py [--questions N] [--save-dir PATH]
"""

import json
import statistics
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings
from src.embeddings import EmbeddingModel
from src.generation import LLMGenerator
from src.retrieval import BaseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.utils import format_docs, load_json


# ---------------------------------------------------------------------------
# LLM-as-judge helpers
# ---------------------------------------------------------------------------

JUDGE_RELEVANCE_PROMPT = """You are an impartial evaluator.

Question:
{question}

Expected answer (reference):
{expected}

System answer:
{answer}

Rate how well the System answer addresses the Question, given the Expected answer as a reference.
Score on a scale of 1 to 5:
1 = completely wrong or irrelevant
2 = mostly wrong, touches the topic but misses key points
3 = partially correct, captures some key points
4 = mostly correct, minor omissions or inaccuracies
5 = fully correct and complete

Reply with ONLY a single integer (1, 2, 3, 4, or 5). No explanation."""

JUDGE_FAITHFULNESS_PROMPT = """You are an impartial evaluator.

Retrieved context:
{context}

System answer:
{answer}

Rate how faithful the System answer is to the Retrieved context — i.e. does the answer only use information present in the context?
Score on a scale of 1 to 5:
1 = answer contains mostly hallucinated facts not in the context
2 = answer contains several facts not in the context
3 = answer is partially grounded, some unsupported claims
4 = answer is mostly grounded, minor unsupported additions
5 = answer is fully grounded in the context

Reply with ONLY a single integer (1, 2, 3, 4, or 5). No explanation."""


def _parse_score(raw: str) -> int:
    """Extract a 1-5 integer from a raw LLM judge response."""
    for ch in raw.strip():
        if ch.isdigit() and ch in "12345":
            return int(ch)
    return 3  # fallback neutral score if parse fails


def judge_answer(generator: LLMGenerator, question: str, expected: str, answer: str, context: str) -> dict:
    """Run both judge prompts and return relevance + faithfulness scores."""
    rel_prompt = JUDGE_RELEVANCE_PROMPT.format(
        question=question, expected=expected, answer=answer
    )
    faith_prompt = JUDGE_FAITHFULNESS_PROMPT.format(
        context=context, answer=answer
    )

    rel_raw = generator.generate_simple(rel_prompt)
    faith_raw = generator.generate_simple(faith_prompt)

    return {
        "relevance_score": _parse_score(rel_raw),
        "faithfulness_score": _parse_score(faith_raw),
    }


# ---------------------------------------------------------------------------
# Baseline RAG evaluation
# ---------------------------------------------------------------------------

def evaluate_baseline(questions: list[dict], generator: LLMGenerator) -> list[dict]:
    print("\n[Baseline RAG] Loading retriever …")
    embedding_model = EmbeddingModel()
    retriever = BaseRetriever(embedding_model=embedding_model)
    retriever.load_vectorstore()

    records = []
    n = len(questions)

    for i, item in enumerate(questions, 1):
        q = item["question"]
        expected = item.get("expected_answer", "")
        print(f"  [{i}/{n}] {q[:70]}…")

        # --- retrieval ---
        t0 = time.perf_counter()
        docs = retriever.retrieve(q)
        retrieval_time = time.perf_counter() - t0

        context_text = format_docs(docs) if docs else ""

        # --- generation ---
        t1 = time.perf_counter()
        answer = generator.generate_answer(q, docs)
        generation_time = time.perf_counter() - t1

        # --- judge ---
        scores = judge_answer(generator, q, expected, answer, context_text)

        records.append({
            "id": item.get("id", i),
            "question": q,
            "category": item.get("category", ""),
            "retrieval_time_s": round(retrieval_time, 4),
            "generation_time_s": round(generation_time, 4),
            "total_time_s": round(retrieval_time + generation_time, 4),
            "answer": answer,
            "answer_len": len(answer.split()),
            **scores,
        })

    return records


# ---------------------------------------------------------------------------
# Graph RAG evaluation
# ---------------------------------------------------------------------------

GRAPH_RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about technical documentation.\n\n"
    "You have been given TWO sources of context:\n\n"
    "1. DOCUMENT CONTEXT — exact passages retrieved from the documentation.\n"
    "2. KNOWLEDGE GRAPH CONTEXT — relationships between key concepts/entities "
    "extracted from the documentation.\n\n"
    "Synthesise BOTH sources to give a complete, accurate answer.\n"
    "If the answer is not in either source, say so — do not make things up.\n\n"
    "Document Context:\n{context}\n\n"
    "Knowledge Graph Context:\n{graph_context}"
)


def evaluate_graph_rag(questions: list[dict], generator: LLMGenerator) -> list[dict]:
    graph_path = settings.GRAPH_DIR / "knowledge_graph.graphml"
    if not graph_path.exists():
        print(
            "\n[Graph RAG] Knowledge graph not found at:\n"
            f"  {graph_path}\n"
            "  Run 'python pipelines/ingest_baseline.py' first and choose the 'sample' directory.\n"
        )
        sys.exit(1)

    print("\n[Graph RAG] Loading retriever and knowledge graph …")
    embedding_model = EmbeddingModel()
    base_retriever = BaseRetriever(embedding_model=embedding_model)
    base_retriever.load_vectorstore()
    graph_retriever = GraphRetriever()

    records = []
    n = len(questions)

    for i, item in enumerate(questions, 1):
        q = item["question"]
        expected = item.get("expected_answer", "")
        print(f"  [{i}/{n}] {q[:70]}…")

        # --- retrieval (vector + graph) ---
        t0 = time.perf_counter()
        vector_docs = base_retriever.retrieve(q)
        graph_docs = graph_retriever.retrieve(q)
        retrieval_time = time.perf_counter() - t0

        vector_context = format_docs(vector_docs) if vector_docs else "No document context found."
        graph_context = (
            "\n".join(doc.page_content for doc in graph_docs)
            if graph_docs
            else "No graph relationships found for this query."
        )

        # --- generation: embed all context directly so PromptTemplate only sees {input} ---
        pre_filled_template = (
            "Answer the following question using ONLY the context below.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            "Document Context:\n" + vector_context + "\n\n"
            "Knowledge Graph Context:\n" + graph_context + "\n\n"
            "Question: {input}\n\nAnswer:"
        )
        t1 = time.perf_counter()
        answer = generator.generate_answer(q, [], prompt_template=pre_filled_template)
        generation_time = time.perf_counter() - t1

        combined_context = f"{vector_context}\n\n--- Knowledge Graph ---\n{graph_context}"

        # --- judge ---
        scores = judge_answer(generator, q, expected, answer, combined_context)

        records.append({
            "id": item.get("id", i),
            "question": q,
            "category": item.get("category", ""),
            "retrieval_time_s": round(retrieval_time, 4),
            "generation_time_s": round(generation_time, 4),
            "total_time_s": round(retrieval_time + generation_time, 4),
            "answer": answer,
            "answer_len": len(answer.split()),
            **scores,
        })

    return records


# ---------------------------------------------------------------------------
# Aggregate + compare
# ---------------------------------------------------------------------------

def _agg(records: list[dict], field: str) -> dict:
    vals = [r[field] for r in records]
    mean = statistics.mean(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": round(mean, 4), "stdev": round(stdev, 4), "min": round(min(vals), 4), "max": round(max(vals), 4)}


def aggregate(records: list[dict]) -> dict:
    return {
        "n": len(records),
        "retrieval_time_s": _agg(records, "retrieval_time_s"),
        "generation_time_s": _agg(records, "generation_time_s"),
        "total_time_s": _agg(records, "total_time_s"),
        "answer_len": _agg(records, "answer_len"),
        "relevance_score": _agg(records, "relevance_score"),
        "faithfulness_score": _agg(records, "faithfulness_score"),
    }


def compute_win_rates(baseline: list[dict], graph: list[dict]) -> dict:
    """Per-question comparison; returns win/tie/loss counts for graph vs baseline."""
    rel_wins = rel_ties = rel_losses = 0
    faith_wins = faith_ties = faith_losses = 0

    for b, g in zip(baseline, graph):
        db = g["relevance_score"] - b["relevance_score"]
        if db > 0:
            rel_wins += 1
        elif db == 0:
            rel_ties += 1
        else:
            rel_losses += 1

        df = g["faithfulness_score"] - b["faithfulness_score"]
        if df > 0:
            faith_wins += 1
        elif df == 0:
            faith_ties += 1
        else:
            faith_losses += 1

    n = len(baseline)
    return {
        "relevance": {
            "graph_wins": rel_wins,
            "ties": rel_ties,
            "baseline_wins": rel_losses,
            "graph_win_rate": round(rel_wins / n, 3),
        },
        "faithfulness": {
            "graph_wins": faith_wins,
            "ties": faith_ties,
            "baseline_wins": faith_losses,
            "graph_win_rate": round(faith_wins / n, 3),
        },
    }


def print_summary(baseline_agg: dict, graph_agg: dict, win_rates: dict):
    def arrow(g_val, b_val, higher_is_better=True):
        diff = g_val - b_val
        if abs(diff) < 1e-6:
            return "  =="
        if (diff > 0) == higher_is_better:
            return f"  ▲ +{abs(diff):.4f}"
        return f"  ▼ -{abs(diff):.4f}"

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS: Baseline RAG  vs  Graph-Enhanced RAG")
    print("=" * 70)

    metrics = [
        ("Avg Retrieval Time (s)",   "retrieval_time_s",   False),
        ("Avg Generation Time (s)",  "generation_time_s",  False),
        ("Avg Total Time (s)",       "total_time_s",       False),
        ("Avg Answer Length (words)","answer_len",         True),
        ("Avg Relevance Score /5",   "relevance_score",    True),
        ("Avg Faithfulness Score /5","faithfulness_score", True),
    ]

    fmt = "  {:<28} {:>12}  {:>12}  {}"
    print(fmt.format("Metric", "Baseline", "Graph RAG", "Δ (Graph−Base)"))
    print("  " + "-" * 66)
    for label, key, higher in metrics:
        b = baseline_agg[key]["mean"]
        g = graph_agg[key]["mean"]
        print(fmt.format(label, f"{b:.4f}", f"{g:.4f}", arrow(g, b, higher)))

    print()
    print("  Win-rate breakdown (Graph RAG vs Baseline, per question):")
    for aspect, wr in win_rates.items():
        print(f"    {aspect.capitalize():<15}  "
              f"Graph wins: {wr['graph_wins']}  "
              f"Ties: {wr['ties']}  "
              f"Baseline wins: {wr['baseline_wins']}  "
              f"(Graph win-rate: {wr['graph_win_rate']:.1%})")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Head-to-head evaluation: Baseline RAG vs Graph-Enhanced RAG"
    )
    parser.add_argument(
        "--questions", "-n",
        type=int,
        default=None,
        help="Number of questions to evaluate (default: all)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=project_root / "evaluation",
        help="Directory to save JSON results (default: evaluation/)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline evaluation (re-use previous results_baseline.json)",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip graph RAG evaluation (re-use previous results_graph.json)",
    )
    args = parser.parse_args()

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load questions
    test_path = project_root / "evaluation" / "test_questions.json"
    data = load_json(test_path)
    questions = data.get("questions", [])
    if args.questions:
        questions = questions[: args.questions]
    print(f"\nLoaded {len(questions)} test questions from {test_path}")

    # Shared LLM (used for generation + judging)
    generator = LLMGenerator()

    # --- Baseline ---
    baseline_out = save_dir / "results_baseline.json"
    if args.skip_baseline and baseline_out.exists():
        print(f"\n[Baseline RAG] Skipping — loading cached results from {baseline_out}")
        with open(baseline_out) as f:
            baseline_data = json.load(f)
        baseline_records = baseline_data.get("per_question", [])
    else:
        baseline_records = evaluate_baseline(questions, generator)
        with open(baseline_out, "w") as f:
            json.dump(
                {"per_question": baseline_records, "aggregate": aggregate(baseline_records)},
                f, indent=2,
            )
        print(f"\n[Baseline RAG] Results saved → {baseline_out}")

    # --- Graph RAG ---
    graph_out = save_dir / "results_graph.json"
    if args.skip_graph and graph_out.exists():
        print(f"\n[Graph RAG] Skipping — loading cached results from {graph_out}")
        with open(graph_out) as f:
            graph_data = json.load(f)
        graph_records = graph_data.get("per_question", [])
    else:
        graph_records = evaluate_graph_rag(questions, generator)
        with open(graph_out, "w") as f:
            json.dump(
                {"per_question": graph_records, "aggregate": aggregate(graph_records)},
                f, indent=2,
            )
        print(f"\n[Graph RAG] Results saved → {graph_out}")

    # --- Comparison ---
    baseline_agg = aggregate(baseline_records)
    graph_agg = aggregate(graph_records)
    win_rates = compute_win_rates(baseline_records, graph_records)

    comparison = {
        "n_questions": len(questions),
        "baseline": baseline_agg,
        "graph_rag": graph_agg,
        "win_rates": win_rates,
    }
    comparison_out = save_dir / "results_comparison.json"
    with open(comparison_out, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparison saved → {comparison_out}")

    print_summary(baseline_agg, graph_agg, win_rates)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted.")
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nEvaluation failed: {e}")
        sys.exit(1)
