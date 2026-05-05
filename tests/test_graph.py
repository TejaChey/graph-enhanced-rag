import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import settings
from src.graph import EntityExtractor, KnowledgeGraph


def test_entity_extractor():
    print("\n" + "=" * 60)
    print("Testing EntityExtractor")
    print("=" * 60)

    extractor = EntityExtractor()
    sample = (
        "NumPy is a Python library developed by Travis Oliphant at Enthought. "
        "It provides support for large arrays and is widely used in SciPy and Pandas."
    )
    entities = extractor.extract_entities(sample)
    print(f"  Text: {sample[:80]}...")
    print(f"  Entities found: {entities}")
    assert isinstance(entities, list), "Expected list"
    print("  PASSED")
    return entities


def test_knowledge_graph(entities):
    print("\n" + "=" * 60)
    print("Testing KnowledgeGraph build & stats")
    print("=" * 60)

    kg = KnowledgeGraph()
    kg.add_entities_from_chunk(entities)

    stats = kg.get_stats()
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Density: {stats['density']}")

    assert stats["nodes"] >= 0, "Node count should be non-negative"
    print("  PASSED")
    return kg


def test_graph_save_load(kg):
    print("\n" + "=" * 60)
    print("Testing graph save & load")
    print("=" * 60)

    saved = kg.save(filename="test_graph.graphml")
    print(f"  Saved to: {saved}")
    assert saved.exists(), "Graph file should exist after save"

    kg2 = KnowledgeGraph()
    kg2.load(filename="test_graph.graphml")

    assert kg2.graph.number_of_nodes() == kg.graph.number_of_nodes(), \
        "Loaded graph should have same nodes"
    print(f"  Loaded {kg2.graph.number_of_nodes()} nodes successfully")
    print("  PASSED")

    # Clean up test file
    saved.unlink()
    print("  Test file cleaned up.")


def test_saved_production_graph():
    print("\n" + "=" * 60)
    print("Testing production graph (from ingestion)")
    print("=" * 60)

    graph_path = settings.GRAPH_DIR / "knowledge_graph.graphml"
    if not graph_path.exists():
        print("  SKIP — No production graph found. Run ingest.py first.")
        return

    kg = KnowledgeGraph()
    kg.load()
    stats = kg.get_stats()
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Top entities: {kg.get_top_entities(5)}")
    assert stats["nodes"] > 0, "Graph should have at least one node"
    print("  PASSED")


def main():
    print("\n" + "=" * 60)
    print("KNOWLEDGE GRAPH TESTS")
    print("=" * 60)

    entities = test_entity_extractor()
    kg = test_knowledge_graph(entities)
    test_graph_save_load(kg)
    test_saved_production_graph()

    print("\n" + "=" * 60)
    print("All graph tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
