from pathlib import Path

import networkx as nx

from config import settings


class KnowledgeGraph:
    """
    A knowledge graph built from document chunks using NetworkX.

    Nodes  = unique named entities (with a 'type' attribute for entity label).
    Edges  = co-occurrence in the same text chunk, with attributes:
               - 'weight'       : how often the pair co-occurred
               - 'relation_type': e.g. uses | part_of | extends | depends_on |
                                  implements | related_to

    The graph can be persisted to / loaded from a GraphML file.
    """

    def __init__(self, graph_dir=None):
        self.graph_dir = Path(graph_dir) if graph_dir else settings.GRAPH_DIR
        self.graph = nx.Graph()

    # ------------------------------------------------------------------
    # Building the graph
    # ------------------------------------------------------------------

    def add_entities_from_chunk(self, entities: list[tuple[str, str]]) -> None:
        """Add entities from a single chunk (co-occurrence only, no relation type)."""
        for text, label in entities:
            if not self.graph.has_node(text):
                self.graph.add_node(text, type=label)
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                node_a = entities[i][0]
                node_b = entities[j][0]
                if self.graph.has_edge(node_a, node_b):
                    self.graph[node_a][node_b]["weight"] += 1
                else:
                    self.graph.add_edge(node_a, node_b, weight=1,
                                        relation_type="related_to")

    def add_typed_entities_from_chunk(
        self,
        entities: list[tuple[str, str]],
        triples: list[tuple[str, str, str]],
    ) -> None:
        """
        Add entities and typed relationships from a single chunk.

        :param entities: list of (entity_text, entity_label) tuples
        :param triples:  list of (entity_a, entity_b, relation_type) from RelationshipClassifier
        """
        # Add / update nodes
        for text, label in entities:
            if not self.graph.has_node(text):
                self.graph.add_node(text, type=label)

        # Add / update edges with relationship type
        for node_a, node_b, rel_type in triples:
            if self.graph.has_edge(node_a, node_b):
                self.graph[node_a][node_b]["weight"] += 1
                # Keep the most specific relation if a better one is found
                current = self.graph[node_a][node_b].get("relation_type", "related_to")
                if current == "related_to" and rel_type != "related_to":
                    self.graph[node_a][node_b]["relation_type"] = rel_type
            else:
                self.graph.add_edge(node_a, node_b, weight=1, relation_type=rel_type)

    def build_from_chunks(
        self, chunks_entities: list[list[tuple[str, str]]]
    ) -> None:
        """Build graph using simple co-occurrence (no relation types)."""
        for entities in chunks_entities:
            self.add_entities_from_chunk(entities)

    def build_from_chunks_with_relations(
        self,
        chunks_with_context: list[tuple[list[tuple[str, str]], str]],
    ) -> None:
        """
        Build graph with typed relationships.

        :param chunks_with_context: output of EntityExtractor.extract_entities_with_context()
                                    i.e. list of (entities, chunk_text) tuples
        """
        from src.graph.relationship_classifier import classify_pairs
        for entities, context_text in chunks_with_context:
            triples = classify_pairs(entities, context_text)
            self.add_typed_entities_from_chunk(entities, triples)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filename: str = "knowledge_graph.graphml") -> Path:
        """Save the graph to a GraphML file and return the path."""
        self.graph_dir.mkdir(parents=True, exist_ok=True)
        path = self.graph_dir / filename
        nx.write_graphml(self.graph, str(path))
        return path

    def load(self, filename: str = "knowledge_graph.graphml") -> None:
        """Load the graph from a GraphML file."""
        path = self.graph_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")
        self.graph = nx.read_graphml(str(path))

    # ------------------------------------------------------------------
    # Querying helpers
    # ------------------------------------------------------------------

    def get_neighbors(self, entity: str) -> list[str]:
        """Return all direct neighbours of an entity node."""
        if entity not in self.graph:
            return []
        return list(self.graph.neighbors(entity))

    def get_neighbors_with_relations(self, entity: str) -> list[tuple[str, str]]:
        """
        Return neighbours with their relationship type.

        Returns list of (neighbour_name, relation_type) tuples.
        """
        if entity not in self.graph:
            return []
        results = []
        for neighbour in self.graph.neighbors(entity):
            rel = self.graph[entity][neighbour].get("relation_type", "related_to")
            results.append((neighbour, rel))
        return results

    def get_relation_type_summary(self) -> dict[str, int]:
        """Return a count of how many edges exist per relationship type."""
        counts: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relation_type", "related_to")
            counts[rel] = counts.get(rel, 0) + 1
        return counts

    def get_top_entities(self, n: int = 10) -> list[tuple[str, int]]:
        """Return the top-n most connected entities (by degree)."""
        degrees = sorted(
            self.graph.degree(), key=lambda x: x[1], reverse=True
        )
        return degrees[:n]

    def get_stats(self) -> dict:
        """Return basic graph statistics."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": round(nx.density(self.graph), 4),
        }


if __name__ == "__main__":
    kg = KnowledgeGraph()
    sample_entities = [
        [("NumPy", "ORG"), ("Python", "PRODUCT"), ("Travis Oliphant", "PERSON")],
        [("NumPy", "ORG"), ("SciPy", "ORG")],
    ]
    kg.build_from_chunks(sample_entities)
    print(f"Stats: {kg.get_stats()}")
    print(f"Top entities: {kg.get_top_entities(5)}")
