import spacy
from langchain_core.documents import Document

from config import settings
from src.graph.knowledge_graph import KnowledgeGraph


class GraphRetriever:

    def __init__(self, graph_dir=None, model_name=None):
        self.kg = KnowledgeGraph(graph_dir=graph_dir)
        self.model_name = model_name or settings.SPACY_MODEL
        self._loaded = False
        self._nlp = None

    def _load_graph(self):
        if not self._loaded:
            self.kg.load()
            self._loaded = True

    def _load_nlp(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError:
                raise OSError(
                    f"SpaCy model '{self.model_name}' not found.\n"
                    f"Run: python -m spacy download {self.model_name}"
                )
        return self._nlp

    def _extract_query_entities(self, query: str) -> list[str]:
        nlp = self._load_nlp()
        doc = nlp(query)
        entities = [ent.text.strip() for ent in doc.ents]

        # Fallback: use filtered words (skip stopwords and punctuation)
        if not entities:
            entities = [
                token.text for token in doc
                if not token.is_stop and not token.is_punct and len(token.text) > 2
            ]
        return entities

    def _find_graph_node(self, mention: str) -> str | None:
        return next(
            (n for n in self.kg.graph.nodes if n.lower() == mention.lower()),
            None,
        )

    def retrieve(self, query: str, top_k: int = 4) -> list[Document]:
        self._load_graph()
        query_entities = self._extract_query_entities(query)
        results = []
        seen_nodes = set()

        for mention in query_entities:
            node = self._find_graph_node(mention)
            if node and node not in seen_nodes:
                seen_nodes.add(node)
                neighbours = self.kg.get_neighbors_with_relations(node)
                if neighbours:
                    lines = [
                        f"  '{node}' --[{rel}]--> {nbr}"
                        for nbr, rel in neighbours
                    ]
                    content = f"Graph relationships for '{node}':\n" + "\n".join(lines)
                    results.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": "knowledge_graph",
                                "entity": node,
                                "num_relations": len(neighbours),
                            },
                        )
                    )
            if len(results) >= top_k:
                break

        return results


if __name__ == "__main__":
    print("GraphRetriever - Phase 2 implemented (with SpaCy NER + typed relations)")
