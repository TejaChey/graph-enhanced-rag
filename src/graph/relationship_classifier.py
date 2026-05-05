import re

# Ordered from most specific to most generic so we match the best one first
_RELATIONSHIP_PATTERNS: list[tuple[str, list[str]]] = [
    ("extends", [
        r"\bextend[s]?\b", r"\bsubclass(es)?\b", r"\binherit[s]?\b",
    ]),
    ("implements", [
        r"\bimplement[s]?\b", r"\brealize[s]?\b", r"\bconforms?\s+to\b",
    ]),
    ("depends_on", [
        r"\bdepend[s]?\s+on\b", r"\brequire[s]?\b", r"\bneeds?\b",
        r"\bbuilt\s+on\b", r"\bbased\s+on\b",
    ]),
    ("part_of", [
        r"\bpart\s+of\b", r"\bcomponent\s+of\b", r"\bbelongs?\s+to\b",
        r"\bincluded\s+in\b", r"\bwithin\b",
    ]),
    ("uses", [
        r"\buses?\b", r"\bapplies?\b", r"\bcalls?\b", r"\binvokes?\b",
        r"\bleverages?\b", r"\bwraps?\b", r"\bprovides?\b", r"\bsupport[s]?\b",
    ]),
]

_DEFAULT_RELATIONSHIP = "related_to"


def classify_relationship(context: str) -> str:
    text = context.lower()
    for rel_type, patterns in _RELATIONSHIP_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text):
                return rel_type
    return _DEFAULT_RELATIONSHIP


def classify_pairs(
    entities: list[tuple[str, str]],
    context: str,
) -> list[tuple[str, str, str]]:
    triples = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            entity_a = entities[i][0]
            entity_b = entities[j][0]
            rel = classify_relationship(context)
            triples.append((entity_a, entity_b, rel))
    return triples


if __name__ == "__main__":
    sample = "NumPy uses Python and depends on C extensions for performance."
    ents = [("NumPy", "ORG"), ("Python", "PRODUCT"), ("C", "PRODUCT")]
    triples = classify_pairs(ents, sample)
    for a, b, rel in triples:
        print(f"  {a!r} --[{rel}]--> {b!r}")
