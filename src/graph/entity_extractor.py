import spacy

from config import settings


class EntityExtractor:

    def __init__(self, model_name=None, entity_types=None):
        self.model_name = model_name or settings.SPACY_MODEL
        self.entity_types = entity_types or settings.ENTITY_TYPES
        self._nlp = None

    def _load_model(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError:
                raise OSError(
                    f"SpaCy model '{self.model_name}' not found.\n"
                    f"Run: python -m spacy download {self.model_name}"
                )
        return self._nlp

    def extract_entities(self, text: str) -> list[tuple[str, str]]:
        nlp = self._load_model()
        doc = nlp(text)
        seen = set()
        entities = []
        for ent in doc.ents:
            if ent.label_ in self.entity_types:
                key = (ent.text.strip(), ent.label_)
                if key not in seen:
                    seen.add(key)
                    entities.append(key)
        return entities

    def extract_entities_from_chunks(self, chunks) -> list[list[tuple[str, str]]]:
        return [self.extract_entities(chunk.page_content) for chunk in chunks]

    def extract_entities_with_context(
        self, chunks
    ) -> list[tuple[list[tuple[str, str]], str]]:
        return [
            (self.extract_entities(chunk.page_content), chunk.page_content)
            for chunk in chunks
        ]


if __name__ == "__main__":
    extractor = EntityExtractor()
    sample = "NumPy is a Python library developed by Travis Oliphant at Enthought."
    print(f"Text: {sample}")
    print(f"Entities: {extractor.extract_entities(sample)}")
