import re

from langchain_core.documents import Document

from src.utils import setup_logger

logger = setup_logger(__name__)


class TextCleaner:
    def remove_extra_whitespace(self, text: str) -> str:
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\n+', '\n\n', text)
        text = text.strip()
        return text

    def remove_special_characters(self, text: str, keep_chars: str = "") -> str:
        pattern = f'[^a-zA-Z0-9\\s{re.escape(keep_chars)}]'
        cleaned = re.sub(pattern, '', text)
        return cleaned

    def clean_document(self, document: Document) -> Document:
        cleaned_content = self.remove_extra_whitespace(document.page_content)
        cleaned_content = self.remove_special_characters(cleaned_content)

        cleaned_doc = Document(
            page_content=cleaned_content,
            metadata=document.metadata
        )

        return cleaned_doc

    def clean_documents(self, documents: list[Document]) -> list[Document]:
        logger.info(f"Cleaning {len(documents)} documents...")

        cleaned = [self.clean_document(doc) for doc in documents]

        logger.info(f"Cleaned {len(cleaned)} documents")
        return cleaned


if __name__ == "__main__":
    # Test the cleaner
    cleaner = TextCleaner()
    test_text = "This   is    a  test   text  with   extra    spaces."
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaner.remove_extra_whitespace(test_text)}")
