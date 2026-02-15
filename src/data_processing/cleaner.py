from typing import List

from langchain.schema import Document

from src.utils import setup_logger

logger = setup_logger(__name__)


class TextCleaner:
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        # TODO: Remove multiple spaces, tabs, newlines
        return text

    @staticmethod
    def remove_special_characters(text: str, keep_chars: str = "") -> str:
        # TODO: Remove unwanted special characters
        return text

    @staticmethod
    def clean_document(document: Document) -> Document:
        # TODO: Implement document cleaning
        return document

    @staticmethod
    def clean_documents(documents: List[Document]) -> List[Document]:
        logger.info(f"Cleaning {len(documents)} documents...")

        # TODO: Implement batch cleaning
        cleaned = documents  # Replace with actual cleaning

        logger.info(f"Cleaned {len(cleaned)} documents")
        return cleaned


if __name__ == "__main__":
    # Test the cleaner
    cleaner = TextCleaner()
    test_text = "This   is    a  test   text  with   extra    spaces."
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaner.remove_extra_whitespace(test_text)}")
