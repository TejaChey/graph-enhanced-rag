from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

from config import settings
from src.utils import setup_logger

logger = setup_logger(__name__)


class DocumentLoader:
    def __init__(self, data_dir: Path = settings.RAW_DATA_DIR):
        self.data_dir: Path = data_dir
        logger.info(f"DocumentLoader initialized with directory: {self.data_dir}")

    def load_markdown_files(self) -> list[Document]:
        logger.info("Loading markdown files...")

        loader = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader
        )

        documents = loader.load()
        return documents

    def load_text_files(self) -> list[Document]:
        logger.info("Loading text files...")

        loader = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.txt",
            loader_cls=TextLoader
        )

        documents = loader.load()
        return documents

    def load_all_documents(self) -> list[Document]:
        logger.info("Loading all documents...")

        all_docs: list[Document] = []

        md_docs = self.load_markdown_files()
        all_docs.extend(md_docs)

        txt_docs = self.load_text_files()
        all_docs.extend(txt_docs)

        logger.info(f"Total documents loaded: {len(all_docs)}")

        return all_docs

if __name__ == "__main__":
    # Test the loader
    loader = DocumentLoader()
    print(f"Data directory: {loader.data_dir}")
    print(f"Files in directory: {list(loader.data_dir.glob('**/*'))[:5]}")
