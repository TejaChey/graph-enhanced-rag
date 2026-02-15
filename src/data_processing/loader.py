from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from config import settings
from src.utils import setup_logger

logger = setup_logger(__name__)


class DocumentLoader:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir or settings.RAW_DATA_DIR
        logger.info(f"DocumentLoader initialized with directory: {self.data_dir}")

    def load_markdown_files(self) -> List[Document]:
        logger.info("Loading markdown files...")

        # TODO: Implement this method
        # Example structure:
        # loader = DirectoryLoader(
        #     str(self.data_dir),
        #     glob="**/*.md",
        #     loader_cls=UnstructuredMarkdownLoader
        # )
        # documents = loader.load()

        raise NotImplementedError("TODO: Implement markdown file loading")

    def load_text_files(self) -> List[Document]:
        logger.info("Loading text files...")
        raise NotImplementedError("TODO: Implement text file loading")

    def load_all_documents(self) -> List[Document]:
        logger.info("Loading all documents...")
        raise NotImplementedError("TODO: Implement combined document loading")


if __name__ == "__main__":
    # Test the loader
    loader = DocumentLoader()
    print(f"Data directory: {loader.data_dir}")
    print(f"Files in directory: {list(loader.data_dir.glob('**/*'))[:5]}")
