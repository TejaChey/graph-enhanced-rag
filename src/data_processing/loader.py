from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from config import settings


class DocumentLoader:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or settings.RAW_DATA_DIR

    def load_markdown_files(self):
        loader = DirectoryLoader(
            path=str(self.data_dir),
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
        )
        return loader.load()

    def load_text_files(self):
        loader = DirectoryLoader(
            path=str(self.data_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "latin-1"},
        )
        return loader.load()

    def load_pdf_files(self):
        loader = DirectoryLoader(
            path=str(self.data_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
        )
        return loader.load()

    def load_pdf_files(self):
        loader = DirectoryLoader(
            path=str(self.data_dir),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
        )
        return loader.load()

    def load_all_documents(self):
        all_docs = []
        all_docs.extend(self.load_markdown_files())
        all_docs.extend(self.load_text_files())
        all_docs.extend(self.load_pdf_files())
        return all_docs


if __name__ == "__main__":
    loader = DocumentLoader()
    print(f"Data directory: {loader.data_dir}")
    print(f"Files in directory: {list(loader.data_dir.glob('**/*'))[:5]}")
