import json
from pathlib import Path

from langchain_core.documents import Document


def count_files_in_directory(directory, extension="*"):
    if not directory.exists():
        return 0
    if extension == "*":
        return len(list(directory.rglob("*.*")))
    return len(list(directory.rglob(f"*.{extension.lstrip('*.')}")))


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def truncate_text(text, max_length=100):
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def estimate_tokens(text):
    return len(text) // 4
