import json
from pathlib import Path
from typing import Any, Dict, List


def count_files_in_directory(directory: Path, extension: str = "*") -> int:
    if not directory.exists():
        return 0

    if extension == "*":
        return len(list(directory.rglob("*.*")))
    else:
        return len(list(directory.rglob(f"*.{extension.lstrip('*.')}")))


def load_json(file_path: Path) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_docs(docs: List[Any]) -> str:
    return "\n\n".join([doc.page_content for doc in docs])


def truncate_text(text: str, max_length: int = 100) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def estimate_tokens(text: str) -> int:
    return len(text) // 4
