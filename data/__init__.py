"""Dataset loaders for A2-RAG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import normalize_documents

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS_PATH = ROOT / "data" / "documents" / "wiki_docs.json"
DEFAULT_QUESTIONS_PATH = ROOT / "data" / "questions" / "nq_1000.json"


def _validate_limit(limit: Optional[int]) -> Optional[int]:
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer or None")
    if limit < 0:
        raise ValueError("limit must be greater than or equal to zero")
    return limit


def _limited(items: List[Any], limit: Optional[int]) -> List[Any]:
    validated = _validate_limit(limit)
    return items if validated is None else items[:validated]


def _read_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_documents(path: str | Path = DEFAULT_DOCUMENTS_PATH, limit: Optional[int] = None) -> List[str]:
    """Load documents as clean strings regardless of JSON record shape."""
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of documents in {path}")
    docs = [doc.content for doc in normalize_documents(raw)]
    return _limited(docs, limit)


def load_document_objects(path: str | Path = DEFAULT_DOCUMENTS_PATH, limit: Optional[int] = None):
    """Load documents as project ``Document`` objects with metadata."""
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of documents in {path}")
    docs = normalize_documents(raw)
    for index, doc in enumerate(docs):
        doc.metadata.setdefault("doc_id", f"corpus-doc-{index:04d}")
        doc.metadata.setdefault("corpus_record_index", index)
    return _limited(docs, limit)


def load_questions(path: str | Path = DEFAULT_QUESTIONS_PATH, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load Natural Questions-style examples."""
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of questions in {path}")
    if not all(isinstance(item, dict) for item in raw):
        raise ValueError(f"Expected every question in {path} to be an object")
    return _limited(raw, limit)


def get_reference_answers(example: Dict[str, Any]) -> List[str]:
    """Return all distinct usable reference answers from common QA schemas."""
    candidates: List[Any] = []
    if "answer" in example:
        answer = example["answer"]
        candidates.extend(answer if isinstance(answer, list) else [answer])

    answers = example.get("answers")
    if isinstance(answers, dict):
        texts = answers.get("text")
        candidates.extend(texts if isinstance(texts, list) else [texts] if texts is not None else [])
    elif isinstance(answers, list):
        for answer in answers:
            if isinstance(answer, dict):
                value = answer.get("text") or answer.get("answer")
                candidates.extend(value if isinstance(value, list) else [value] if value is not None else [])
            else:
                candidates.append(answer)

    distinct: List[str] = []
    for candidate in candidates:
        text = "" if candidate is None else str(candidate).strip()
        if text and text not in distinct:
            distinct.append(text)
    return distinct


def get_reference_answer(example: Dict[str, Any]) -> str:
    """Return the first usable answer from common QA schemas."""
    answers = get_reference_answers(example)
    return answers[0] if answers else ""
