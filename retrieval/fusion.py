"""Rank-fusion utilities for multi-query retrieval."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, List, Sequence, Tuple

from utils import Document, normalize_document


def _document_key(item: object) -> Tuple[str, str]:
    doc = normalize_document(item)
    source = doc.metadata.get("source") or doc.metadata.get("title") or doc.metadata.get("url") or doc.metadata.get("index")
    return doc.content.strip(), str(source or "")


def dedupe_preserve_order(items: Iterable[object]) -> List[Document]:
    seen = OrderedDict()
    for item in items:
        doc = normalize_document(item)
        key = _document_key(doc)
        if key[0] and key not in seen:
            seen[key] = doc
    return list(seen.values())


def reciprocal_rank_fusion(rankings: Sequence[Sequence[object]], k: int = 60, limit: int | None = None) -> List[Document]:
    """Fuse ranked lists using Reciprocal Rank Fusion.

    RRF is simple, deterministic, and robust when each query variant retrieves a
    slightly different view of the corpus.
    """
    scores = {}
    first_seen = {}
    documents = {}
    counter = 0
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            doc = normalize_document(item)
            key = _document_key(doc)
            if not key[0]:
                continue
            if key not in first_seen:
                first_seen[key] = counter
                documents[key] = doc
                counter += 1
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    ordered = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
    selected = ordered[:limit] if limit else ordered
    return [documents[key] for key in selected]
