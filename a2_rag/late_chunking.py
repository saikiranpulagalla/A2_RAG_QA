"""Context assembly helpers for retrieved chunks."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from config import ENABLE_CONTEXT_COMPRESSION, MAX_CONTEXT_CHARS, MAX_CONTEXT_SENTENCES
from retrieval.context_compressor import format_compressed_context
from utils import format_evidence_block, handle_empty_retrieval, setup_logger

logger = setup_logger(__name__)


def extract_context_with_trace(
    query: str,
    docs: Sequence[Any],
    separator: str = "\n\n",
    max_length: int | None = MAX_CONTEXT_CHARS,
) -> Tuple[str, Dict[str, object]]:
    """Return sanitized context plus trace metadata.

    V4 defaults to sentence-level context compression. This reduces irrelevant
    instructions/noise in the prompt while preserving numbered evidence blocks.
    """
    if isinstance(max_length, bool) or (max_length is not None and not isinstance(max_length, int)):
        raise TypeError("max_length must be an integer or None")
    budget = MAX_CONTEXT_CHARS if max_length is None else max_length
    if budget <= 0:
        return "", {"enabled": bool(ENABLE_CONTEXT_COMPRESSION), "selected_sentences": 0}
    if not docs:
        return handle_empty_retrieval([]), {"enabled": bool(ENABLE_CONTEXT_COMPRESSION), "selected_sentences": 0}
    if ENABLE_CONTEXT_COMPRESSION:
        return format_compressed_context(
            query,
            docs,
            max_chars=budget,
            max_sentences=MAX_CONTEXT_SENTENCES,
        )
    context = format_evidence_block(docs, max_chars=budget)
    return context, {"enabled": False, "selected_sentences": 0}


def extract_context(docs: Sequence[Any], separator: str = "\n\n", max_length: int | None = MAX_CONTEXT_CHARS) -> str:
    """Backward-compatible context extraction without query-aware compression."""
    if isinstance(max_length, bool) or (max_length is not None and not isinstance(max_length, int)):
        raise TypeError("max_length must be an integer or None")
    budget = MAX_CONTEXT_CHARS if max_length is None else max_length
    if budget <= 0:
        return ""
    if not docs:
        return handle_empty_retrieval([])
    return format_evidence_block(docs, max_chars=budget)


def prepare_context_for_qa(docs: Sequence[Any], include_source_info: bool = False) -> str:
    return extract_context(docs)
