"""Deterministic boundary-aware text chunking with bounded overlap."""

from __future__ import annotations

from typing import Sequence


DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")


def split_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    separators: Sequence[str] = DEFAULT_SEPARATORS,
) -> list[str]:
    """Split text into bounded chunks, preferring natural boundaries.

    The next window begins ``chunk_overlap`` characters before the previous
    boundary. Long unbroken spans fall back to a hard character boundary.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if (
        isinstance(chunk_overlap, bool)
        or not isinstance(chunk_overlap, int)
        or chunk_overlap < 0
        or chunk_overlap >= chunk_size
    ):
        raise ValueError("chunk_overlap must be an integer from 0 to chunk_size - 1")
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        hard_end = min(start + chunk_size, text_length)
        end = hard_end
        if hard_end < text_length:
            minimum_boundary = start + max(1, chunk_size // 2)
            for separator in separators:
                if not separator:
                    continue
                boundary = text.rfind(separator, minimum_boundary, hard_end)
                if boundary >= minimum_boundary:
                    end = boundary + len(separator)
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break

        next_start = end - chunk_overlap
        start = next_start if next_start > start else start + 1

    return chunks
