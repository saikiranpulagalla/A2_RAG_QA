"""Sentence-level context compression for safer, cheaper RAG prompts.

This module implements a deterministic variant of the CRAG-style
"focus/filter" idea: retrieved chunks are decomposed into evidence sentences,
scored against the query, and recomposed into a compact context. It avoids an
extra LLM call, preserves source/evidence ids, and strips obvious instruction
injection lines before the generator sees them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Sequence

from utils import Document, lexical_relevance, normalize_documents, safe_source_label, strip_obvious_injection_lines, tokenize, truncate_to_char_limit


@dataclass(frozen=True)
class EvidenceSentence:
    """A sentence selected for the prompt context."""

    evidence_id: int
    source_index: int
    sentence_index: int
    text: str
    score: float
    source: str | None = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def split_evidence_sentences(text: str) -> List[str]:
    """Split text into reasonably stable sentence-like units.

    We intentionally keep this simple and deterministic; QA datasets often use
    short factual passages where heavyweight NLP sentence splitting is not worth
    another dependency.
    """
    cleaned = strip_obvious_injection_lines(text or "")
    if not cleaned:
        return []
    # First split paragraphs/lines, then sentence punctuation. Keep semicolon
    # clauses together because they often carry needed answer qualifiers.
    raw_parts: List[str] = []
    for block in re.split(r"\n+", cleaned):
        raw_parts.extend(re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", block.strip()))
    parts = [p.strip() for p in raw_parts if len(p.strip()) >= 3]
    return parts or [cleaned.strip()]


def _source_label(metadata: Dict[str, Any], fallback: int) -> str | None:
    return safe_source_label(metadata, fallback)


def _coverage_bonus(query: str, sentence: str) -> float:
    """Reward sentences that cover rare-looking query entities/numbers."""
    q_tokens = set(tokenize(query))
    s_tokens = set(tokenize(sentence))
    if not q_tokens or not s_tokens:
        return 0.0
    coverage = len(q_tokens & s_tokens) / len(q_tokens)
    number_bonus = 0.10 if re.search(r"\b\d{2,4}\b", query) and re.search(r"\b\d{2,4}\b", sentence) else 0.0
    capital_terms = re.findall(r"\b[A-Z][a-zA-Z0-9]+\b", query or "")
    entity_bonus = 0.0
    if capital_terms:
        hits = sum(1 for term in capital_terms if term.lower() in sentence.lower())
        entity_bonus = min(0.15, 0.05 * hits)
    return min(0.25, 0.15 * coverage + number_bonus + entity_bonus)


def select_evidence_sentences(
    query: str,
    docs: Sequence[Any],
    *,
    max_sentences: int = 10,
    per_doc_limit: int = 3,
    min_score: float = 0.01,
) -> List[EvidenceSentence]:
    """Select the most query-relevant evidence sentences from retrieved docs."""
    candidates: List[EvidenceSentence] = []
    evidence_id = 1
    for source_index, doc in enumerate(normalize_documents(docs), start=1):
        source = _source_label(doc.metadata, source_index)
        scored_for_doc: List[EvidenceSentence] = []
        for sentence_index, sentence in enumerate(split_evidence_sentences(doc.content), start=1):
            score = lexical_relevance(query, sentence) + _coverage_bonus(query, sentence)
            if score >= min_score:
                scored_for_doc.append(
                    EvidenceSentence(
                        evidence_id=evidence_id,
                        source_index=source_index,
                        sentence_index=sentence_index,
                        text=sentence,
                        score=round(float(score), 6),
                        source=source,
                    )
                )
                evidence_id += 1
        scored_for_doc.sort(key=lambda s: (-s.score, s.sentence_index))
        candidates.extend(scored_for_doc[:per_doc_limit])

    candidates.sort(key=lambda s: (-s.score, s.source_index, s.sentence_index))
    selected = candidates[:max_sentences]
    # Re-order selected evidence by source/sentence order for readability after
    # ranking has chosen the relevant subset.
    selected.sort(key=lambda s: (s.source_index, s.sentence_index))
    return selected


def compressed_evidence_documents(query: str, docs: Sequence[Any], *, max_sentences: int = 10) -> List[Document]:
    """Return selected sentences as Document objects with evidence metadata."""
    selected = select_evidence_sentences(query, docs, max_sentences=max_sentences)
    return [
        Document(
            content=item.text,
            metadata={
                "evidence_id": item.evidence_id,
                "source_index": item.source_index,
                "sentence_index": item.sentence_index,
                "source": item.source,
                "compression_score": item.score,
            },
        )
        for item in selected
    ]


def format_compressed_context(query: str, docs: Sequence[Any], *, max_chars: int = 4500, max_sentences: int = 10) -> tuple[str, Dict[str, object]]:
    """Return a compact context string and trace metadata."""
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError("max_chars must be an integer")
    if max_chars <= 0:
        return "", {
            "enabled": True,
            "selected_sentences": 0,
            "compression_ratio": 0.0,
            "evidence": [],
        }
    selected = select_evidence_sentences(query, docs, max_sentences=max_sentences)
    if not selected:
        return "[No relevant context found]", {
            "enabled": True,
            "selected_sentences": 0,
            "compression_ratio": 0.0,
            "evidence": [],
        }

    original_chars = sum(len(d.content) for d in normalize_documents(docs)) or 1
    blocks: List[str] = []
    for item in selected:
        header = f"[Evidence {item.evidence_id} | source={item.source} | score={item.score:.3f}]"
        block = f"{header}\n{item.text}"
        separator_length = 2 if blocks else 0
        used = len("\n\n".join(blocks))
        remaining = max_chars - used - separator_length
        if remaining <= 0:
            break
        block = truncate_to_char_limit(block, remaining)
        blocks.append(block)

    context = "\n\n".join(blocks) if blocks else truncate_to_char_limit("[No relevant context found]", max_chars)
    metadata = {
        "enabled": True,
        "selected_sentences": len(blocks),
        "source_documents": len(normalize_documents(docs)),
        "original_chars": original_chars,
        "compressed_chars": len(context),
        "compression_ratio": round(len(context) / max(1, original_chars), 4),
        "evidence": [item.to_dict() for item in selected[: len(blocks)]],
    }
    return context, metadata
