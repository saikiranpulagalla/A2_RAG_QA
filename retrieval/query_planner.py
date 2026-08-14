"""Lightweight adaptive query planning for RAG.

This module keeps the project dependency-light while adding a senior-level
retrieval planner: classify query shape, choose a retrieval budget, and produce
safe lexical query variants for fusion. It does not call an LLM, so it is cheap
and deterministic in tests.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List

from config import CHILD_K, PARENT_K
from utils import tokenize


_MULTI_HOP_MARKERS = {
    "compare", "contrast", "difference", "relationship", "between", "cause", "caused",
    "effect", "impact", "influence", "why", "how", "timeline", "before", "after",
}
_GLOBAL_MARKERS = {
    "main themes", "overview", "summarize the dataset", "across the corpus", "overall",
    "patterns", "trend", "trends", "common", "recurring",
}
_SOURCE_BOUND_MARKERS = {
    "according to", "from the document", "in the document", "from the report", "in this paper",
    "based on the context", "provided context", "source says",
}
_TEMPORAL_MARKERS = {
    "latest", "current", "today", "yesterday", "tomorrow", "recent", "new", "updated",
    "price", "schedule", "deadline", "release",
}


@dataclass(frozen=True)
class QueryPlan:
    """Decision-support metadata for retrieval."""

    query: str
    complexity: str
    parent_k: int
    child_k: int
    variants: List[str] = field(default_factory=list)
    is_multi_hop: bool = False
    is_global: bool = False
    is_source_bound: bool = False
    is_temporal_or_dynamic: bool = False
    rationale: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _contains_phrase(query_l: str, phrases) -> bool:
    return any(phrase in query_l for phrase in phrases)


def _entity_like_terms(query: str) -> List[str]:
    """Extract cheap entity-like spans without NLP dependencies."""
    spans = re.findall(r"\b(?:[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,4}|\d{3,4})\b", query or "")
    cleaned: List[str] = []
    for span in spans:
        if span.lower() in {"what", "when", "where", "which", "who", "why", "how"}:
            continue
        if span not in cleaned:
            cleaned.append(span)
    return cleaned[:5]


def build_query_variants(query: str, max_variants: int = 4) -> List[str]:
    """Generate deterministic query variants for reciprocal-rank fusion.

    The variants are deliberately conservative: no invented facts, no LLM call,
    just entity-focused and keyword-focused rewrites that reduce semantic drift.
    """
    query = (query or "").strip()
    if not query:
        return []

    variants: List[str] = [query]
    terms = tokenize(query)
    if terms:
        keyword_query = " ".join(terms[:12])
        if keyword_query and keyword_query not in variants:
            variants.append(keyword_query)

    entities = _entity_like_terms(query)
    if entities:
        entity_query = " ".join(entities)
        if entity_query and entity_query not in variants:
            variants.append(entity_query)

    # For multi-hop questions, keep relation words with entities/keywords.
    relation_terms = [t for t in terms if t in _MULTI_HOP_MARKERS]
    if relation_terms and entities:
        relation_query = " ".join(entities + relation_terms[:4])
        if relation_query and relation_query not in variants:
            variants.append(relation_query)

    return variants[:max_variants]


def analyze_query(query: str) -> QueryPlan:
    """Create an adaptive retrieval plan for a query."""
    q = (query or "").strip()
    q_l = q.lower()
    tokens = tokenize(q_l)

    is_global = _contains_phrase(q_l, _GLOBAL_MARKERS)
    is_source_bound = _contains_phrase(q_l, _SOURCE_BOUND_MARKERS)
    is_temporal = any(term in tokens or term in q_l for term in _TEMPORAL_MARKERS)
    is_multi_hop = any(term in tokens for term in _MULTI_HOP_MARKERS) or len(re.findall(r"\?", q)) > 1
    has_numbers = bool(re.search(r"\b\d{3,4}\b", q))
    long_query = len(tokens) > 18

    score = 0
    score += 2 if is_global else 0
    score += 2 if is_multi_hop else 0
    score += 1 if is_source_bound else 0
    score += 1 if is_temporal else 0
    score += 1 if has_numbers else 0
    score += 1 if long_query else 0

    if score >= 4:
        complexity = "high"
        parent_k = max(PARENT_K, 6)
        child_k = max(CHILD_K, 6)
    elif score >= 2:
        complexity = "medium"
        parent_k = max(PARENT_K, 5)
        child_k = max(CHILD_K, 5)
    else:
        complexity = "low"
        parent_k = PARENT_K
        child_k = CHILD_K

    variants = build_query_variants(q, max_variants=4 if complexity != "low" else 3)
    rationale_bits = []
    if is_global:
        rationale_bits.append("global/corpus-level")
    if is_multi_hop:
        rationale_bits.append("multi-hop")
    if is_source_bound:
        rationale_bits.append("source-bound")
    if is_temporal:
        rationale_bits.append("temporal/dynamic")
    if not rationale_bits:
        rationale_bits.append("simple factual")

    return QueryPlan(
        query=q,
        complexity=complexity,
        parent_k=parent_k,
        child_k=child_k,
        variants=variants,
        is_multi_hop=is_multi_hop,
        is_global=is_global,
        is_source_bound=is_source_bound,
        is_temporal_or_dynamic=is_temporal,
        rationale=", ".join(rationale_bits),
    )
