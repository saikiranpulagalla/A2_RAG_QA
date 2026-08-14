"""Context diversity helpers for retrieval results.

A RAG system can retrieve many near-duplicate chunks. That wastes context budget
and makes the generator overfit to one passage. These helpers implement a small
MMR-style selector using lexical relevance and token overlap, avoiding heavy
cross-encoder dependencies.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, List, Sequence, Set

from utils import Document, lexical_relevance, normalize_document, normalize_documents, tokenize


def _token_set(text: Any) -> Set[str]:
    return set(tokenize(normalize_document(text).content))


def jaccard_similarity(a: Any, b: Any) -> float:
    a_tokens = _token_set(a)
    b_tokens = _token_set(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def duplicate_context_rate(docs: Sequence[Any], threshold: float = 0.85) -> float:
    """Approximate fraction of near-duplicate document pairs."""
    docs = list(docs or [])
    if len(docs) < 2:
        return 0.0
    pairs = list(combinations(range(len(docs)), 2))
    near_duplicates = sum(1 for i, j in pairs if jaccard_similarity(docs[i], docs[j]) >= threshold)
    return near_duplicates / len(pairs)


def select_diverse_contexts(query: str, docs: Sequence[Any], k: int, lambda_mult: float = 0.75) -> List[Document]:
    """Greedy MMR-like selection over already-ranked documents.

    ``lambda_mult`` close to 1 prioritizes relevance; lower values prioritize
    diversity. The incoming ranking is respected through deterministic tie-breaks.
    """
    candidates = normalize_documents(docs)
    if not candidates or k <= 0:
        return []

    selected: List[Document] = []
    candidate_indices = list(range(len(candidates)))
    relevance = [lexical_relevance(query, doc) for doc in candidates]

    while candidate_indices and len(selected) < k:
        best_idx = None
        best_score = None
        for idx in candidate_indices:
            diversity_penalty = max((jaccard_similarity(candidates[idx], chosen) for chosen in selected), default=0.0)
            # Keep a tiny dense/rank prior so pure semantic dense hits with low lexical
            # overlap are not completely discarded.
            rank_prior = 1.0 / (idx + 1)
            score = lambda_mult * relevance[idx] + 0.10 * rank_prior - (1.0 - lambda_mult) * diversity_penalty
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(candidates[best_idx])
        candidate_indices.remove(best_idx)
    return selected
