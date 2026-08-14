"""Dependency-light sparse/hybrid retrieval helpers.

Dense embeddings are good at semantic matching, but they can miss exact names,
numbers, acronyms, and rare terms. This module adds a small BM25-style sparse
retriever that can be fused with dense results using Reciprocal Rank Fusion.
It deliberately avoids heavyweight search services so the project remains
portable for an internship/demo submission.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from utils import Document, normalize_documents, tokenize


def _term_frequencies(texts: Sequence[str]) -> Tuple[List[Counter], Dict[str, int], float]:
    term_freqs: List[Counter] = []
    doc_freq: Dict[str, int] = defaultdict(int)
    total_len = 0
    for text in texts:
        tokens = tokenize(text)
        counts = Counter(tokens)
        term_freqs.append(counts)
        total_len += len(tokens)
        for term in counts:
            doc_freq[term] += 1
    avg_len = total_len / max(1, len(texts))
    return term_freqs, dict(doc_freq), avg_len


@dataclass
class SparseRetriever:
    """Precomputed BM25-style index for repeated sparse retrieval.

    V3 recomputed BM25 statistics for every query. V4 keeps the same behavior
    but precomputes term frequencies/doc frequencies once per corpus, which is
    important when dense and sparse retrieval are fused across many query
    variants or evaluation examples.
    """

    docs: Sequence[Any]
    k1: float = 1.5
    b: float = 0.75
    texts: List[str] = field(init=False)
    documents: List[Document] = field(init=False)
    term_freqs: List[Counter] = field(init=False)
    doc_freq: Dict[str, int] = field(init=False)
    avg_len: float = field(init=False)

    def __post_init__(self) -> None:
        self.documents = normalize_documents(self.docs)
        self.texts = [doc.content for doc in self.documents]
        self.term_freqs, self.doc_freq, self.avg_len = _term_frequencies(self.texts)

    def scores(self, query: str) -> List[float]:
        query_terms = tokenize(query)
        if not self.texts or not query_terms:
            return [0.0 for _ in self.texts]
        n_docs = len(self.texts)
        scores: List[float] = []
        for counts in self.term_freqs:
            doc_len = sum(counts.values()) or 1
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if tf <= 0:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_len, 1e-9))
                score += idf * ((tf * (self.k1 + 1)) / max(denom, 1e-9))
            scores.append(float(score))
        return scores

    def search(self, query: str, k: int = 5, *, min_score: float = 0.0) -> List[Document]:
        if isinstance(k, bool) or not isinstance(k, int):
            raise ValueError("k must be an integer")
        if k <= 0:
            return []
        if not self.texts:
            return []
        scores = self.scores(query)
        ranked = sorted(range(len(self.documents)), key=lambda idx: (-scores[idx], idx))
        return [self.documents[idx] for idx in ranked[: min(k, len(ranked))] if scores[idx] > min_score]

    def score_map(self, query: str) -> Dict[str, float]:
        scores = self.scores(query)
        return {text: score for text, score in zip(self.texts, scores) if text}


def bm25_scores(query: str, docs: Sequence[Any], *, k1: float = 1.5, b: float = 0.75) -> List[float]:
    """Return BM25-style sparse scores for ``docs``."""
    return SparseRetriever(docs, k1=k1, b=b).scores(query)


def sparse_search(query: str, docs: Sequence[Any], k: int = 5) -> List[Document]:
    """Return top-k documents by BM25-style lexical score."""
    return SparseRetriever(docs).search(query, k=k, min_score=0.0)


def sparse_score_map(query: str, docs: Sequence[Any]) -> Dict[str, float]:
    """Return a text->score map for tracing/debugging."""
    return SparseRetriever(docs).score_map(query)
