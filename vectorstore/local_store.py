"""Small in-process cosine vector index for the research prototype.

The project does not expose a vector database server or need persistence.  A
local NumPy index therefore has a smaller attack surface and a clearer lifecycle
than embedding a general-purpose multi-tenant database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np

from config import ENABLE_LEXICAL_RERANK, LEXICAL_RERANK_WEIGHT
from embeddings.embedding_cache import embed_with_cache
from utils import Document, RetrievalException, normalize_documents, rerank_by_lexical_signal, setup_logger

logger = setup_logger(__name__)


@dataclass
class LocalVectorStore:
    """Immutable document matrix with normalized cosine vectors."""

    documents: List[Document]
    vectors: np.ndarray
    name: str = "documents"

    def count(self) -> int:
        return len(self.documents)


def _normalized_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    try:
        matrix = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise RetrievalException(f"Embedding vectors are not numeric: {exc}") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise RetrievalException(f"Expected a non-empty 2D embedding matrix, got shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise RetrievalException("Embedding vectors contain NaN or infinite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1.0)


def build_local_store(docs: Sequence[Any], index_name: str = "documents") -> LocalVectorStore:
    documents = normalize_documents(docs)
    if not documents:
        raise RetrievalException("Cannot build local vector index from empty documents")
    vectors = embed_with_cache([doc.content for doc in documents])
    if len(vectors) != len(documents):
        raise RetrievalException(
            f"Embedding backend returned {len(vectors)} vectors for {len(documents)} documents"
        )
    return LocalVectorStore(documents=documents, vectors=_normalized_matrix(vectors), name=index_name)


def similarity_search_local(
    vector_store: LocalVectorStore,
    query: str,
    k: int = 5,
    fallback_empty: bool = True,
) -> List[Document]:
    try:
        if isinstance(k, bool) or not isinstance(k, int):
            raise RetrievalException("k must be an integer")
        if k <= 0:
            return []
        if vector_store.count() == 0:
            return []
        query_vectors = embed_with_cache([query], persist=False)
        if not query_vectors or not any(float(value) != 0.0 for value in query_vectors[0]):
            logger.warning("Local vector search received a zero embedding query")
            return []
        query_matrix = _normalized_matrix(query_vectors)
        if query_matrix.shape[1] != vector_store.vectors.shape[1]:
            raise RetrievalException(
                "Query/document embedding dimensions differ: "
                f"{query_matrix.shape[1]} != {vector_store.vectors.shape[1]}"
            )
        scores = vector_store.vectors @ query_matrix[0]
        limit = min(k, vector_store.count())
        ranked_indices = np.argsort(-scores, kind="stable")[:limit]
        documents = [vector_store.documents[int(idx)] for idx in ranked_indices]
        if ENABLE_LEXICAL_RERANK and documents:
            ordered_texts = rerank_by_lexical_signal(query, documents, weight=LEXICAL_RERANK_WEIGHT)
            remaining = list(documents)
            reranked = []
            for text in ordered_texts:
                match_index = next((idx for idx, doc in enumerate(remaining) if doc.content == text), None)
                if match_index is not None:
                    reranked.append(remaining.pop(match_index))
            documents = reranked
        return documents
    except Exception as exc:
        if fallback_empty:
            logger.warning("Local vector search failed: %s", exc)
            return []
        if isinstance(exc, RetrievalException):
            raise
        raise RetrievalException(f"Local vector search failed: {exc}") from exc


def clear_local_store_cache() -> None:
    """Compatibility no-op; local indexes are owned by their pipeline objects."""
