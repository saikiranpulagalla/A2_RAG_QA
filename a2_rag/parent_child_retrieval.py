"""Hierarchical parent-child retrieval for A2-RAG.

The retriever combines production-friendly ideas without pretending to implement
heavy research systems end-to-end:
1. adaptive retrieval budgets based on query complexity,
2. deterministic multi-query variants with Reciprocal Rank Fusion,
3. hybrid dense + BM25-style sparse retrieval for rare entities/numbers,
4. retrieval-time child chunking with lightweight source context,
5. MMR-style diversity control, and
6. corrective quality/security checks for weak or suspicious retrieval.

This is retrieval-time chunking, not true token-level late chunking; the docs
state the distinction honestly.
"""

from __future__ import annotations

from collections import OrderedDict
import threading
from typing import Any, Dict, List, Sequence

from config import (
    CHILD_K,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    ENABLE_CORRECTIVE_RETRIEVAL,
    ENABLE_MMR_DIVERSITY,
    ENABLE_SPARSE_RETRIEVAL,
    MIN_RETRIEVAL_RELEVANCE,
    MMR_DIVERSITY_LAMBDA,
    PARENT_K,
    CORRECTIVE_RETRY_MULTIPLIER,
    MAX_CORRECTIVE_VARIANTS,
    MAX_PARENT_INDEX_CACHE,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)
from retrieval.diversity import duplicate_context_rate, select_diverse_contexts
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.hybrid import SparseRetriever
from retrieval.query_planner import analyze_query, build_query_variants
from retrieval.text_splitter import split_text
from utils import (
    RAGException,
    Document,
    RetrievalResult,
    corpus_fingerprint,
    detect_prompt_injection,
    lexical_relevance,
    normalize_documents,
    quarantine_suspicious_documents,
    setup_logger,
)
from vectorstore.local_store import build_local_store, similarity_search_local

logger = setup_logger(__name__)
_parent_index_cache: OrderedDict[str, tuple[List[Document], Any, SparseRetriever]] = OrderedDict()
_parent_index_lock = threading.RLock()


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RAGException(f"{name} must be a positive integer")
    return value


def chunk_documents(docs: Sequence[Any], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """Chunk documents while preserving structured, untrusted provenance."""
    _positive_int(chunk_size, "chunk_size")
    if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int) or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise RAGException("chunk_overlap must be a non-negative integer smaller than chunk_size")
    normalized = normalize_documents(docs)
    chunks: List[Document] = []
    for doc_idx, doc in enumerate(normalized):
        base_metadata = dict(doc.metadata)
        base_metadata.setdefault("index", doc_idx)
        cursor = 0
        for chunk_idx, chunk in enumerate(
            split_text(doc.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            chunk = chunk.strip()
            if chunk:
                metadata = dict(base_metadata)
                if "chunk_index" in metadata:
                    metadata["parent_chunk_index"] = metadata["chunk_index"]
                if "chunk_char_start" in metadata:
                    metadata["parent_chunk_char_start"] = metadata["chunk_char_start"]
                if "chunk_char_end" in metadata:
                    metadata["parent_chunk_char_end"] = metadata["chunk_char_end"]
                start = doc.content.find(chunk, cursor)
                if start < 0:
                    start = doc.content.find(chunk)
                start = max(0, start)
                cursor = start + max(1, len(chunk) - chunk_overlap)
                metadata["chunk_index"] = chunk_idx
                metadata["chunk_char_start"] = start
                metadata["chunk_char_end"] = start + len(chunk)
                chunks.append(Document(content=chunk, metadata=metadata))
    return chunks


def _get_parent_index(parent_docs: Sequence[Any]) -> tuple[List[Document], Any, SparseRetriever]:
    cache_key = corpus_fingerprint(parent_docs) + f":{PARENT_CHUNK_SIZE}:{PARENT_CHUNK_OVERLAP}"
    with _parent_index_lock:
        cached = _parent_index_cache.get(cache_key)
        if cached is not None:
            _parent_index_cache.move_to_end(cache_key)
            return cached

        parent_units = chunk_documents(
            parent_docs,
            chunk_size=PARENT_CHUNK_SIZE,
            chunk_overlap=PARENT_CHUNK_OVERLAP,
        )
        if not parent_units:
            raise RAGException("No parent retrieval units could be created")
        vector_store = build_local_store(parent_units, index_name="parent_store")
        sparse_retriever = SparseRetriever(parent_units)
        value = (parent_units, vector_store, sparse_retriever)
        _parent_index_cache[cache_key] = value
        while len(_parent_index_cache) > MAX_PARENT_INDEX_CACHE:
            _parent_index_cache.popitem(last=False)
        return value


def prepare_parent_index(parent_docs: Sequence[Any]) -> int:
    """Build the reusable parent index outside per-query latency timing."""
    parent_units, _vector_store, _sparse_retriever = _get_parent_index(parent_docs)
    return len(parent_units)


def _query_variants(query: str) -> List[str]:
    plan = analyze_query(query)
    return plan.variants or [query]


def _retrieve_with_fusion(
    collection,
    corpus_docs: Sequence[Any],
    query: str,
    k: int,
    variants: Sequence[str] | None = None,
    sparse_retriever: SparseRetriever | None = None,
) -> List[Document]:
    variants = list(variants or _query_variants(query))
    rankings = []
    # Retrieve a slightly larger pool before diversity selection.
    pool_k = max(k, min(k * 2, len(corpus_docs) if corpus_docs else k))
    for variant in variants:
        dense_docs = similarity_search_local(collection, variant, k=pool_k, fallback_empty=False)
        rankings.append(dense_docs)
        if ENABLE_SPARSE_RETRIEVAL:
            sparse = sparse_retriever or SparseRetriever(corpus_docs)
            rankings.append(sparse.search(variant, k=pool_k))

    fused = reciprocal_rank_fusion(rankings, limit=max(k, pool_k))
    if ENABLE_MMR_DIVERSITY:
        return select_diverse_contexts(query, fused, k=k, lambda_mult=MMR_DIVERSITY_LAMBDA)
    return fused[:k]


def _retrieve_safe_with_backfill(
    collection,
    corpus_docs: Sequence[Any],
    query: str,
    k: int,
    *,
    variants: Sequence[str],
    sparse_retriever: SparseRetriever | None,
) -> tuple[List[Document], int, int]:
    """Quarantine risky results and expand the candidate pool when needed.

    Retrieval rankings are untrusted too: dropping a malicious top result must
    not turn a safe lower-ranked passage into an artificial empty retrieval.
    """
    raw_docs = _retrieve_with_fusion(
        collection,
        corpus_docs,
        query,
        k=k,
        variants=variants,
        sparse_retriever=sparse_retriever,
    )
    safe_docs, _ = quarantine_suspicious_documents(raw_docs)
    suspicious_keys = {doc.content for doc in normalize_documents(raw_docs) if detect_prompt_injection(doc.content)["is_suspicious"]}
    searches = 1
    if len(safe_docs) >= k or len(corpus_docs) <= len(raw_docs):
        return safe_docs[:k], len(suspicious_keys), searches

    expanded_k = min(len(corpus_docs), max(k * 4, k + 8))
    expanded_docs = _retrieve_with_fusion(
        collection,
        corpus_docs,
        query,
        k=expanded_k,
        variants=variants,
        sparse_retriever=sparse_retriever,
    )
    searches += 1
    combined: List[Document] = []
    seen = set()
    for doc in [*normalize_documents(raw_docs), *normalize_documents(expanded_docs)]:
        if doc.content in seen:
            continue
        seen.add(doc.content)
        combined.append(doc)
    safe_docs, _ = quarantine_suspicious_documents(combined)
    suspicious_keys.update(doc.content for doc in combined if detect_prompt_injection(doc.content)["is_suspicious"])
    return safe_docs[:k], len(suspicious_keys), searches


def _quality_report(query: str, docs: Sequence[Any], *, quarantined_count: int = 0) -> Dict[str, Any]:
    scores = [lexical_relevance(query, doc) for doc in docs]
    suspicious = [detect_prompt_injection(doc.content) for doc in normalize_documents(docs)]
    max_score = max(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0
    suspicious_count = quarantined_count + sum(1 for r in suspicious if r["is_suspicious"])
    duplicate_rate = duplicate_context_rate(docs)
    if not docs and suspicious_count:
        status = "suspicious_context_blocked"
    elif not docs:
        status = "empty"
    elif suspicious_count:
        status = "suspicious_context_quarantined"
    elif max_score < MIN_RETRIEVAL_RELEVANCE:
        status = "weak"
    else:
        status = "good"
    return {
        "status": status,
        "max_lexical_relevance": max_score,
        "avg_lexical_relevance": avg_score,
        "suspicious_chunks": suspicious_count,
        "quarantined_suspicious_chunks": quarantined_count,
        "duplicate_context_rate": duplicate_rate,
        "per_chunk_scores": scores,
    }


def retrieve_parents(query: str, parent_docs: Sequence[Any], k: int = PARENT_K, variants: Sequence[str] | None = None) -> RetrievalResult:
    try:
        _positive_int(k, "k")
        parent_units, parent_store, parent_sparse = _get_parent_index(parent_docs)
        effective_variants = list(variants or _query_variants(query))
        docs, quarantined, searches = _retrieve_safe_with_backfill(
            parent_store,
            parent_units,
            query,
            k,
            variants=effective_variants,
            sparse_retriever=parent_sparse,
        )
        quality = _quality_report(query, docs, quarantined_count=quarantined)
        vector_queries = max(1, len(effective_variants)) * searches
        sparse_queries = vector_queries if ENABLE_SPARSE_RETRIEVAL else 0
        return RetrievalResult(
            documents=docs,
            scores=quality.get("per_chunk_scores", []),
            retrieval_stage="parent",
            num_vector_queries=vector_queries,
            strategy="parent_multiquery_rrf_hybrid_sparse_dense" if ENABLE_SPARSE_RETRIEVAL else "parent_multiquery_rrf_dense_lexical",
            metadata={
                "quality": quality,
                "num_sparse_queries": sparse_queries,
                "security_backfill_searches": searches - 1,
            },
        )
    except Exception as exc:
        raise RAGException(f"Parent retrieval failed: {exc}") from exc


def parent_child_retrieve(
    query: str,
    parent_docs: Sequence[Any],
    parent_k: int = PARENT_K,
    child_k: int = CHILD_K,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    adaptive: bool = True,
) -> RetrievalResult:
    try:
        _positive_int(parent_k, "parent_k")
        _positive_int(child_k, "child_k")
        _positive_int(chunk_size, "chunk_size")
        if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int) or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise RAGException("chunk_overlap must be a non-negative integer smaller than chunk_size")
        plan = analyze_query(query)
        variants = plan.variants or [query]
        effective_parent_k = max(parent_k, plan.parent_k) if adaptive else parent_k
        effective_child_k = max(child_k, plan.child_k) if adaptive else child_k

        parent_result = retrieve_parents(query, parent_docs, k=effective_parent_k, variants=variants)
        if not parent_result.documents:
            return RetrievalResult(
                documents=[],
                retrieval_stage="child",
                num_vector_queries=parent_result.num_vector_queries,
                strategy="parent_child_multiquery_rrf_hybrid",
                metadata={
                    "query_plan": plan.to_dict(),
                    "quality": _quality_report(query, []),
                    "num_sparse_queries": parent_result.metadata.get("num_sparse_queries", 0),
                },
            )

        child_chunks = chunk_documents(parent_result.documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not child_chunks:
            return RetrievalResult(
                documents=[],
                retrieval_stage="child",
                num_vector_queries=parent_result.num_vector_queries,
                strategy="parent_child_multiquery_rrf_hybrid",
                metadata={
                    "query_plan": plan.to_dict(),
                    "quality": _quality_report(query, []),
                    "num_sparse_queries": parent_result.metadata.get("num_sparse_queries", 0),
                },
            )

        child_store = build_local_store(child_chunks, index_name="child_store")
        child_sparse = SparseRetriever(child_chunks)
        child_docs, quarantined, child_searches = _retrieve_safe_with_backfill(
            child_store,
            child_chunks,
            query,
            effective_child_k,
            variants=variants,
            sparse_retriever=child_sparse,
        )

        quality = _quality_report(query, child_docs, quarantined_count=quarantined)
        child_vector_queries = len(variants) * child_searches
        child_sparse_queries = len(variants) * child_searches if ENABLE_SPARSE_RETRIEVAL else 0
        corrective_retry = {"attempted": False, "accepted": False}
        retry_backfill_searches = 0
        if ENABLE_CORRECTIVE_RETRIEVAL and quality.get("status") in {"weak", "empty"}:
            retry_variants = build_query_variants(query, max_variants=MAX_CORRECTIVE_VARIANTS)
            retry_k = max(effective_child_k + 1, effective_child_k * CORRECTIVE_RETRY_MULTIPLIER)
            retry_pool, retry_quarantined, retry_searches = _retrieve_safe_with_backfill(
                child_store,
                child_chunks,
                query,
                retry_k,
                variants=retry_variants,
                sparse_retriever=child_sparse,
            )
            retry_docs = select_diverse_contexts(query, retry_pool, k=effective_child_k, lambda_mult=MMR_DIVERSITY_LAMBDA)
            retry_quality = _quality_report(query, retry_docs, quarantined_count=retry_quarantined)
            child_vector_queries += len(retry_variants) * retry_searches
            child_sparse_queries += len(retry_variants) * retry_searches if ENABLE_SPARSE_RETRIEVAL else 0
            retry_backfill_searches = retry_searches - 1
            original_score = float(quality.get("max_lexical_relevance", 0.0) or 0.0)
            retry_score = float(retry_quality.get("max_lexical_relevance", 0.0) or 0.0)
            accepted = bool(retry_docs) and (not child_docs or retry_score >= original_score)
            corrective_retry = {
                "attempted": True,
                "accepted": accepted,
                "variants": retry_variants,
                "original_status": quality.get("status"),
                "retry_status": retry_quality.get("status"),
                "original_max_lexical_relevance": original_score,
                "retry_max_lexical_relevance": retry_score,
            }
            if accepted:
                child_docs = retry_docs
                quality = retry_quality
        return RetrievalResult(
            documents=child_docs,
            scores=quality.get("per_chunk_scores", []),
            retrieval_stage="child",
            num_vector_queries=parent_result.num_vector_queries + child_vector_queries,
            strategy=(
                "adaptive_parent_child_multiquery_rrf_hybrid_contextual_mmr"
                if ENABLE_SPARSE_RETRIEVAL and ENABLE_MMR_DIVERSITY
                else "adaptive_parent_child_multiquery_rrf_contextual_chunks"
            ),
            context_length=sum(len(d.content) for d in normalize_documents(child_docs)),
            metadata={
                "query_plan": plan.to_dict(),
                "parent_quality": parent_result.metadata.get("quality", {}),
                "quality": quality,
                "parent_k": effective_parent_k,
                "child_k": effective_child_k,
                "query_variants": variants,
                "num_sparse_queries": parent_result.metadata.get("num_sparse_queries", 0) + child_sparse_queries,
                "security_backfill_searches": parent_result.metadata.get("security_backfill_searches", 0) + child_searches - 1 + retry_backfill_searches,
                "corrective_retry": corrective_retry,
            },
        )
    except Exception as exc:
        raise RAGException(f"Two-stage retrieval failed: {exc}") from exc


def clear_parent_cache() -> None:
    with _parent_index_lock:
        _parent_index_cache.clear()
