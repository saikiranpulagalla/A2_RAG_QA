"""Baseline RAG pipeline.

Baseline is intentionally simple: chunk the entire corpus upfront, index all
chunks once, always retrieve, then answer from retrieved context. V3 keeps it
fair but stronger by using the same lightweight hybrid sparse+dense fusion as
A2-RAG, while still avoiding adaptive routing and parent-child retrieval.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Union

from a2_rag.late_chunking import extract_context_with_trace
from a2_rag.parent_child_retrieval import chunk_documents
from config import CHUNK_OVERLAP, CHUNK_SIZE, ENABLE_MMR_DIVERSITY, ENABLE_SPARSE_RETRIEVAL, MMR_DIVERSITY_LAMBDA, TOP_K
from providers.llm_factory import create_llm, invoke_llm_with_usage
from retrieval.diversity import duplicate_context_rate, select_diverse_contexts
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.hybrid import SparseRetriever
from utils import (
    QA_PROMPT_TEMPLATE,
    RAGException,
    RetrievalResult,
    STATIC_CORPUS_ABSTENTION,
    enforce_evidence_grounding,
    lexical_relevance,
    normalize_documents,
    quarantine_suspicious_documents,
    serialize_documents,
    setup_logger,
    static_corpus_policy,
    validate_query,
)
from vectorstore.local_store import build_local_store, similarity_search_local

logger = setup_logger(__name__)


class BaselineRAG:
    """Always-retrieve early-chunked RAG baseline."""

    def __init__(self, documents: Sequence[Any], model: str | None = None, k: int = TOP_K):
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise RAGException("k must be a positive integer")
        self.k = k
        self.documents = normalize_documents(documents)
        if not self.documents:
            raise RAGException("BaselineRAG requires at least one non-empty document")
        self.llm, self.provider_name = create_llm(model=model, purpose="generation")
        self.chunks = []
        self.vector_store = None
        self.sparse_retriever = None

    def _ensure_prepared(self) -> None:
        if self.vector_store is not None and self.sparse_retriever is not None:
            return
        self.chunks = chunk_documents(self.documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        if not self.chunks:
            raise RAGException("No chunks created for baseline index")
        self.vector_store = build_local_store(self.chunks, index_name="baseline_store")
        self.sparse_retriever = SparseRetriever(self.chunks)

    def prepare(self) -> int:
        """Build the index outside query latency timing for fair evaluation."""
        self._ensure_prepared()
        assert self.vector_store is not None
        return self.vector_store.count()

    def retrieve(self, query: str) -> RetrievalResult:
        query = validate_query(query)
        self._ensure_prepared()
        assert self.vector_store is not None
        assert self.sparse_retriever is not None
        dense_docs = similarity_search_local(self.vector_store, query, k=max(self.k * 2, self.k), fallback_empty=True)
        rankings = [dense_docs]
        sparse_queries = 0
        if ENABLE_SPARSE_RETRIEVAL:
            rankings.append(self.sparse_retriever.search(query, k=max(self.k * 2, self.k)))
            sparse_queries = 1
        docs = reciprocal_rank_fusion(rankings, limit=max(self.k * 2, self.k))
        if ENABLE_MMR_DIVERSITY:
            docs = select_diverse_contexts(query, docs, k=self.k, lambda_mult=MMR_DIVERSITY_LAMBDA)
        else:
            docs = docs[: self.k]

        docs, quarantined = quarantine_suspicious_documents(docs)
        scores = [lexical_relevance(query, doc) for doc in docs]
        return RetrievalResult(
            documents=docs,
            scores=scores,
            retrieval_stage="single",
            num_vector_queries=1,
            strategy="early_chunk_hybrid_sparse_dense_mmr" if ENABLE_SPARSE_RETRIEVAL else "early_chunk_dense_lexical",
            context_length=sum(len(doc.content) for doc in normalize_documents(docs)),
            metadata={
                "quality": {
                    "status": (
                        "suspicious_context_quarantined"
                        if quarantined and docs
                        else "suspicious_context_blocked"
                        if quarantined
                        else "good"
                        if docs
                        else "empty"
                    ),
                    "max_lexical_relevance": max(scores) if scores else 0.0,
                    "avg_lexical_relevance": sum(scores) / len(scores) if scores else 0.0,
                    "suspicious_chunks": quarantined,
                    "quarantined_suspicious_chunks": quarantined,
                    "duplicate_context_rate": duplicate_context_rate(docs),
                    "per_chunk_scores": scores,
                },
                "num_sparse_queries": sparse_queries,
            },
        )

    def answer(self, query: str, return_context: bool = False, return_metadata: bool = False) -> Union[str, Dict[str, Any]]:
        query = validate_query(query)
        start = time.time()
        policy = static_corpus_policy(query)
        if policy["blocked"]:
            metadata = {
                "answer": STATIC_CORPUS_ABSTENTION,
                "context": None,
                "policy": policy,
                "decision": {
                    "needs_retrieval": False,
                    "confidence": 1.0,
                    "reasoning": policy["reason"],
                    "source": "static_corpus_policy",
                    "llm_calls": 0,
                },
                "retrieval": {
                    "documents": [],
                    "context_length": 0,
                    "num_documents": 0,
                    "num_vector_queries": 0,
                    "num_sparse_queries": 0,
                    "stage": "policy_blocked",
                    "strategy": "none",
                    "scores": [],
                    "quality": {"status": "policy_blocked"},
                    "context_compression": {},
                },
                "grounding": {
                    "status": "abstained",
                    "is_supported": True,
                    "support_score": 1.0,
                    "rationale": policy["reason"],
                },
                "usage": {
                    "decision_llm_calls": 0,
                    "generation_llm_calls": 0,
                    "vector_queries": 0,
                    "sparse_queries": 0,
                    "total_llm_calls": 0,
                    "total_operations": 0,
                    "latency_sec": time.time() - start,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "model": self.provider_name,
            }
            return metadata if return_metadata or return_context else STATIC_CORPUS_ABSTENTION
        retrieval = self.retrieve(query)
        context, compression = extract_context_with_trace(query, retrieval.documents)
        retrieval.metadata["context_compression"] = compression
        prompt = QA_PROMPT_TEMPLATE.format(context=context, query=query)

        try:
            generation = invoke_llm_with_usage(self.llm, prompt)
            answer = generation.text
        except Exception as exc:
            raise RAGException(f"Baseline generation failed: {exc}") from exc

        answer, grounding = enforce_evidence_grounding(
            answer,
            context,
            retrieval_quality_status=retrieval.metadata.get("quality", {}).get("status"),
        )
        metadata = {
            "answer": answer,
            "policy": policy,
            "context": context if return_context else None,
            "decision": {
                "needs_retrieval": True,
                "confidence": 1.0,
                "reasoning": "Baseline always retrieves",
                "source": "fixed",
                "llm_calls": 0,
            },
            "retrieval": {
                "documents": serialize_documents(retrieval.documents),
                "context_length": len(context),
                "num_documents": len(retrieval.documents),
                "num_vector_queries": retrieval.num_vector_queries,
                "num_sparse_queries": retrieval.metadata.get("num_sparse_queries", 0),
                "stage": retrieval.retrieval_stage,
                "strategy": retrieval.strategy,
                "scores": retrieval.scores,
                "quality": retrieval.metadata.get("quality", {}),
                "context_compression": retrieval.metadata.get("context_compression", {}),
            },
            "grounding": grounding,
            "usage": {
                "decision_llm_calls": 0,
                "generation_llm_calls": 1,
                "vector_queries": retrieval.num_vector_queries,
                "sparse_queries": retrieval.metadata.get("num_sparse_queries", 0),
                "total_llm_calls": 1,
                "total_operations": 1 + retrieval.num_vector_queries + retrieval.metadata.get("num_sparse_queries", 0),
                "latency_sec": time.time() - start,
                "input_tokens": generation.input_tokens,
                "output_tokens": generation.output_tokens,
                "total_tokens": generation.total_tokens,
            },
            "model": self.provider_name,
        }
        if return_metadata or return_context:
            return metadata
        return answer

    def batch_answer(self, queries: List[str], return_metadata: bool = False):
        return [self.answer(query, return_metadata=return_metadata) for query in queries]
