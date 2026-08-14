"""A2-RAG pipeline: adaptive retrieval + hierarchical retrieval."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Union

from a2_rag.agent_decision import needs_retrieval
from a2_rag.late_chunking import extract_context_with_trace
from a2_rag.parent_child_retrieval import parent_child_retrieve, prepare_parent_index
from providers.llm_factory import create_llm, invoke_llm_with_usage
from utils import (
    NAIVE_QA_PROMPT_TEMPLATE,
    QA_PROMPT_TEMPLATE,
    AgentDecision,
    RAGException,
    RetrievalResult,
    STATIC_CORPUS_ABSTENTION,
    answer_support_report,
    enforce_evidence_grounding,
    serialize_documents,
    setup_logger,
    static_corpus_policy,
    normalize_documents,
    validate_query,
)

logger = setup_logger(__name__)


class A2RAG:
    """Adaptive and explainable RAG system."""

    def __init__(self, documents: Sequence[Any], model: str | None = None):
        self.documents = normalize_documents(documents)
        if not self.documents:
            raise RAGException("A2RAG requires at least one non-empty document")
        self.llm, self.provider_name = create_llm(model=model, purpose="generation")

    def prepare(self) -> int:
        """Build the reusable parent index outside query latency timing."""
        return prepare_parent_index(self.documents)

    def _make_retrieval_decision(self, query: str) -> AgentDecision:
        query = validate_query(query)
        return needs_retrieval(query, use_llm=True, fallback_to_heuristic=True)

    def _retrieve_context(self, query: str) -> tuple[str, RetrievalResult]:
        query = validate_query(query)
        result = parent_child_retrieve(query, self.documents)
        context, compression = extract_context_with_trace(query, result.documents)
        result.metadata["context_compression"] = compression
        return context, result

    def answer(self, query: str, return_metadata: bool = False) -> Union[str, Dict[str, Any]]:
        query = validate_query(query)
        start = time.time()
        policy = static_corpus_policy(query)
        if policy["blocked"]:
            if not return_metadata:
                return STATIC_CORPUS_ABSTENTION
            return {
                "answer": STATIC_CORPUS_ABSTENTION,
                "policy": policy,
                "decision": {
                    "needs_retrieval": False,
                    "confidence": 1.0,
                    "reasoning": policy["reason"],
                    "source": "static_corpus_policy",
                    "llm_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
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
                    "query_plan": {},
                    "query_variants": [],
                    "parent_quality": {},
                    "context_compression": {},
                    "corrective_retry": {},
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
        decision = self._make_retrieval_decision(query)
        retrieval_result = RetrievalResult(documents=[], retrieval_stage="decision_only", num_vector_queries=0, strategy="none")

        if decision.needs_retrieval:
            context, retrieval_result = self._retrieve_context(query)
            prompt = QA_PROMPT_TEMPLATE.format(context=context, query=query)
        else:
            context = "[Retrieval skipped by adaptive router]"
            prompt = NAIVE_QA_PROMPT_TEMPLATE.format(query=query)

        try:
            generation = invoke_llm_with_usage(self.llm, prompt)
            answer = generation.text
        except Exception as exc:
            raise RAGException(f"A2-RAG generation failed: {exc}") from exc

        if decision.needs_retrieval:
            answer, grounding = enforce_evidence_grounding(
                answer,
                context,
                retrieval_quality_status=retrieval_result.metadata.get("quality", {}).get("status"),
            )
        else:
            grounding = answer_support_report(answer, [])
        if not decision.needs_retrieval and grounding.get("status") == "no_context":
            grounding = {
                **grounding,
                "status": "retrieval_skipped",
                "is_supported": None,
                "support_score": None,
                "rationale": "Retrieval was intentionally skipped by the adaptive router; context-grounding is not applicable.",
            }

        if not return_metadata:
            return answer

        return {
            "answer": answer,
            "policy": policy,
            "decision": {
                "needs_retrieval": decision.needs_retrieval,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "source": decision.source,
                "llm_calls": decision.llm_calls,
                "input_tokens": getattr(decision, "input_tokens", 0),
                "output_tokens": getattr(decision, "output_tokens", 0),
                "total_tokens": getattr(decision, "total_tokens", 0),
            },
            "retrieval": {
                "documents": serialize_documents(retrieval_result.documents),
                "context_length": len(context),
                "num_documents": len(retrieval_result.documents),
                "num_vector_queries": retrieval_result.num_vector_queries,
                "num_sparse_queries": retrieval_result.metadata.get("num_sparse_queries", 0),
                "stage": retrieval_result.retrieval_stage,
                "strategy": retrieval_result.strategy,
                "scores": retrieval_result.scores,
                "quality": retrieval_result.metadata.get("quality", {}),
                "query_plan": retrieval_result.metadata.get("query_plan", {}),
                "query_variants": retrieval_result.metadata.get("query_variants", []),
                "parent_quality": retrieval_result.metadata.get("parent_quality", {}),
                "context_compression": retrieval_result.metadata.get("context_compression", {}),
                "corrective_retry": retrieval_result.metadata.get("corrective_retry", {}),
            },
            "grounding": grounding,
            "usage": {
                "decision_llm_calls": decision.llm_calls,
                "generation_llm_calls": 1,
                "vector_queries": retrieval_result.num_vector_queries,
                "sparse_queries": retrieval_result.metadata.get("num_sparse_queries", 0),
                "total_llm_calls": 1 + decision.llm_calls,
                "total_operations": 1 + decision.llm_calls + retrieval_result.num_vector_queries + retrieval_result.metadata.get("num_sparse_queries", 0),
                "latency_sec": time.time() - start,
                "input_tokens": getattr(decision, "input_tokens", 0) + generation.input_tokens,
                "output_tokens": getattr(decision, "output_tokens", 0) + generation.output_tokens,
                "total_tokens": getattr(decision, "total_tokens", 0) + generation.total_tokens,
            },
            "model": self.provider_name,
        }

    def batch_answer(self, queries: List[str], return_metadata: bool = False):
        return [self.answer(query, return_metadata=return_metadata) for query in queries]
