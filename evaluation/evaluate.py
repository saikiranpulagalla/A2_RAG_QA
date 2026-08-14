"""Evaluation utilities for A2-RAG."""

from __future__ import annotations

import csv
import inspect
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from data import get_reference_answers
from evaluation.benchmark_contract import load_expected_abstention_indexes
from utils import documents_to_texts, is_abstention, normalize_document, normalize_documents, setup_logger

logger = setup_logger(__name__)


def normalize_answer(answer: str) -> str:
    answer = (answer or "").lower()
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    answer = re.sub(r"[^a-z0-9\s]", " ", answer)
    return " ".join(answer.split())


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def best_reference_metrics(prediction: str, references: Sequence[str]) -> tuple[bool, float]:
    """Score a prediction against every accepted reference answer."""
    usable = [reference for reference in references if str(reference or "").strip()]
    if not usable:
        return False, 0.0
    return any(exact_match(prediction, reference) for reference in usable), max(
        f1_score(prediction, reference) for reference in usable
    )


def _document_id(document: Any, fallback: int) -> str:
    normalized = normalize_document(document, default_index=fallback)
    return str(normalized.metadata.get("doc_id", normalized.metadata.get("index", fallback)))


def retrieval_hit_rate(
    answer: str | Sequence[str],
    retrieved_docs: Sequence[Any],
    exact_match_required: bool = False,
    *,
    gold_document_ids: Sequence[str] | None = None,
) -> bool:
    if gold_document_ids:
        expected = {str(item) for item in gold_document_ids}
        return any(_document_id(doc, index) in expected for index, doc in enumerate(retrieved_docs))
    answers = [answer] if isinstance(answer, str) else list(answer)
    answer_norms = [normalize_answer(item) for item in answers if normalize_answer(item)]
    if not answer_norms or not retrieved_docs:
        return False
    for doc in documents_to_texts(retrieved_docs):
        doc_norm = normalize_answer(doc)
        for answer_norm in answer_norms:
            if exact_match_required and doc_norm == answer_norm:
                return True
            if not exact_match_required and re.search(
                rf"(?<![a-z0-9]){re.escape(answer_norm)}(?![a-z0-9])", doc_norm
            ):
                return True
    return False


def short_answer(text: Any) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    first_line = text.splitlines()[0].strip()
    return first_line


class EvaluationResult:
    def __init__(self, setup_latency_sec: float = 0.0) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.setup_latency_sec = float(setup_latency_sec)

    def add_row(self, row: Dict[str, Any]) -> None:
        self.rows.append(row)

    @property
    def exact_matches(self) -> List[bool]:
        return [bool(r.get("em")) for r in self.rows if r.get("scored_qa", True)]

    @property
    def f1_scores(self) -> List[float]:
        return [float(r.get("f1", 0.0)) for r in self.rows if r.get("scored_qa", True)]

    @property
    def retrieval_hits(self) -> List[bool]:
        return [
            bool(r.get("retrieval_hit"))
            for r in self.rows
            if r.get("scored_qa", True) and r.get("retrieval_evaluable", False)
        ]

    @property
    def latencies(self) -> List[float]:
        return [float(r.get("latency_sec", 0.0)) for r in self.rows]

    @property
    def num_queries(self) -> List[int]:
        return [int(r.get("total_operations", 0)) for r in self.rows]

    @property
    def metrics_metadata(self) -> List[Dict[str, Any]]:
        return self.rows

    def summary(self) -> Dict[str, Any]:
        total_examples = len(self.rows)
        qa_rows = [row for row in self.rows if row.get("scored_qa", True)]
        retrieval_rows = [row for row in qa_rows if row.get("retrieval_evaluable", False)]
        qa_examples = len(qa_rows)
        abstention_rows = [row for row in self.rows if row.get("expected_abstention")]
        if not total_examples:
            return {
                "em": 0.0,
                "f1": 0.0,
                "hit_rate": 0.0,
                "avg_latency": 0.0,
                "amortized_latency": self.setup_latency_sec,
                "setup_latency_sec": self.setup_latency_sec,
                "avg_num_queries": 0.0,
                "num_examples": 0,
                "num_qa_examples": 0,
                "num_expected_abstentions": 0,
                "abstention_success_rate": 0.0,
                "benchmark_type": "paired_closed_corpus",
            }

        def average(rows: Sequence[Dict[str, Any]], key: str) -> float:
            return sum(float(row.get(key, 0.0) or 0.0) for row in rows) / max(1, len(rows))

        return {
            "em": sum(self.exact_matches) / max(1, qa_examples),
            "f1": sum(self.f1_scores) / max(1, qa_examples),
            "hit_rate": sum(self.retrieval_hits) / max(1, len(retrieval_rows)),
            "avg_latency": sum(self.latencies) / total_examples,
            "amortized_latency": (sum(self.latencies) + self.setup_latency_sec) / total_examples,
            "setup_latency_sec": self.setup_latency_sec,
            "avg_num_queries": sum(self.num_queries) / total_examples,
            "avg_llm_calls": average(self.rows, "total_llm_calls"),
            "avg_vector_queries": average(self.rows, "vector_queries"),
            "retrieval_rate": sum(1 for r in qa_rows if r.get("needs_retrieval")) / max(1, qa_examples),
            "weak_retrieval_rate": sum(1 for r in qa_rows if r.get("retrieval_quality_status") in {"weak", "empty"}) / max(1, qa_examples),
            "suspicious_context_rate": sum(1 for r in qa_rows if float(r.get("suspicious_chunks", 0) or 0) > 0) / max(1, qa_examples),
            "answer_support_rate": sum(1 for r in qa_rows if r.get("answer_supported")) / max(1, qa_examples),
            "abstention_rate": sum(1 for r in self.rows if str(r.get("grounding_status", "")).startswith("abstained")) / total_examples,
            "abstention_success_rate": sum(1 for r in abstention_rows if r.get("abstention_success")) / max(1, len(abstention_rows)),
            "avg_grounding_score": average(qa_rows, "grounding_score"),
            "avg_sparse_queries": average(self.rows, "sparse_queries"),
            "avg_duplicate_context_rate": average(self.rows, "duplicate_context_rate"),
            "corrective_retry_rate": sum(1 for r in self.rows if r.get("corrective_retry_attempted")) / total_examples,
            "corrective_retry_accept_rate": sum(1 for r in self.rows if r.get("corrective_retry_accepted"))
            / max(1, sum(1 for r in self.rows if r.get("corrective_retry_attempted"))),
            "avg_context_compression_ratio": average(self.rows, "context_compression_ratio"),
            "avg_selected_evidence_sentences": average(self.rows, "selected_evidence_sentences"),
            "avg_input_tokens": average(self.rows, "input_tokens"),
            "avg_output_tokens": average(self.rows, "output_tokens"),
            "avg_total_tokens": average(self.rows, "total_tokens"),
            "avg_reciprocal_rank": average(qa_rows, "reciprocal_rank"),
            "num_examples": total_examples,
            "num_qa_examples": qa_examples,
            "num_expected_abstentions": len(abstention_rows),
            "num_retrieval_evaluable": len(retrieval_rows),
            "benchmark_type": "paired_closed_corpus",
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"summary": self.summary(), "setup_latency_sec": self.setup_latency_sec, "rows": self.rows}


def _call_model(model: Any, question: str) -> Dict[str, Any]:
    answer_method = model.answer
    try:
        parameters = inspect.signature(answer_method).parameters.values()
        accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
        parameter_names = {parameter.name for parameter in parameters}
    except (TypeError, ValueError):
        accepts_kwargs = True
        parameter_names = set()

    if accepts_kwargs or "return_metadata" in parameter_names:
        raw = answer_method(question, return_metadata=True)
    elif "return_context" in parameter_names:
        raw = answer_method(question, return_context=True)
    else:
        raw = answer_method(question)
    if isinstance(raw, dict):
        return raw
    return {"answer": raw, "decision": {}, "retrieval": {}, "usage": {}}


def evaluate_rag(
    model: Any,
    questions: Sequence[Dict[str, Any]],
    include_retrieval_metrics: bool = True,
    initial_setup_latency_sec: float = 0.0,
    expected_abstention_indexes: Sequence[int] | None = None,
) -> EvaluationResult:
    expected_abstentions = (
        set(load_expected_abstention_indexes())
        if expected_abstention_indexes is None
        else set(expected_abstention_indexes)
    )
    result = EvaluationResult(setup_latency_sec=initial_setup_latency_sec)
    corpus_by_context: Dict[str, List[str]] = {}
    for index, doc in enumerate(normalize_documents(getattr(model, "documents", []))):
        corpus_by_context.setdefault(doc.content, []).append(_document_id(doc, index))
    prepare = getattr(model, "prepare", None)
    if callable(prepare):
        setup_started = time.perf_counter()
        prepare()
        result.setup_latency_sec += time.perf_counter() - setup_started
    for idx, example in enumerate(questions, start=1):
        question = example.get("question", "")
        references = [short_answer(answer) for answer in get_reference_answers(example)]
        references = [answer for answer in references if answer]
        if not question or not references:
            continue
        reference = references[0]
        gold_document_ids = corpus_by_context.get(str(example.get("context", "")), [])

        started = time.perf_counter()
        try:
            output = _call_model(model, question)
            latency = time.perf_counter() - started
            prediction = short_answer(output.get("answer", ""))
            retrieval = output.get("retrieval") or {}
            decision = output.get("decision") or {}
            usage = output.get("usage") or {}
            grounding = output.get("grounding") or {}
            policy = output.get("policy") or {}
            context_compression = retrieval.get("context_compression") or {}
            corrective_retry = retrieval.get("corrective_retry") or {}
            retrieved_docs = retrieval.get("documents") or []
            # Benchmark labels are a fixture owned by the evaluator. A model is
            # not allowed to improve EM/F1 by declaring an answerable row out
            # of scope.
            expected_abstention = idx in expected_abstentions
            model_policy_blocked = bool(policy.get("blocked"))
            hit = (
                retrieval_hit_rate(references, retrieved_docs, gold_document_ids=gold_document_ids)
                if include_retrieval_metrics and gold_document_ids
                else False
            )
            reciprocal_rank = 0.0
            gold_ids = {str(item) for item in gold_document_ids}
            for rank, doc in enumerate(retrieved_docs, start=1):
                if _document_id(doc, rank - 1) in gold_ids:
                    reciprocal_rank = 1.0 / rank
                    break
            em, f1 = best_reference_metrics(prediction, references) if not expected_abstention else (False, 0.0)
            row = {
                "index": idx,
                "question": question,
                "reference": reference,
                "references": references,
                "prediction": prediction,
                "em": em,
                "f1": f1,
                "retrieval_hit": hit,
                "gold_document_ids": gold_document_ids,
                "retrieval_evaluable": bool(gold_document_ids),
                "recall_at_k": float(hit) if gold_document_ids else None,
                "reciprocal_rank": reciprocal_rank if gold_document_ids else None,
                "expected_abstention": expected_abstention,
                "abstention_success": is_abstention(prediction) if expected_abstention else None,
                "scored_qa": not expected_abstention,
                "model_policy_blocked": model_policy_blocked,
                "evaluation_category": (
                    "expected_abstention"
                    if expected_abstention
                    else "unexpected_model_policy_abstention"
                    if model_policy_blocked
                    else "paired_closed_corpus_qa"
                ),
                "needs_retrieval": bool(decision.get("needs_retrieval", bool(retrieved_docs))),
                "decision_confidence": decision.get("confidence"),
                "decision_source": decision.get("source"),
                "decision_reasoning": decision.get("reasoning"),
                "retrieved_docs": retrieved_docs,
                "num_retrieved_docs": retrieval.get("num_documents", len(retrieved_docs)),
                "context_length": retrieval.get("context_length", 0),
                "retrieval_stage": retrieval.get("stage"),
                "retrieval_strategy": retrieval.get("strategy"),
                "retrieval_quality_status": (retrieval.get("quality") or {}).get("status"),
                "max_lexical_relevance": (retrieval.get("quality") or {}).get("max_lexical_relevance", 0.0),
                "avg_lexical_relevance": (retrieval.get("quality") or {}).get("avg_lexical_relevance", 0.0),
                "suspicious_chunks": (retrieval.get("quality") or {}).get("suspicious_chunks", 0),
                "duplicate_context_rate": (retrieval.get("quality") or {}).get("duplicate_context_rate", 0.0),
                "query_complexity": (retrieval.get("query_plan") or {}).get("complexity"),
                "grounding_status": grounding.get("status"),
                "answer_supported": bool(grounding.get("is_supported", False)),
                "grounding_score": grounding.get("support_score", 0.0),
                "unsupported_fact_markers": grounding.get("unsupported_fact_markers", []),
                "context_compression_ratio": context_compression.get("compression_ratio", 0.0),
                "selected_evidence_sentences": context_compression.get("selected_sentences", 0),
                "corrective_retry_attempted": bool(corrective_retry.get("attempted", False)),
                "corrective_retry_accepted": bool(corrective_retry.get("accepted", False)),
                "latency_sec": latency,
                "reported_latency_sec": usage.get("latency_sec"),
                "decision_llm_calls": usage.get("decision_llm_calls", 0),
                "generation_llm_calls": usage.get("generation_llm_calls", 0),
                "total_llm_calls": usage.get("total_llm_calls", 0),
                "vector_queries": usage.get("vector_queries", 0),
                "sparse_queries": usage.get("sparse_queries", retrieval.get("num_sparse_queries", 0)),
                "total_operations": usage.get("total_operations", 0),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "model": output.get("model"),
            }
        except Exception as exc:
            row = {
                "index": idx,
                "question": question,
                "reference": reference,
                "prediction": "",
                "em": False,
                "f1": 0.0,
                "retrieval_hit": False,
                "gold_document_ids": gold_document_ids,
                "retrieval_evaluable": bool(gold_document_ids),
                "reciprocal_rank": 0.0 if gold_document_ids else None,
                "expected_abstention": False,
                "abstention_success": None,
                "scored_qa": True,
                "evaluation_category": "error",
                "error": str(exc),
                "latency_sec": time.perf_counter() - started,
                "total_operations": 0,
            }
        result.add_row(row)
    return result


def compare_models(models: Dict[str, Any], questions: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    return {name: evaluate_rag(model, questions).summary() for name, model in models.items()}


def export_results_to_json(results: EvaluationResult, filepath: str) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results.to_dict(), indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def export_comparison_csv(model_results: Dict[str, Dict[str, float]], filepath: str) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Benchmark Type", "Model", "EM", "F1", "Gold Recall@k", "Gold MRR", "QA Examples", "Expected Abstentions",
        "Abstention Success Rate", "Retrieval Rate", "Avg Query Latency", "Setup Latency", "Amortized Latency",
        "Avg LLM Calls", "Avg Vector Queries", "Avg Sparse Queries", "Avg Operations", "Weak Retrieval Rate",
        "Suspicious Context Rate", "Answer Support Rate", "Avg Grounding Score", "Abstention Rate",
        "Avg Duplicate Context Rate", "Corrective Retry Rate", "Corrective Accept Rate",
        "Avg Context Compression Ratio", "Avg Selected Evidence Sentences", "Avg Input Tokens", "Avg Output Tokens", "Avg Total Tokens", "Num Examples",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, metrics in model_results.items():
            writer.writerow(_csv_safe_row({
                "Benchmark Type": metrics.get("benchmark_type", "paired_closed_corpus"),
                "Model": name,
                "EM": f"{metrics.get('em', 0.0):.4f}",
                "F1": f"{metrics.get('f1', 0.0):.4f}",
                "Gold Recall@k": f"{metrics.get('hit_rate', 0.0):.4f}",
                "Gold MRR": f"{metrics.get('avg_reciprocal_rank', 0.0):.4f}",
                "QA Examples": int(metrics.get("num_qa_examples", metrics.get("num_examples", 0))),
                "Expected Abstentions": int(metrics.get("num_expected_abstentions", 0)),
                "Abstention Success Rate": f"{metrics.get('abstention_success_rate', 0.0):.4f}",
                "Retrieval Rate": f"{metrics.get('retrieval_rate', 0.0):.4f}",
                "Avg Query Latency": f"{metrics.get('avg_latency', 0.0):.4f}",
                "Setup Latency": f"{metrics.get('setup_latency_sec', 0.0):.4f}",
                "Amortized Latency": f"{metrics.get('amortized_latency', 0.0):.4f}",
                "Avg LLM Calls": f"{metrics.get('avg_llm_calls', 0.0):.2f}",
                "Avg Vector Queries": f"{metrics.get('avg_vector_queries', 0.0):.2f}",
                "Avg Sparse Queries": f"{metrics.get('avg_sparse_queries', 0.0):.2f}",
                "Avg Operations": f"{metrics.get('avg_num_queries', 0.0):.2f}",
                "Weak Retrieval Rate": f"{metrics.get('weak_retrieval_rate', 0.0):.4f}",
                "Suspicious Context Rate": f"{metrics.get('suspicious_context_rate', 0.0):.4f}",
                "Answer Support Rate": f"{metrics.get('answer_support_rate', 0.0):.4f}",
                "Avg Grounding Score": f"{metrics.get('avg_grounding_score', 0.0):.4f}",
                "Abstention Rate": f"{metrics.get('abstention_rate', 0.0):.4f}",
                "Avg Duplicate Context Rate": f"{metrics.get('avg_duplicate_context_rate', 0.0):.4f}",
                "Corrective Retry Rate": f"{metrics.get('corrective_retry_rate', 0.0):.4f}",
                "Corrective Accept Rate": f"{metrics.get('corrective_retry_accept_rate', 0.0):.4f}",
                "Avg Context Compression Ratio": f"{metrics.get('avg_context_compression_ratio', 0.0):.4f}",
                "Avg Selected Evidence Sentences": f"{metrics.get('avg_selected_evidence_sentences', 0.0):.2f}",
                "Avg Input Tokens": f"{metrics.get('avg_input_tokens', 0.0):.2f}",
                "Avg Output Tokens": f"{metrics.get('avg_output_tokens', 0.0):.2f}",
                "Avg Total Tokens": f"{metrics.get('avg_total_tokens', 0.0):.2f}",
                "Num Examples": int(metrics.get("num_examples", 0)),
            }))


def export_per_question_csv(results: EvaluationResult, model_name: str, filepath: str) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "index", "question", "reference", "references", "prediction", "em", "f1", "retrieval_hit",
        "needs_retrieval", "decision_confidence", "decision_source", "retrieval_stage",
        "retrieval_quality_status", "max_lexical_relevance", "suspicious_chunks", "duplicate_context_rate", "query_complexity",
        "grounding_status", "answer_supported", "grounding_score", "unsupported_fact_markers",
        "gold_document_ids", "retrieval_evaluable", "recall_at_k", "reciprocal_rank", "expected_abstention", "abstention_success", "evaluation_category",
        "context_compression_ratio", "selected_evidence_sentences", "corrective_retry_attempted", "corrective_retry_accepted",
        "num_retrieved_docs", "latency_sec", "total_llm_calls", "vector_queries", "sparse_queries", "total_operations",
        "input_tokens", "output_tokens", "total_tokens", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results.rows:
            writer.writerow(_csv_safe_row({"model": model_name, **row}))


def _csv_safe_value(value: Any) -> Any:
    """Prevent untrusted values from becoming spreadsheet formulas on export."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, (list, tuple, dict)):
        value = json.dumps(value, ensure_ascii=False, default=str)
    text = str(value)
    if text.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def _csv_safe_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _csv_safe_value(value) for key, value in row.items()}


def generate_evaluation_summary(model_results: Dict[str, Dict[str, float]], output_file: str = "results/summary.txt") -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["A2-RAG Evaluation Summary", "=" * 30, "", "Benchmark: paired closed corpus; not open-domain QA performance.", ""]
    for name, m in model_results.items():
        lines.extend([
            f"{name}:",
            f"  Benchmark:       {m.get('benchmark_type', 'paired_closed_corpus')}",
            f"  EM:              {m.get('em', 0.0):.4f}",
            f"  F1:              {m.get('f1', 0.0):.4f}",
            f"  Gold Recall@k:   {m.get('hit_rate', 0.0):.4f}",
            f"  Gold MRR:        {m.get('avg_reciprocal_rank', 0.0):.4f}",
            f"  QA Examples:     {m.get('num_qa_examples', m.get('num_examples', 0))}",
            f"  Expected Abst.:  {m.get('num_expected_abstentions', 0)}",
            f"  Abstention OK:   {m.get('abstention_success_rate', 0.0):.4f}",
            f"  Retrieval Rate:  {m.get('retrieval_rate', 0.0):.4f}",
            f"  Avg LLM Calls:   {m.get('avg_llm_calls', 0.0):.2f}",
            f"  Avg Vector Qs:   {m.get('avg_vector_queries', 0.0):.2f}",
            f"  Avg Latency:     {m.get('avg_latency', 0.0):.4f}s",
            f"  Setup Latency:   {m.get('setup_latency_sec', 0.0):.4f}s",
            f"  Amortized:       {m.get('amortized_latency', 0.0):.4f}s",
            f"  Weak Retrieval:  {m.get('weak_retrieval_rate', 0.0):.4f}",
            f"  Suspicious Ctx:  {m.get('suspicious_context_rate', 0.0):.4f}",
            f"  Answer Support:  {m.get('answer_support_rate', 0.0):.4f}",
            f"  Grounding Score: {m.get('avg_grounding_score', 0.0):.4f}",
            f"  Duplicate Ctx:   {m.get('avg_duplicate_context_rate', 0.0):.4f}",
            f"  Corrective Retry:{m.get('corrective_retry_rate', 0.0):.4f}",
            f"  Compression:     {m.get('avg_context_compression_ratio', 0.0):.4f}",
            "",
        ])
    lines.append("Note: regenerate this file after every run; do not compare against stale result files.")
    path.write_text("\n".join(lines), encoding="utf-8")
