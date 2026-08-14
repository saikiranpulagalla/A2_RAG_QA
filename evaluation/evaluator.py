"""Command helpers for running evaluations."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from a2_rag.a2_pipeline import A2RAG
from baseline_rag.baseline_pipeline import BaselineRAG
from config import EVAL_NUM_EXAMPLES, NUM_DOCS
from data import load_document_objects, load_questions
from evaluation.evaluate import (
    evaluate_rag,
    export_comparison_csv,
    export_per_question_csv,
    export_results_to_json,
    generate_evaluation_summary,
)
from utils import setup_logger
import config as project_config

logger = setup_logger(__name__)
RESULTS_DIR = Path("results")


def _config_snapshot() -> Dict[str, object]:
    keys = [
        "EMBEDDING_PROVIDER", "LOCAL_EMBEDDING_MODEL", "LOCAL_EMBEDDING_MODEL_REVISION", "OPENAI_EMBEDDING_MODEL", "GOOGLE_EMBEDDING_MODEL",
        "USE_OPENROUTER", "OPENROUTER_MODEL", "GEMINI_MODEL", "OPENAI_MODEL", "MAX_TOKENS",
        "NUM_DOCS", "TOP_K", "PARENT_K", "CHILD_K", "CHUNK_SIZE", "CHUNK_OVERLAP", "PARENT_CHUNK_SIZE", "PARENT_CHUNK_OVERLAP", "MAX_CONTEXT_CHARS",
        "ENABLE_CONTEXT_COMPRESSION", "MAX_CONTEXT_SENTENCES", "ENABLE_LEXICAL_RERANK", "ENABLE_SPARSE_RETRIEVAL", "ENABLE_MMR_DIVERSITY",
        "MMR_DIVERSITY_LAMBDA", "MIN_RETRIEVAL_RELEVANCE", "ENABLE_CORRECTIVE_RETRIEVAL", "CORRECTIVE_RETRY_MULTIPLIER", "MAX_CORRECTIVE_VARIANTS",
    ]
    return {
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "config": {key: getattr(project_config, key, None) for key in keys},
        "environment": {
            "has_openrouter_key": bool(os.getenv("OPENROUTER_API_KEY")),
            "has_google_key": bool(os.getenv("GOOGLE_API_KEY")),
            "has_openai_key": bool(os.getenv("OPENAI_API_KEY")),
            "offline_extractive_mode": os.getenv("A2_RAG_OFFLINE", "0") in {"1", "true", "True"},
        },
    }


def run_sample_evaluation(sample_size: int = EVAL_NUM_EXAMPLES, num_docs: int = NUM_DOCS) -> Dict[str, Dict[str, float]]:
    RESULTS_DIR.mkdir(exist_ok=True)
    docs = load_document_objects(limit=num_docs)
    questions = load_questions(limit=sample_size)
    models = {}
    for name, model_class in (("Baseline", BaselineRAG), ("A2-RAG", A2RAG)):
        started = time.perf_counter()
        model = model_class(docs)
        models[name] = (model, time.perf_counter() - started)

    summaries: Dict[str, Dict[str, float]] = {}
    for name, (model, construction_latency) in models.items():
        result = evaluate_rag(model, questions, initial_setup_latency_sec=construction_latency)
        summaries[name] = result.summary()
        export_per_question_csv(result, name, str(RESULTS_DIR / f"{name.lower().replace('-', '').replace(' ', '_')}_per_question.csv"))
        export_results_to_json(result, str(RESULTS_DIR / f"{name.lower().replace('-', '').replace(' ', '_')}_details.json"))

    export_comparison_csv(summaries, str(RESULTS_DIR / "comparison.csv"))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (RESULTS_DIR / "run_config.json").write_text(json.dumps(_config_snapshot(), indent=2), encoding="utf-8")
    generate_evaluation_summary(summaries, str(RESULTS_DIR / "summary.txt"))
    return summaries
