"""Minimal runnable example for A2-RAG.

Usage:
    python example_usage.py --sample-size 5 --num-docs 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from a2_rag.a2_pipeline import A2RAG
from baseline_rag.baseline_pipeline import BaselineRAG
from data import load_documents, load_questions
from evaluation.evaluate import evaluate_rag, export_comparison_csv, export_per_question_csv, export_results_to_json, generate_evaluation_summary
from evaluation.evaluator import _config_snapshot
from providers.llm_factory import offline_mode_enabled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--num-docs", type=int, default=100)
    args = parser.parse_args()

    if offline_mode_enabled():
        raise SystemExit("Offline extractive mode is demo-only and cannot produce benchmark result artifacts.")

    docs = load_documents(limit=args.num_docs)
    questions = load_questions(limit=args.sample_size)
    models = {
        "Baseline": BaselineRAG(docs),
        "A2-RAG": A2RAG(docs),
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    summaries = {}
    for name, model in models.items():
        result = evaluate_rag(model, questions)
        summaries[name] = result.summary()
        export_per_question_csv(result, name, str(results_dir / f"{name.lower().replace('-', '')}_per_question.csv"))
        export_results_to_json(result, str(results_dir / f"{name.lower().replace('-', '')}_details.json"))

    export_comparison_csv(summaries, str(results_dir / "comparison.csv"))
    (results_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (results_dir / "run_config.json").write_text(json.dumps(_config_snapshot(), indent=2), encoding="utf-8")
    generate_evaluation_summary(summaries, str(results_dir / "summary.txt"))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
