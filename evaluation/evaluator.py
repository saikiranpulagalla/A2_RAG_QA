import os
import pandas as pd
from evaluation.evaluate import evaluate_rag, compare_models
from config import EVAL_NUM_EXAMPLES
from utils import setup_logger
from baseline_rag.baseline_pipeline import BaselineRAG
from a2_rag.a2_pipeline import A2RAG
import json

logger = setup_logger(__name__)

RESULTS_DIR = os.path.join(os.getcwd(), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

PER_QUERY_CSV = os.path.join(RESULTS_DIR, 'per_query_results.csv')
COMPARISON_CSV = os.path.join(RESULTS_DIR, 'comparison.csv')
SUMMARY_JSON = os.path.join(RESULTS_DIR, 'summary.json')


def run_sample_evaluation(sample_size: int = 50):
    """
    Run evaluation on a small sample and save per-question CSV and summary CSV/JSON.
    """
    # Load dataset
    import json
    with open('data/questions/nq_1000.json', 'r') as f:
        data = json.load(f)

    questions = data[:sample_size]

    # Load small wiki docs
    with open('data/documents/wiki_docs.json', 'r') as f:
        docs = json.load(f)
    docs = docs[:10]

    # Initialize models
    baseline = BaselineRAG(docs)
    a2 = A2RAG(docs)

    # Evaluate both
    models = {"Baseline": baseline, "A²-RAG": a2}

    per_query_rows = []
    comp = {}
    for name, model in models.items():
        res = evaluate_rag(model, questions)
        summary = res.summary()
        comp[name] = summary
        # Save per-question rows using metrics_metadata (which now includes latency and model metadata)
        for em, f1, hit, meta, lat, nq in zip(
            res.exact_matches, res.f1_scores, res.retrieval_hits, res.metrics_metadata, res.latencies, res.num_queries
        ):
            row = {
                'model': name,
                'question': meta.get('question'),
                'reference': meta.get('reference'),
                'prediction': meta.get('prediction'),
                'em': 1 if em else 0,
                'f1': float(f1),
                'retrieval_hit': bool(hit),
                'needs_retrieval': None if meta.get('model_metadata')=={} else meta.get('model_metadata',{}).get('decision',{}).get('needs_retrieval'),
                'latency': float(lat),
                'num_queries': int(nq)
            }
            per_query_rows.append(row)

    # Save per-query CSV
    df = pd.DataFrame(per_query_rows)
    df.to_csv(PER_QUERY_CSV, index=False)
    logger.info(f"Per-query results saved to {PER_QUERY_CSV}")

    # Save comparison CSV (use summaries we already computed)
    try:
        from evaluation.evaluate import export_comparison_csv
        export_comparison_csv(comp, COMPARISON_CSV)
        logger.info(f"Comparison CSV saved to {COMPARISON_CSV}")
    except Exception:
        # Fallback: write a simple CSV
        pd.DataFrame([{ 'Model': k, **v } for k, v in comp.items()]).to_csv(COMPARISON_CSV, index=False)

    # Save JSON summary
    with open(SUMMARY_JSON, 'w') as f:
        json.dump(comp, f, indent=2)
    logger.info(f"Summary JSON saved to {SUMMARY_JSON}")


def load_results_df():
    if os.path.exists(PER_QUERY_CSV):
        return pd.read_csv(PER_QUERY_CSV)
    return pd.DataFrame()
