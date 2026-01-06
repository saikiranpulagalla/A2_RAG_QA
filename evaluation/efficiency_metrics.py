"""
Efficiency metrics for A²-RAG evaluation.

Tracks:
- Percentage of queries where retrieval was skipped
- Number of retrieval calls saved vs baseline
- Average context size (tokens) used
- Latency comparison
"""

import json
import logging
import csv
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class EfficencyMetrics:
    """Track efficiency metrics for RAG evaluation."""
    total_queries: int = 0
    queries_with_retrieval: int = 0
    queries_without_retrieval: int = 0
    total_context_size: int = 0  # tokens (approximate)
    total_retrieval_calls: int = 0
    avg_response_time_ms: float = 0.0
    
    @property
    def retrieval_skipped_percent(self) -> float:
        """Percentage of queries where retrieval was skipped."""
        if self.total_queries == 0:
            return 0.0
        return (self.queries_without_retrieval / self.total_queries) * 100
    
    @property
    def avg_context_size(self) -> float:
        """Average context size per query."""
        if self.total_queries == 0:
            return 0.0
        return self.total_context_size / self.total_queries
    
    @property
    def retrieval_calls_saved_vs_baseline(self) -> int:
        """Estimated retrieval calls saved vs always-retrieve baseline."""
        return self.queries_without_retrieval
    
    def summary(self) -> str:
        """Return human-readable summary."""
        return f"""
=== EFFICIENCY METRICS ===
Total Queries: {self.total_queries}
  - With Retrieval: {self.queries_with_retrieval} ({100*self.queries_with_retrieval/max(1,self.total_queries):.1f}%)
  - Without Retrieval: {self.queries_without_retrieval} ({self.retrieval_skipped_percent:.1f}%)
Retrieval Calls Saved: {self.retrieval_calls_saved_vs_baseline} ({self.retrieval_skipped_percent:.1f}% of queries)
Avg Context Size: {self.avg_context_size:.0f} tokens
Total Retrieval Calls: {self.total_retrieval_calls}
Avg Response Time: {self.avg_response_time_ms:.2f}ms
"""


@dataclass
class QueryResult:
    """Per-query evaluation result for CSV export."""
    query_id: int
    query: str
    predicted_answer: str
    reference_answer: str
    exact_match: bool
    f1_score: float
    
    # A²-RAG specific
    decision_needs_retrieval: bool = None
    decision_confidence: float = None
    decision_reasoning: str = None
    decision_source: str = None
    context_size: int = 0
    response_time_ms: float = 0.0
    
    # Baseline specific
    retrieval_hit_rate: bool = None


def save_query_results_csv(
    results: List[QueryResult],
    filename: str = "results/evaluation_details.csv"
) -> None:
    """Save per-query results to CSV for paper tables."""
    try:
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [field.name for field in results[0].__dataclass_fields__.values()]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))
        
        logger.info(f"Saved {len(results)} query results to {filename}")
    except Exception as e:
        logger.error(f"Failed to save CSV: {str(e)}")


def log_efficiency_summary(baseline_metrics: EfficencyMetrics, a2rag_metrics: EfficencyMetrics) -> None:
    """Log efficiency comparison between baseline and A²-RAG."""
    logger.info("\n" + "="*60)
    logger.info("EFFICIENCY COMPARISON")
    logger.info("="*60)
    
    logger.info("\nBASELINE RAG (Always Retrieves):")
    logger.info(baseline_metrics.summary())
    
    logger.info("\nA²-RAG (Agentic Retrieval Decision):")
    logger.info(a2rag_metrics.summary())
    
    logger.info("\nEFFICIENCY GAINS:")
    retrieval_savings = baseline_metrics.total_retrieval_calls - a2rag_metrics.total_retrieval_calls
    logger.info(f"  Retrieval Calls Saved: {retrieval_savings} ({100*retrieval_savings/max(1,baseline_metrics.total_retrieval_calls):.1f}%)")
    logger.info(f"  Context Reduction: {100*(1 - a2rag_metrics.avg_context_size/max(1,baseline_metrics.avg_context_size)):.1f}%")
    if baseline_metrics.avg_response_time_ms > 0:
        speedup = baseline_metrics.avg_response_time_ms / a2rag_metrics.avg_response_time_ms
        logger.info(f"  Speedup: {speedup:.2f}x faster")
    logger.info("="*60)
