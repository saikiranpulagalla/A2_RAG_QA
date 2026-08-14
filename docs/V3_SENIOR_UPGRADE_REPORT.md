# V3 Senior Upgrade Report

This pass upgrades the project from a cleaner prototype into a more defensible RAG engineering submission. The focus is practical senior-engineering value: improve retrieval recall/precision, make hallucination paths visible, and make evaluation reproducible without claiming heavyweight systems that are not fully implemented.

## Implemented

### 1. Hybrid sparse + dense retrieval

Added `retrieval/hybrid.py` with a dependency-light BM25-style sparse retriever. Both Baseline and A2-RAG now fuse dense Chroma results with sparse lexical results using Reciprocal Rank Fusion.

Why it matters:
- Dense embeddings can miss exact names, acronyms, IDs, dates, and rare terms.
- Sparse retrieval improves exact-match recall.
- RRF keeps the design deterministic and easy to explain.

### 2. MMR-style context diversity

Added `retrieval/diversity.py` with token-overlap diversity selection and duplicate-context-rate measurement.

Why it matters:
- RAG often wastes context on near-duplicate chunks.
- Diversity improves coverage for multi-hop questions.
- Duplicate-context rate is now exported as an evaluation/debug metric.

### 3. Answer-support / grounding trace

Added deterministic grounding checks in `utils.answer_support_report()` and exposed them from both pipelines.

The trace reports:
- `supported`, `weak_or_unsupported`, `abstained`, `no_context`, or `empty_answer`,
- support score,
- answer-token recall in retrieved context,
- supporting evidence indexes.

This is not a replacement for LLM-judge faithfulness, but it catches obvious hallucination paths without extra API cost.

### 4. Reproducible run config snapshot

Evaluation now writes `results/run_config.json`, containing key model/retrieval/chunking settings and whether API keys/offline mode were used.

Why it matters:
- Old results cannot be fairly compared after code/config changes.
- Reviewers can see exactly what settings produced a result file.

### 5. Offline extractive demo mode

Added `A2_RAG_OFFLINE=1` support through `ExtractiveOfflineLLM`.

Use it only for:
- UI demo without API keys,
- no-network smoke tests,
- debugging retrieval/context behavior.

Do **not** use it for benchmark claims.

### 6. Expanded tests and audit coverage

Added tests for:
- sparse retrieval rare-term recovery,
- diversity duplicate reduction,
- grounding support/unsupported cases,
- abstention behavior,
- offline extractive fallback.

The audit script now checks the new V3 modules and report.

## Still deliberately not implemented

- Full GraphRAG: requires graph extraction, community summaries, and graph-specific retrieval.
- Full RAPTOR: requires recursive clustering and abstractive summaries.
- True token-level late chunking: requires token embedding pooling from long-context encoders.
- HyDE: useful, but adds an LLM call and can introduce hallucinated query assumptions.
- Cross-encoder reranking: valuable next step, but optional because it adds runtime/model weight.
- LLM-judge faithfulness: useful behind a `--judge` flag, but not required for a no-key deterministic evaluation path.

## Best remaining V4 ideas

1. Optional cross-encoder reranking behind `ENABLE_CROSS_ENCODER_RERANK`.
2. Optional LLM judge evaluation behind `--judge`.
3. CRAG-style corrective fallback when retrieval quality is `weak` or `empty`.
4. CI workflow with `pytest`, audit script, and a no-key offline smoke run.
5. Better source metadata preservation in Chroma instead of source prefixes only.
