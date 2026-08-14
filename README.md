# A2-RAG: Adaptive and Agentic Retrieval-Augmented Generation

A2-RAG is a research prototype for comparing a standard always-retrieve RAG baseline with an adaptive RAG pipeline that decides when retrieval is needed.

## What is implemented

| Component | Baseline RAG | A2-RAG |
|---|---:|---:|
| Early corpus chunking | Yes | No |
| Always retrieves | Yes | No |
| Adaptive retrieval decision | No | Yes |
| Parent-child retrieval | No | Yes |
| Adaptive query planning / retrieval budget | No | Yes |
| Multi-query reciprocal-rank fusion | No | Yes |
| Hybrid sparse+dense retrieval | Yes | Yes |
| MMR-style context diversity | Yes | Yes |
| Retrieval-time chunking of selected parents | No | Yes |
| Structured source provenance on chunks | Yes | Yes |
| Prompt-injection detection and context sanitization | Yes | Yes |
| Retrieval-quality trace | Yes | Yes |
| Answer-support / grounding trace | Yes | Yes |
| Config snapshot for reproducibility | Yes | Yes |
| Unified metadata/cost trace | Yes | Yes |
| Local embedding default | Yes | Yes |

## Architecture

Baseline:

```text
Documents -> early chunking -> local cosine index -> hybrid sparse+dense retrieval -> RRF/MMR -> grounded answer + support trace
```

A2-RAG:

```text
Question -> routing decision
        -> query planner chooses budget + safe query variants
        -> if retrieval needed: parent hybrid multi-query retrieval -> RRF/MMR -> contextual child chunking -> child hybrid multi-query retrieval -> quality/security trace -> grounded answer + support trace
        -> if retrieval skipped: direct short answer
```

The project uses the term "retrieval-time chunking" instead of claiming true token-level late chunking. True late chunking requires contextual token embeddings before pooling; this prototype chunks after parent retrieval. Long parent documents are bounded before embedding so transformer truncation cannot hide their tails.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --require-hashes -r requirements-runtime.lock.txt
cp .env.example .env
```

Set one LLM key in `.env` for real evaluation:

```text
OPENROUTER_API_KEY=...
# or GOOGLE_API_KEY=...
# or OPENAI_API_KEY=...
```

For no-key local demos only, set `A2_RAG_OFFLINE=1`. That uses a deterministic extractive fallback and should not be used for benchmark claims. OpenRouter uses the OpenAI-compatible base URL configured in `config.py`.

## Run

```bash
python example_usage.py --sample-size 5 --num-docs 100
streamlit run ui/app.py
pytest
```

## Important files

```text
data/__init__.py                  Dataset loaders
providers/llm_factory.py          OpenRouter/Gemini/OpenAI factory
a2_rag/agent_decision.py          Adaptive retrieval router
a2_rag/parent_child_retrieval.py  Hierarchical retrieval
baseline_rag/baseline_pipeline.py Always-retrieve baseline
retrieval/query_planner.py        Adaptive budget + deterministic query variants
retrieval/fusion.py               Reciprocal-rank fusion utilities
retrieval/hybrid.py               BM25-style sparse retrieval
retrieval/diversity.py            MMR-style context diversity
evaluation/evaluate.py            EM/F1/hit-rate/cost/quality tracing
ui/app.py                         Streamlit demo
scripts/audit_project.py           No-network readiness audit
scripts/build_release.py           Clean release zip builder
scripts/verify_environment.py      Virtualenv and lockfile helper
vectorstore/local_store.py         In-process cosine vector index
```

## Evaluation metrics

- **Exact Match**: normalized exact answer equality.
- **F1**: token-level multiset F1, not set overlap.
- **Gold-passage Recall@k / MRR**: retrieval rank against document IDs derived from each paired benchmark context.
- **Retrieval Rate**: fraction of queries where A2-RAG chose retrieval.
- **Weak Retrieval Rate**: fraction of evaluated questions where retrieved evidence appears empty or weak.
- **Suspicious Context Rate**: fraction of evaluated questions where retrieved chunks contain prompt-injection-like text.
- **Answer Support Rate / Grounding Score**: deterministic check that generated answer tokens are present in retrieved evidence.
- **Duplicate Context Rate**: approximate near-duplicate chunk rate after retrieval selection.
- **LLM / vector / sparse calls**: separated to avoid misleading "API call" accounting.
- **Token use**: provider-reported input, output, and total tokens when available.
- **Latency**: model/index setup, per-query latency, and amortized latency are reported separately.

## Engineering notes

The current version fixes earlier issues:

- dict-shaped documents are normalized correctly instead of embedded as `str(dict)`;
- Baseline actually performs early chunking;
- OpenRouter is initialized without mutating global environment variables;
- evaluation metadata schema is unified across Baseline and A2-RAG;
- stale result files should be regenerated, not trusted;
- retrieved context is treated as untrusted evidence in prompts;
- source provenance stays structured through dense/sparse retrieval, fusion, diversity selection, UI traces, and evaluation; bundled records have stable local IDs but not original external provenance;
- local embeddings fail closed instead of silently sending corpus text to a cloud provider;
- the runtime uses a small in-process NumPy index rather than shipping unused Chroma/FAISS server surfaces;
- A2-RAG now has deterministic multi-query fusion, hybrid sparse+dense retrieval, MMR-style diversity, adaptive retrieval budgets, lightweight contextual chunk prefixes, retrieval quality tracing, answer-support tracing, and suspicious-context filtering.

## Future improvements

Strong next upgrades:

1. add optional cross-encoder reranking for higher retrieval precision;
2. add LLM-judged faithfulness/context precision metrics behind a `--judge` flag;
3. add an optional CRAG-style fallback branch that can call a trusted web/search connector when local evidence is weak;
4. add GraphRAG only if the corpus contains rich entity relationships and global summarization questions matter;
5. expand the existing CI with type checks and broader mock-LLM integration coverage.

See `docs/V8_INDEPENDENT_AUDIT_REMEDIATION.md` for the latest independently cross-checked remediation pass.
See `docs/BENCHMARK_AND_DATA_GOVERNANCE.md` for the paired-corpus evaluation boundary and data-governance limitations.
See `docs/RELEASE_AND_REPRODUCIBILITY.md` for release packaging and lockfile workflow.
See `docs/SECURITY_MODEL.md` for trust boundaries, supported deployment, and residual risk.

## V4 senior engineering additions

This version adds deterministic CRAG-style corrective retry, sentence-level context compression, stronger answer grounding checks for unsupported dates/names/numbers, cached BM25-style sparse retrieval, and a GitHub Actions CI workflow. These upgrades are lightweight and reproducible; the project still does not claim full GraphRAG, full RAPTOR, or true token-level late chunking. See `docs/V4_SENIOR_UPGRADE_REPORT.md`.
