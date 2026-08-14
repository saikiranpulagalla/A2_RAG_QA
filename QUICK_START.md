# A2-RAG Quick Start

## 1. Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --require-hashes -r requirements-runtime.lock.txt
cp .env.example .env
```

Fill at least one LLM key in `.env`: `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`, or `OPENAI_API_KEY`. For a no-key local demo only, set `A2_RAG_OFFLINE=1`; do not use offline extractive mode for benchmark claims.
Local HuggingFace embeddings are used by default, so no embedding API key is required.

## 2. Run one evaluation

```bash
python example_usage.py --sample-size 5 --num-docs 100
```

Outputs are regenerated in `results/`:

```text
comparison.csv
summary.json
summary.txt
run_config.json
baseline_per_question.csv
a2rag_per_question.csv
```

## 3. Run the UI

```bash
# Safe default: configure a token before serving any requests.
set A2_RAG_ACCESS_TOKEN=replace-with-a-long-random-secret
streamlit run ui/app.py
```

For a deliberate localhost-only demo with no token, set
`A2_RAG_LOCAL_DEMO=1`. Never combine that setting with `A2_RAG_PUBLIC=1`.

## 4. Use from Python

```python
from data import load_document_objects
from a2_rag.a2_pipeline import A2RAG

_docs = load_document_objects("data/documents/wiki_docs.json", limit=100)
rag = A2RAG(_docs)
print(rag.answer("Who wrote Pride and Prejudice?", return_metadata=True))
```

## Notes

- Baseline RAG now performs true early chunking over the whole corpus.
- A2-RAG uses adaptive routing plus parent-child retrieval-time chunking.
- Do not trust old result files after code changes; regenerate them with `example_usage.py`.
- A2-RAG also records query complexity, multi-query fusion variants, hybrid sparse query counts, retrieval-quality status, grounding/support status, duplicate-context rate, and suspicious-context counts in the metadata trace.
- Run `pytest -q` before submission; the current no-network unit suite should pass without API keys.
- This is a paired, closed-corpus benchmark. Do not describe its metrics as open-domain or general QA performance.

## V4 senior engineering additions

This version adds deterministic CRAG-style corrective retry, sentence-level context compression, stronger answer grounding checks for unsupported dates/names/numbers, cached BM25-style sparse retrieval, and a GitHub Actions CI workflow. These upgrades are lightweight and reproducible; the project still does not claim full GraphRAG, full RAPTOR, or true token-level late chunking. See `docs/V4_SENIOR_UPGRADE_REPORT.md`.
