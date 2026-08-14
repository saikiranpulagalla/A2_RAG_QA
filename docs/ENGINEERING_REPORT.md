# Senior Engineering Implementation Report

> Historical report. It contains retired implementation claims and is excluded
> from release packages. The current audit disposition is in
> `V8_INDEPENDENT_AUDIT_REMEDIATION.md`.

## Current state after cleanup

The repository has been upgraded from a fragile demo into a cleaner research prototype. The code now has a runnable path, test coverage for the riskiest pure functions, and a consistent metadata schema across Baseline RAG and A2-RAG.

## High-impact fixes implemented

1. **Dataset normalization fixed**
   - `wiki_docs.json` records shaped like `{ "text": "..." }` are now converted to clean text.
   - The old `str(dict)` embedding bug has been removed.

2. **Provider setup fixed**
   - Added `providers/llm_factory.py`.
   - OpenRouter, Gemini, and OpenAI are initialized explicitly.
   - No global mutation of `OPENAI_API_KEY` or `OPENAI_API_BASE` is used.

3. **Baseline fairness fixed**
   - Baseline now performs true early chunking over the full corpus before indexing.
   - This makes the comparison with A2-RAG more defensible.

4. **A2-RAG metadata fixed**
   - Decision LLM calls, generation LLM calls, vector queries, and total operations are tracked separately.
   - Heuristic decisions no longer count as LLM calls.

5. **Prompt-injection boundary added**
   - Retrieved context is explicitly treated as untrusted evidence.
   - The generator is instructed not to follow instructions embedded inside retrieved text.

6. **Evaluation fixed**
   - F1 now uses token counts via `Counter`, not set overlap.
   - Baseline and A2-RAG return the same metadata schema.
   - Hit-rate extraction is stable for both systems.

7. **Stale results removed**
   - Old contradictory PNG/CSV/JSON/TXT outputs were deleted.
   - The `results/` folder now contains instructions to regenerate outputs.

8. **Runnable entry points added**
   - `data/__init__.py`
   - `example_usage.py`
   - improved `ui/app.py`
   - updated `README.md` and `QUICK_START.md`

9. **Tests added**
   - 33 tests currently pass in the top-level project.
   - Covered document normalization, metrics, routing heuristics, query planning, rank fusion, prompt-injection sanitization, metadata schemas, query validation, and real Chroma offline smoke paths.

10. **Adaptive query planning added**
   - `retrieval/query_planner.py` classifies simple, medium, and high-complexity questions.
   - A2-RAG now expands retrieval budgets for multi-hop/global/source-bound questions.
   - Deterministic query variants support multi-query retrieval without extra LLM cost.

11. **Reciprocal-rank fusion added**
   - `retrieval/fusion.py` fuses ranked lists from query variants.
   - This improves robustness when dense retrieval misses due to wording mismatch.

12. **Corrective retrieval-quality trace added**
   - Retrieval now reports lexical relevance, weak/empty status, and suspicious chunk counts.
   - Evaluation exports weak retrieval and suspicious-context rates.

13. **Retrieved-context sanitization added**
   - Obvious prompt-injection-like lines are removed from evidence blocks before prompting.
   - The raw retrieval trace still exposes quality flags for debugging.

## New architecture

```text
providers/llm_factory.py          LLM creation and invocation
embeddings/embedder.py            Embedding provider selection
data/__init__.py                  Dataset loading and answer extraction
utils.py                          Schemas, prompt templates, normalization, lexical scoring
baseline_rag/baseline_pipeline.py Early-chunk always-retrieve baseline
a2_rag/agent_decision.py          Heuristic-first adaptive router
a2_rag/parent_child_retrieval.py  Parent-child retrieval-time chunking
retrieval/query_planner.py        Adaptive retrieval planning
retrieval/fusion.py               Reciprocal-rank fusion
evaluation/evaluate.py            Evaluation and export
ui/app.py                         Streamlit interface
```

## Inspired by current RAG practice

- Self-RAG motivates adaptive retrieval instead of retrieving for every query.
- CRAG motivates lightweight retrieval-quality checking and corrective paths; this repo now has lexical relevance/status hooks for weak, empty, and suspicious retrieval.
- RAPTOR motivates hierarchical/broader-context retrieval for multi-step questions; this repo implements a lightweight parent-child version, not full recursive abstractive clustering.
- GraphRAG motivates graph-based retrieval for corpus-level/global questions; this repo now detects global questions but deliberately avoids GraphRAG until the corpus/entity structure justifies it.
- HyDE motivates query-side semantic bridging; this repo uses conservative no-LLM query variants now and leaves optional HyDE as a future API-cost tradeoff.
- RAGAS-style metrics motivate separating retrieval quality from answer quality.
- LangChain's parent-document and contextual-compression patterns support the parent-child and reranking direction.
- OWASP prompt-injection guidance motivated the explicit untrusted-context boundary.

## What is still missing

1. **True cross-encoder reranking**
   - Current reranking is dependency-free lexical reranking.
   - Best next step: add optional `BAAI/bge-reranker-base` or similar cross-encoder reranker.

2. **Faithfulness evaluation**
   - Add LLM-judged faithfulness and context precision metrics behind an optional flag.

3. **Corrective retrieval branch**
   - The quality trace exists now. Next step: if evidence is weak, trigger trusted fallback search or ask for clarification depending on product policy.

4. **Mock LLM integration tests**
   - Current schema tests use monkeypatching.
   - Add a proper fake provider for no-network CI.

5. **PDF docs not regenerated**
   - The markdown docs are corrected.
   - Existing PDFs are left untouched and may still contain old claims.

## Verification performed

```bash
python -m compileall -q .
python -m ruff check .
pytest -q
python scripts/audit_project.py
```

Result:

```text
33 passed
```
