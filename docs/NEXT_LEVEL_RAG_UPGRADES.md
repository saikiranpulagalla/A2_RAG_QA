# Next-Level A2-RAG Upgrade Notes

> Historical upgrade notes. Chroma references describe the pre-V6 implementation; the current runtime uses the local NumPy index described in `V6_AUDIT_REMEDIATION_REPORT.md`.

This document records the second senior-engineering pass. See `V6_AUDIT_REMEDIATION_REPORT.md` for the current audit disposition, and `V3_SENIOR_UPGRADE_REPORT.md` for the historical hybrid retrieval, diversity, grounding, and reproducibility upgrades.

## Implemented in this pass

1. **Adaptive query planner**
   - Detects low/medium/high complexity queries.
   - Expands parent/child retrieval budgets for multi-hop, global, temporal, and source-bound questions.
   - Emits metadata for UI/evaluation.

2. **Deterministic multi-query retrieval**
   - Builds safe keyword/entity variants without an extra LLM call.
   - Uses Reciprocal Rank Fusion to merge ranked lists.
   - Reduces wording-mismatch failures while keeping cost predictable.

3. **Contextual chunk prefixes**
   - Adds compact `[source=...] [chunk=...]` prefixes before embedding child chunks.
   - This approximates contextual retrieval cheaply without LLM-generated chunk descriptions.

4. **Corrective quality trace**
   - Computes lexical relevance scores for retrieved chunks.
   - Labels retrieval as `good`, `weak`, `empty`, or `suspicious_context_present`.
   - Exports weak retrieval and suspicious-context rates in evaluation.

5. **Prompt-injection hygiene**
   - Flags suspicious retrieved chunks.
   - Removes obvious injection-like lines before the context is placed into the QA prompt.
   - Keeps the context boundary explicit: retrieved text is evidence, not instructions.

## Implemented in V3 after this pass

6. **Hybrid sparse + dense retrieval**
   - Adds BM25-style sparse retrieval and fuses it with dense Chroma results.
   - Improves exact rare-term/name/date recall without external search services.

7. **MMR-style context diversity**
   - Reduces duplicate chunks in final context.
   - Exports duplicate-context rate for debugging.

8. **Answer-support trace**
   - Checks whether generated answers are supported by retrieved evidence.
   - Exports support rate and grounding score without extra LLM calls.

9. **Reproducible run config**
   - Writes `results/run_config.json` with key settings and key/offline-mode status.

## Deliberately not implemented yet

1. **Full RAPTOR**
   - Needs recursive clustering and LLM summarization of clusters.
   - Worth adding only for long documents and global/multi-step questions.

2. **Full GraphRAG**
   - Needs entity extraction, graph construction, community summaries, and graph-guided retrieval.
   - Worth adding only when questions ask about global themes, networks, or relationships across a corpus.

3. **HyDE**
   - Useful for semantic-gap retrieval, but adds an LLM call per retrieval query and can hallucinate query-side assumptions.
   - Best as an optional flag for hard queries only.

4. **Cross-encoder reranking**
   - Strong next upgrade for retrieval precision.
   - Keep it optional because it adds model download/runtime cost.

5. **LLM-judge faithfulness evaluation**
   - Useful but should be behind a `--judge` flag because it adds cost and judge-model bias.

## Suggested thesis/demo framing

Call the current system:

> A heuristic-first adaptive RAG prototype with parent-child retrieval, deterministic multi-query fusion, retrieval-quality tracing, and prompt-injection-aware context assembly.

Avoid claiming:

- production-grade security,
- true token-level late chunking,
- full Self-RAG,
- full CRAG,
- GraphRAG,
- RAPTOR.
