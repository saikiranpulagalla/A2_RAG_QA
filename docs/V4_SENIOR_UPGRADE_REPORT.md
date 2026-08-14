# A2-RAG V4 Senior Upgrade Report

V4 focuses on production-quality retrieval behavior rather than adding unsupported buzzwords.

## What changed

### 1. CRAG-style corrective retry

When retrieval quality is `weak` or `empty`, A2-RAG now performs a deterministic corrective retry with expanded query variants and a larger retrieval pool. The retry is accepted only if it improves or matches the original retrieval quality. This follows the practical idea behind Corrective RAG: do not blindly generate from poor retrieval.

### 2. Sentence-level context compression

Retrieved chunks are decomposed into evidence sentences, scored against the query, filtered, and recomposed into a compact numbered evidence block. This reduces prompt noise and obvious indirect prompt-injection text before generation.

### 3. Stronger grounding checks

The grounding trace now checks explicit answer fact markers such as dates, numbers, acronyms, and title-cased names. If the generated answer introduces facts absent from retrieved context, the answer is marked `weak_or_unsupported` even when generic token overlap is high.

### 4. Cached sparse retrieval

The BM25-style sparse retriever now precomputes term/document statistics once per corpus. This keeps hybrid retrieval deterministic while avoiding repeated BM25 setup during evaluation or multi-query retrieval.

### 5. CI workflow

A GitHub Actions workflow now runs Python compilation, tests, and the project audit script.

## What V4 still does not claim

- It is not full GraphRAG.
- It is not full RAPTOR.
- It is not true token-level late chunking.
- It does not use a live web fallback during evaluation.
- It does not use LLM-as-judge metrics unless added later with API keys.

## Best future V5 upgrades

1. Optional cross-encoder reranking behind a config flag.
2. LLM-judge faithfulness scoring for final reports.
3. Real external-source fallback for weak retrieval, with explicit source citations.
4. Persistent Chroma collections with full metadata filtering.
5. Dataset-specific answer normalization and confidence calibration.
