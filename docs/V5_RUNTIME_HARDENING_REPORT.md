# V5 Runtime Hardening Report

> Superseded by `docs/V7_INDEPENDENT_AUDIT_REMEDIATION.md`. The Chroma-specific
> work below is historical: Chroma and its compatibility module were removed
> from the active runtime in the later remediation pass.

This pass addressed the runtime, dependency, and packaging findings from the deeper audit.

## Fixed

1. Chroma current-version compatibility
   - `CachedEmbeddingFunction` now exposes `__call__`, `embed_documents`, `embed_query`, `name`, `get_config`, `build_from_config`, `is_legacy`, `default_space`, `supported_spaces`, and config validation helpers.
   - This fixes Chroma 1.3 query-time failures where Chroma calls `embed_query(input=...)`.

2. Chroma collection collisions
   - `build_chroma` now combines the logical collection name with a corpus fingerprint.
   - Baseline and A2-RAG no longer delete or overwrite each other's collections inside a long-lived process.

3. Real Chroma smoke tests
   - Added `tests/test_chroma_integration_offline.py`.
   - These tests use real Chroma when installed and skip only when Chroma is missing.

4. Offline no-network embeddings
   - Added deterministic hash embeddings for `A2_RAG_OFFLINE=1` / `A2_RAG_TEST_EMBEDDINGS=1`.
   - Offline tests no longer require sentence-transformers downloads or API keys.

5. Core query validation
   - Added `validate_query` in `utils.py`.
   - Baseline and A2-RAG now reject empty and overlong queries outside the Streamlit UI path too.

6. Provider configuration hygiene
   - The LLM factory now loads `.env` through `python-dotenv` when available.
   - `purpose="decision"` now uses `DECISION_LLM_MODEL` for OpenRouter unless a caller explicitly passes a model.

7. Dependency hygiene
   - Added bounded dependency ranges in `requirements.txt`.
   - Added `constraints.txt`.
   - CI installs with `-c constraints.txt`.

8. CI and test discovery hardening
   - Added `pyproject.toml` with pytest settings that keep the nested project copy out of top-level collection.
   - CI now runs Python 3.11 and 3.12, `pip check`, compile, ruff undefined-name checks, tests, and the project audit.

9. Artifact cleanup
   - Added `scripts/clean_artifacts.py`.
   - `.ruff_cache/` is now ignored alongside `.pytest_cache/`.

## Verification

Run:

```bash
python -m compileall -q .
ruff check .
pytest -q
python scripts/audit_project.py
```

## Remaining future work

- Optional cross-encoder reranking.
- Optional LLM-judge faithfulness scoring.
- Preserve richer source metadata in Chroma for citation-heavy workflows.
- Regenerate PDFs only when they are intended to ship; the default release zip excludes them.
