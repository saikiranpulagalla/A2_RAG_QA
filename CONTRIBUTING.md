# Contributing

Keep changes scoped to the active root implementation. Do not restore or run
the historical archive as part of normal development.

Before submitting a change, update the relevant tests and run the hash-lock
verification, lint, full test suite, project audit, and release builder. When
runtime dependencies change, regenerate `requirements-runtime.lock.txt`; when
developer tooling changes, regenerate `requirements.lock.txt` as well.

Changes that alter retrieval metrics, corpus handling, source metadata, or
safety policy must also update `docs/BENCHMARK_AND_DATA_GOVERNANCE.md` or
`docs/SECURITY_MODEL.md` as appropriate.
