# V6 Audit Remediation Report

> Superseded by `docs/V7_INDEPENDENT_AUDIT_REMEDIATION.md`. This report records
> the prior pass; its nested-copy retention and test-count statements are not
> the current release state.

## Scope

This report cross-checks the external audit against the source tree, datasets, release behavior, an isolated hash-locked environment, and current vulnerability records as of 2026-07-15. The external report was treated as evidence to verify, not as an authoritative specification.

## Confirmed And Remediated

| Finding | Evidence | Remediation |
|---|---|---|
| Release could include `.env` files and generated results | Confirmed in the old release inclusion logic | Default packaging now excludes environment variants, result outputs, secret-store/key formats, symlinks, and scans included text for high-confidence secrets. Archives are atomic and reproducible with manifest hashes. |
| Metadata could inject prompt-like instructions | Confirmed: source metadata was concatenated into chunk text | Retrieval keeps `Document` metadata structured. Source labels are single-line sanitized; role-like and instruction-like chunks are detected and quarantined before generation. |
| Embedding cache mixed vectors from incompatible models | Confirmed: old key was only text hash | Cache format v2 namespaces vectors by embedder identity, validates dimensions/finite values, ignores legacy format, uses a lock file, reload/merge, and atomic replacement. |
| Evaluation damaged abbreviations and ignored alternatives | Confirmed: `St. Louis` was truncated and 242 dataset questions have multiple references | Evaluation preserves punctuation, scores maximum EM/F1 over all references, and evaluates retrieval hit against every accepted answer. |
| Baseline/A2 timing was not comparable | Confirmed: A2 parent index was first-built inside the first answer | A2 supports `prepare()`; evaluator records construction/setup separately from query latency and reports amortized latency. Child indexes are ordinary request-local objects and are released after each answer. |
| Local embedding could silently fall back to cloud | Confirmed | Provider selection is now fail-closed. Cross-provider fallback requires `A2_RAG_ALLOW_EMBEDDING_FAILOVER=1`; the local model is pinned to an immutable revision. |
| Long documents could be transformer-truncated before parent retrieval | Confirmed: 9 documents exceed 10,000 characters; largest is 45,907 | Parent retrieval uses bounded overlapping parent chunks before embedding. |
| `limit=0` and negative limits were unsafe | Confirmed | Loaders now make `0` mean zero records, reject negatives/non-integers, and validate all JSON container shapes. |
| Streamlit dependency range included a fixed vulnerable version | Confirmed: installed 1.51.0 was affected by PYSEC-2026-212 | Requirement floor is now `streamlit>=1.53.1`; lock resolves 1.59.2. |
| Chroma/FAISS added unused dependency surface | Confirmed: Chroma used only as an in-process temporary index; FAISS path was unused | Replaced by a small NumPy cosine index; removed ChromaDB, FAISS, langchain-community, direct pandas, and matplotlib from declared runtime dependencies. |
| Locked LangChain packages had published advisories | Confirmed by a locked `pip-audit`: `langchain-core` 0.3.86 and `langchain-openai` 0.3.35 were affected; the old text-splitter line was also affected | Upgraded the lock to `langchain-core` 1.4.9, `langchain-openai` 1.3.5, `langchain-google-genai` 4.2.7, and `langchain-huggingface` 1.2.2. The final strict locked audit reports no known vulnerabilities. |
| Explicit offline mode could lose to ambient cloud credentials | Found during post-upgrade regression validation | `A2_RAG_OFFLINE=1` now takes precedence before any cloud-provider construction. A regression test sets all three cloud credentials and proves no cloud client is built. |
| Text splitting had disproportionate startup and dependency cost | Measured after the secure upgrade: importing `langchain_text_splitters` took about 101 seconds while splitting took 4 ms | Replaced it with a tested local boundary-aware splitter with deterministic overlap, hard-boundary fallback, and argument validation; removed the package from requirements and the lock. |
| Repository maintenance scripts recursively traversed protected trees | Confirmed by a 124-second audit timeout; the cleaner could also descend into `.venv` and delete dependency bytecode caches | Audit and cleanup traversal now prune the virtualenv, nested comparison copy, Git metadata, and release output top-down. A safety test proves project caches are removed while protected-tree files remain untouched. |

## Cross-Checked But Not Mischaracterized

- The Chroma tenant-authorization advisory is a serious server deployment concern, but the original code used `EphemeralClient`, not a networked Chroma server. Removing the dependency nevertheless eliminates that future misuse path for this project.
- PDF reports are stale research artifacts. They are retained as user assets, excluded from default releases, and must be regenerated/reviewed before explicit inclusion.
- `A2_RAG_QA-main` was divergent, not a byte-for-byte duplicate. It was subsequently verified, archived under `historical/`, and removed from the active tree.
- The empty `.git` directory confirms no usable local history. This is a provenance gap that code changes cannot reconstruct honestly.

## Deployment Boundary

The Streamlit app now supports an optional access token and per-session throttling, but it remains a private-demo control. It is not a substitute for production identity, shared rate limiting, TLS, tenant isolation, observability, retention policy, or incident response. Current/static and personalized high-stakes requests explicitly abstain rather than claiming live or professional authority.

## Verification Contract

`requirements.lock.txt` is a universal SHA-256 lock generated from `requirements-dev.txt` and `constraints.txt`. CI uses immutable action SHAs, least-privilege `contents: read`, installs only from that lock, runs `pip check`, lint, tests, the repository audit, a locked dependency audit, and emits a CycloneDX SBOM.

Final local verification in the isolated environment includes 53 passing tests, Ruff, scoped bytecode compilation, `pip check`, hash-lock verification, the no-network repository audit, and strict locked `pip-audit` with no known vulnerabilities. The deterministic release contains the selected source/data/documentation files plus `RELEASE_MANIFEST.txt`; an artifact-level test recomputes every listed SHA-256 digest from the archived bytes. The nested comparison copy, PDFs, results, caches, virtualenv, repository metadata, and real environment/secret files are absent by default.

The remaining residual risks are documented in `docs/SECURITY_MODEL.md`; they require a deliberately designed production service rather than another prototype patch.
