# V7 Independent Audit Cross-Check And Remediation

## Result

The independent audit identified several material release and reliability
problems. They were reproduced against the active root before remediation;
the audit was not accepted blindly.

The active repository is now safer as a research prototype and private demo.
It is still not approved for public multi-tenant production deployment.

## Confirmed And Fixed

| Area | Cross-check result | Remediation |
|---|---|---|
| Release identity | Confirmed: an underscore-named default could target a stale archive. | The sole default output is `dist/a2-rag-release.zip`; the builder verifies exact membership, fixed timestamps, manifest, and archived bytes. |
| Divergent duplicate app | Confirmed: `A2_RAG_QA-main` was a separately runnable legacy implementation with obsolete retrieval/dependency behavior. | Its 100 files were archived byte-for-byte with a SHA-256 sidecar in `historical/`, then removed from the active tree and release/audit traversal. |
| Offline routing | Confirmed: malformed extractive routing output could skip retrieval for ambiguous queries. | Offline mode now routes ambiguous queries conservatively to retrieval without calling the extractive model as a router. |
| Grounding | Confirmed: support was checked against retrieved documents rather than the compressed evidence actually sent to generation. | Grounding now checks the exact bounded prompt evidence and abstains for empty, weak, suspicious, or unsupported evidence. |
| Limits and cache | Confirmed: non-positive retrieval limits and unbounded persistent cache/query storage were unsafe. | Retrieval validates `k`; query vectors are not persisted; cache entries are model-scoped, bounded, atomic, and lock ownership is verified before cleanup. |
| Evaluation | Confirmed: answer-substring retrieval scoring and generic report wording overstated the benchmark. | The evaluator uses stable local document IDs, paired-context gold Recall@k/MRR, separates expected abstentions, prevents CSV formulas, and labels exports as paired closed-corpus metrics. |
| Provider behavior | Confirmed: Google/OpenAI ignored an explicit caller model; evaluator retries could duplicate calls. | Providers honor model overrides; evaluator inspects the callable signature and invokes once. |
| UI controls | Confirmed: public serving could be enabled without authentication, and raw exceptions were exposed. | Public mode now requires an access token; generic errors are shown; selected mode only is initialized; process-local request/concurrency limits are applied. |
| Dependency/release hygiene | Confirmed: runtime reproducibility was mixed with developer tooling. | A separate universal hashed runtime lock, scheduled CI audit, immutable action pin, SBOM, canonical release build, and retained CI artifacts are in place. |
| Obsolete backend surface | Found during cross-check: `vectorstore/chroma_store.py` still advertised a retired backend. | The compatibility module was removed from the active runtime; the repository audit fails if it returns. |

## Audit Claim That Was Qualified

The supplied benchmark is indeed paired closed-corpus: each of the 1,000
question contexts occurs verbatim in the 1,000 document records (991 unique
passages). That concern is valid. The audit's reported reference-answer total
was not: the project loader resolves 1,220 accepted answer strings, 1,177 of
them unique, with 34 repeated texts and a maximum reuse of six. The evaluation
documentation and exports now state the legitimate paired-corpus boundary.

## Remaining Deployment Blocks

- Pattern-based prompt-injection detection is defense in depth, not a formal
  isolation boundary for hostile documents.
- The access token, rate limiter, and concurrency limiter are process-local
  demo controls. Production needs external identity, TLS, shared quotas,
  tenancy isolation, audit logs, monitoring, incident response, and spend
  controls.
- The bundled corpus still lacks verified source URLs, licenses, collection
  dates, PII review, and redistribution authorization.
- The project has no usable local Git history, signed release attestation, or
  user-selected license. Those provenance decisions cannot be fabricated by a
  code patch.
- No live-provider or model-download smoke test can prove cloud behavior in a
  no-network local audit. CI/unit coverage exercises deterministic boundaries
  only.

See `SECURITY.md`, `docs/SECURITY_MODEL.md`, and
`docs/BENCHMARK_AND_DATA_GOVERNANCE.md` for the active boundaries and release
requirements.

## Verification

The final local validation passed 72 tests, Ruff, scoped bytecode compilation,
both hashed-lock checks, the no-network repository audit, and strict
`pip-audit` checks for both development and runtime locks. The release builder
is rerun after this validation and independently verifies every archived source
file against its manifest.
