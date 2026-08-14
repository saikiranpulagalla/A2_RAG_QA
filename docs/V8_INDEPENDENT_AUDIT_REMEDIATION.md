# V8 Independent Audit Cross-Check And Remediation

## Verdict

The external audit was substantially correct. Its critical behavior claims were
reproduced against the active tree before changes were made. This pass makes
the project safer and more honest as a private research prototype; it does not
make it suitable for public, multi-tenant deployment or general benchmark
claims.

## Reproduced And Fixed

| Finding | Cross-check | Remediation |
|---|---|---|
| Abstention-prefix grounding bypass | Confirmed: an abstention followed by `Secret is 42.` was returned as supported. | Only the exact canonical evidence abstention is accepted. Prefix-plus-content answers are replaced by the canonical abstention and traced as malformed. |
| Policy-biased evaluation | Confirmed: 22 bundled questions were blocked by substring policy and then excluded according to model output. | Policy matching is phrase/intent-aware; benchmark labels now come from a hash-bound immutable fixture, not model policy. The bundled fixture deliberately contains no expected abstentions, so unexpected refusals are scored as QA failures. |
| Dynamic/advice policy bypasses | Confirmed for `According to you`, medication mixing, and Tesla-share advice. | Source-bound language must name a supplied document/corpus/report. Medication interaction and investment-advice phrasing are now covered. |
| Provider model mismatch | Confirmed: pipelines passed an OpenRouter-formatted default to every provider. | Pipelines pass `None`; the selected provider chooses its compatible default. `USE_FALLBACK_MODEL=False` now actually disables direct OpenAI fallback. |
| Release clobbering | Confirmed: a dry run accepted `README.md` as release output. | Only a `.zip` directly in `dist/` is accepted, including dry runs. |
| Public UI opt-in auth | Confirmed: an externally bound Streamlit process could be unauthenticated if a flag was omitted. | The UI now fails closed unless an access token is set or `A2_RAG_LOCAL_DEMO=1` is deliberately set for a non-public demo. Cache entries are bounded, invalid rate config is safe, and rejected concurrent work does not consume quota. |
| Injection false positives and retrieval starvation | Confirmed: benign `passwords` text was quarantined; suspicious top results could exhaust the evidence budget. | Detection requires contextual exfiltration language, whole suspicious documents are excluded from prompt assembly, and retrieval expands its candidate pool to backfill safe results. |
| Router/parser and data edge cases | Confirmed: `MAYBE`/`NOPE` became no-retrieval; a string document became characters; `0` reference answers disappeared; default loaders depended on CWD; zero context budget became 4,500 characters. | Exact router syntax, token-bound keyword matching, atomic string-document normalization, zero-answer preservation, root-resolved defaults, and exact zero-budget behavior are covered by tests. |
| Evaluation comparability/accounting | Confirmed: model policy could alter denominators; baseline setup was eager; latency could be model-reported; retry accept rate had the wrong denominator. | Baseline preparation is deferred and measured, evaluator wall-clock latency is authoritative, retrieval metrics only average evaluable rows, grounding excludes fixture abstentions, and retry acceptance divides by retry attempts. |
| Stale release/document artifacts | Confirmed: both root PDFs carried retired configurations and unsupported/contradictory claims. | PDFs were moved to `historical/stale-reports/`, each with a SHA-256 sidecar. Superseded reports and historical artifacts are excluded from release archives. |
| CI/runtime reproducibility | Confirmed: CI only installed the developer lock. | CI now has a separate runtime-lock installation job. Lockfile headers are portable and no longer reveal local absolute paths. |

## Claims Correctly Kept As Limits

- The bundled data remains a paired closed-corpus retrieval set, not evidence
  of open-domain, fresh, safe, or production performance.
- Regex-based prompt-injection detection and token-overlap grounding remain
  guardrails, not a formal security or semantic-entailment boundary.
- The embedding cache remains local-only and not encrypted. Its path and the
  process-local UI limits are deployment configuration boundaries, not tenancy
  controls.
- The runtime lock can still resolve a large Torch/CUDA dependency graph on
  Linux. Replacing it needs a separately reviewed CPU-only packaging strategy,
  not an unpinned index shortcut.
- The project still has no usable Git history, license, signed provenance,
  verified source URLs, data rights, collection dates, or PII review. Those
  are governance decisions that code cannot manufacture.

## Release Rule

Do not publicly deploy this Streamlit process. A production service needs
external identity, TLS, shared rate/spend limits, tenancy isolation, central
logging, retention controls, incident response, and an independently reviewed
corpus/provenance policy.

## Verification

Final local validation passed 92 tests, Ruff, scoped bytecode compilation, and
both isolated hashed-lock checks. A fresh remote vulnerability lookup was not
run in this pass because the environment correctly blocked disclosure of the
private lockfile graph to an external audit service. The existing CycloneDX
SBOM remains a dependency inventory, not a fresh CVE verdict. The final strict
no-cache package audit passed. The self-verifying deterministic release and
its authoritative SHA-256 sidecar are rebuilt after this documentation update.
