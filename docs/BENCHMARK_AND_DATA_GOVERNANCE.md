# Benchmark And Data Governance

## Evaluation Boundary

The bundled benchmark is a paired, closed-corpus retrieval evaluation. Every
question record includes a `context` field that appears verbatim in the bundled
corpus. It therefore measures whether a system can retrieve its paired evidence,
not open-domain knowledge, web freshness, or general question-answering ability.

The evaluator assigns stable local `corpus-doc-####` identifiers, derives gold
document IDs from exact paired-context matches, and reports Recall@k and MRR.
Expected static-corpus policy abstentions are reported separately and are not
counted as ordinary QA failures.

Do not publish EM, F1, Recall@k, or MRR from this dataset as claims of
open-domain, held-out, real-world, or safety performance. Meaningful external
claims require held-out queries, distractor passages, adversarial retrieval
tests, and task-specific human review.

## Data Governance Gap

The supplied corpus records contain text only. They do not contain verified
original URLs, authors, licenses, collection dates, PII review status, or a
redistribution authorization. Local document IDs provide traceability within
this folder only; they are not citations.

Until a source manifest is supplied and reviewed, treat the corpus as a
research-only bundled artifact. Do not republish it, present local IDs as
external citations, or use it for a public product.
