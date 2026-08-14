# Security Model

## Supported Deployment

A2-RAG is a single-process research and evaluation application. Its supported runtime is a trusted local machine or a private development environment. It is not a multi-tenant service and must not be presented as one.

The runtime uses an in-process NumPy cosine index. ChromaDB and FAISS were removed because the project does not need a networked or persistent vector database. This also removes Chroma server authorization and remote-model-loading advisories from the dependency surface.

## Trust Boundaries

- User questions are untrusted and length-limited.
- Corpus text and metadata are untrusted. Metadata remains structured, source labels are sanitized, role-like instructions are detected, and suspicious chunks are quarantined before generation.
- Local embedding is fail-closed. A failed local model load does not send corpus text to a hosted provider unless `A2_RAG_ALLOW_EMBEDDING_FAILOVER=1` is explicitly set.
- The local embedding model is pinned to an immutable Hugging Face revision.
- Dynamic/current questions and personalized medical, legal, or financial advice are rejected because the bundled corpus is static.
- Release packaging excludes `.env` variants, generated results, credentials, private-key formats, caches, and local tooling metadata. It also scans release-bound text for high-confidence key patterns.

## Demo Access Controls

The Streamlit UI fails closed unless `A2_RAG_ACCESS_TOKEN` is set. The only
tokenless path is an explicit `A2_RAG_LOCAL_DEMO=1` setting for a trusted
localhost/private demo; it is rejected when `A2_RAG_PUBLIC=1`. The UI uses a
constant-time token comparison and applies bounded in-process request and
concurrency controls.

These controls are suitable only for a private demo. A public deployment still requires authentication at a trusted reverse proxy or identity-aware platform, centralized rate limiting, TLS, audit logging, secret management, tenant isolation, data retention rules, and abuse monitoring.

## Residual Risk

Prompt-injection detection is heuristic and cannot prove that retrieved text is safe. Deterministic grounding checks are useful guardrails but are not semantic entailment proofs. Do not process secrets, regulated personal data, or untrusted tenant corpora without a separate production security design and review.
