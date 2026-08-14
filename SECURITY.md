# Security Policy

## Supported Scope

The repository root is the only active implementation. The historical archive
is not supported or deployable. This project is supported as a research
prototype or private demo, not as a public multi-tenant service.

The Streamlit UI fails closed unless `A2_RAG_ACCESS_TOKEN` is configured. The
only tokenless path is the explicit `A2_RAG_LOCAL_DEMO=1` localhost/private
demo setting, which cannot be combined with public mode.

Do not expose the Streamlit app publicly unless a deployment adds external
identity, TLS, shared rate limiting, concurrency and spend controls, central
logging, secret management, tenant isolation, and an incident-response path.

## Reporting

Do not place credentials or sensitive corpus material in issues, public logs,
or generated evaluation exports. Report security concerns privately to the
maintainer responsible for the repository before public disclosure.

## Known Boundaries

- Prompt-injection pattern detection is defense in depth, not a proof that
  untrusted retrieved text is safe.
- Local document IDs are not original-source citations.
- The embedding cache is a local convenience cache and is not encrypted.
- The bundled data lacks verified source, license, and PII-review metadata.
- In-process UI quotas do not protect multiple replicas or identify users.
