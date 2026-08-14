# Release and Reproducibility

This project keeps the root folder as the canonical source. The divergent legacy
implementation is preserved only as an excluded historical archive.

## Clean Environment

Use an isolated virtualenv before trusting `pip check` or writing a lockfile:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install --require-hashes -r requirements-runtime.lock.txt
.venv\Scripts\python scripts\verify_environment.py --verify-lock requirements.lock.txt
```

On macOS/Linux, replace `.venv\Scripts\python` with `.venv/bin/python`.

Regenerate the universal runtime hash lock only when dependency inputs intentionally change. This requires `uv`:

```bash
.venv\Scripts\python scripts\verify_environment.py --write-runtime-lock
```

`requirements-runtime.lock.txt` is resolved from `requirements.txt`; `requirements.lock.txt` is the broader CI/developer lock resolved from `requirements-dev.txt`. Neither is a `pip freeze` of the active machine. Do not validate releases from a global Python installation because unrelated packages make `pip check` unreliable.

Audit the locked dependency graph and generate a CycloneDX SBOM:

```bash
.venv\Scripts\python -m pip_audit -r requirements.lock.txt --disable-pip --strict
.venv\Scripts\python -m pip_audit -r requirements.lock.txt --disable-pip --strict --format cyclonedx-json --output dist/sbom.cdx.json
```

## Release Package

Build the default release artifact:

```bash
python scripts/build_release.py
```

The default release zip excludes:

- `A2_RAG_QA-main/`
- quarantined historical PDFs and superseded engineering reports
- `.git/`, `.agents/`, `.codex/`
- generated Python/test/lint caches
- local virtual environments
- `.env` variants, credentials, private keys, and secret stores
- generated `results/` containing prompts, answers, or retrieved evidence

The archive uses fixed ZIP metadata and a sorted file order, includes per-file SHA-256 values in `RELEASE_MANIFEST.txt`, and is written atomically. Identical source bytes produce an identical archive hash.

`--include-pdfs` applies only to future, reviewed PDFs intentionally placed in
the active root. Quarantined historical PDFs are permanently excluded:

```bash
python scripts/build_release.py --include-pdfs
```

Preview the file count without writing an artifact:

```bash
python scripts/build_release.py --dry-run
```

`scripts/clean_artifacts.py` removes generated caches only from the canonical project tree. It deliberately prunes `.venv`, `A2_RAG_QA-main`, `.git`, and `dist` so cleanup cannot mutate the isolated environment, retained comparison copy, repository metadata, or completed release artifacts.
