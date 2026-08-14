"""Small no-network repository audit for submission readiness.

Usage:
    python scripts/audit_project.py
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

# The audit imports a small contract helper. Do not let that validation create
# bytecode that causes its own strict package check to fail.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
NON_PROJECT_DIRS = {".git", ".venv", "A2_RAG_QA-main", "dist", "historical"}
GENERATED_CACHE_DIRS = {".cache", "__pycache__", ".pytest_cache", ".ruff_cache"}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _repo_files(suffixes: set[str]):
    for current, dirnames, filenames in os.walk(ROOT, topdown=True):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in NON_PROJECT_DIRS and name not in GENERATED_CACHE_DIRS
        ]
        current_path = Path(current)
        for filename in filenames:
            path = current_path / filename
            if path.suffix.lower() in suffixes:
                yield path


def _py_files():
    return list(_repo_files({".py"}))


def check_syntax() -> list[str]:
    errors = []
    for path in _py_files():
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            errors.append(f"Syntax error in {path.relative_to(ROOT)}: {exc}")
    return errors


def check_dataset_shape() -> list[str]:
    errors = []
    docs_path = ROOT / "data" / "documents" / "wiki_docs.json"
    questions_path = ROOT / "data" / "questions" / "nq_1000.json"
    for path in [docs_path, questions_path]:
        if not path.exists():
            errors.append(f"Missing dataset file: {path.relative_to(ROOT)}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list) or not data:
                errors.append(f"Expected non-empty list in {path.relative_to(ROOT)}")
        except Exception as exc:
            errors.append(f"Invalid JSON in {path.relative_to(ROOT)}: {exc}")
    return errors


def check_benchmark_contract() -> list[str]:
    try:
        from evaluation.benchmark_contract import load_expected_abstention_indexes

        load_expected_abstention_indexes()
    except Exception as exc:
        return [f"Invalid benchmark expectations fixture: {exc}"]
    return []


def check_required_files() -> list[str]:
    required = [
        "README.md",
        "QUICK_START.md",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements.lock.txt",
        "requirements-runtime.lock.txt",
        "constraints.txt",
        "pyproject.toml",
        ".env.example",
        "data/__init__.py",
        "data/benchmark_expectations.json",
        "example_usage.py",
        "providers/llm_factory.py",
        "retrieval/query_planner.py",
        "retrieval/fusion.py",
        "retrieval/hybrid.py",
        "retrieval/diversity.py",
        "retrieval/context_compressor.py",
        "retrieval/text_splitter.py",
        "evaluation/benchmark_contract.py",
        "tests",
        "tests/test_local_vectorstore_offline.py",
        "tests/test_release_packaging.py",
        "scripts/build_release.py",
        "scripts/clean_artifacts.py",
        "scripts/verify_environment.py",
        "docs/V4_SENIOR_UPGRADE_REPORT.md",
        "docs/V5_RUNTIME_HARDENING_REPORT.md",
        "docs/V6_AUDIT_REMEDIATION_REPORT.md",
        "docs/V7_INDEPENDENT_AUDIT_REMEDIATION.md",
        "docs/V8_INDEPENDENT_AUDIT_REMEDIATION.md",
        "docs/RELEASE_AND_REPRODUCIBILITY.md",
        "docs/SECURITY_MODEL.md",
        "docs/BENCHMARK_AND_DATA_GOVERNANCE.md",
        "vectorstore/local_store.py",
        ".github/workflows/ci.yml",
    ]
    return [f"Missing required path: {item}" for item in required if not (ROOT / item).exists()]


def check_no_generated_caches() -> list[str]:
    """Strict packaging check, disabled by default.

    Test/compile commands naturally create runtime caches. Set
    A2_RAG_STRICT_PACKAGE_AUDIT=1 immediately before zipping/releasing to fail
    if generated cache artifacts are present.
    """
    if os.getenv("A2_RAG_STRICT_PACKAGE_AUDIT", "0") not in {"1", "true", "True"}:
        return []
    bad = []
    for current, dirnames, _filenames in os.walk(ROOT, topdown=True):
        dirnames[:] = [name for name in dirnames if name not in NON_PROJECT_DIRS]
        current_path = Path(current)
        cache_names = [name for name in dirnames if name in GENERATED_CACHE_DIRS]
        for name in cache_names:
            path = current_path / name
            bad.append(f"Generated cache artifact should not be packaged: {path.relative_to(ROOT)}")
        dirnames[:] = [name for name in dirnames if name not in GENERATED_CACHE_DIRS]
    return bad[:20]


def check_local_vectorstore_api() -> list[str]:
    errors = []
    try:
        from vectorstore.local_store import LocalVectorStore, build_local_store, similarity_search_local

        if not callable(build_local_store) or not callable(similarity_search_local):
            errors.append("Local vector store build/search API is not callable")
        if not hasattr(LocalVectorStore, "count"):
            errors.append("LocalVectorStore must expose count()")
    except Exception as exc:
        errors.append(f"Could not validate local vector store API: {exc}")
    return errors


def check_hashed_lock() -> list[str]:
    errors = []
    for name in ("requirements.lock.txt", "requirements-runtime.lock.txt"):
        path = ROOT / name
        if not path.exists():
            errors.append(f"Missing hashed dependency lock: {name}")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "--hash=sha256:" not in text:
            errors.append(f"{name} does not contain artifact hashes")
        removed_packages = ("chromadb==", "faiss-cpu==", "langchain-text-splitters==")
        if any(package in text for package in removed_packages):
            errors.append(f"Removed retrieval dependencies remain in {name}")
    return errors


def check_release_confidentiality_policy() -> list[str]:
    try:
        from scripts.build_release import DEFAULT_OUTPUT, DIST_DIR, EXCLUDED_DIRS, EXCLUDED_FILES, SENSITIVE_NAMES, SENSITIVE_SUFFIXES
    except Exception as exc:
        return [f"Could not inspect release confidentiality policy: {exc}"]
    errors = []
    if "results" not in EXCLUDED_DIRS:
        errors.append("Release policy must exclude generated results")
    if "secrets.toml" not in SENSITIVE_NAMES:
        errors.append("Release policy must exclude common secret stores")
    if ".pem" not in SENSITIVE_SUFFIXES:
        errors.append("Release policy must exclude private key files")
    if "historical" not in EXCLUDED_DIRS:
        errors.append("Release policy must exclude historical legacy archives")
    if DEFAULT_OUTPUT.name != "a2-rag-release.zip":
        errors.append("Release policy must use the canonical a2-rag-release.zip output name")
    if DEFAULT_OUTPUT.parent != DIST_DIR or DIST_DIR.parent != ROOT:
        errors.append("Release output must be constrained to the project dist directory")
    if "docs/ENGINEERING_REPORT.md" not in EXCLUDED_FILES:
        errors.append("Release policy must exclude superseded engineering reports")
    return errors


def check_retired_runtime_paths() -> list[str]:
    retired = {
        "A2_RAG_QA-main": "Nested legacy application must remain archived, not runnable in the active tree",
        "vectorstore/chroma_store.py": "Retired Chroma compatibility module must not remain in the active runtime",
        "dist/a2_rag_qa_release.zip": "Stale underscore-named release archive must not be shipped",
    }
    return [message for relative_path, message in retired.items() if (ROOT / relative_path).exists()]


def check_docs_current() -> list[str]:
    errors = []
    stale_markers = ["19" + " passed", "24" + " passed", "See `docs/V3_SENIOR_UPGRADE_REPORT.md` for the " + "latest"]
    for path in [ROOT / "README.md", ROOT / "QUICK_START.md", ROOT / "docs" / "ENGINEERING_REPORT.md"]:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in stale_markers:
            if marker in text:
                errors.append(f"Stale documentation marker in {path.relative_to(ROOT)}: {marker}")
    return errors


def check_mojibake() -> list[str]:
    bad = []
    marker_a = "A" + "\u00c3\u201a"
    marker_b = "\u00c3\u201a" + "\u00c2\xb2"
    for path in _repo_files({".md", ".py"}):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.name == "audit_project.py":
            continue
        if marker_a in text or marker_b in text:
            bad.append(f"Mojibake marker found in {path.relative_to(ROOT)}")
    return bad


def main() -> int:
    errors = []
    errors.extend(check_required_files())
    errors.extend(check_syntax())
    errors.extend(check_dataset_shape())
    errors.extend(check_benchmark_contract())
    errors.extend(check_no_generated_caches())
    errors.extend(check_local_vectorstore_api())
    errors.extend(check_hashed_lock())
    errors.extend(check_release_confidentiality_policy())
    errors.extend(check_retired_runtime_paths())
    errors.extend(check_docs_current())
    errors.extend(check_mojibake())

    if errors:
        print("A2-RAG audit failed:")
        for err in errors:
            print(f"- {err}")
        return 1

    print("A2-RAG audit passed: required files, syntax, datasets, local vector index, hashed lock, docs, and release policy look OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
