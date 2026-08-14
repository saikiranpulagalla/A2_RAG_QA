"""Remove generated local artifacts before zipping/submission."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERNS = ["__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"]
PROTECTED_DIRS = {".git", ".venv", "A2_RAG_QA-main", "dist", "historical"}


def _inside_root(path: Path, root_resolved: Path) -> bool:
    try:
        path.resolve().relative_to(root_resolved)
        return True
    except ValueError:
        return False


def clean_generated_artifacts(root: Path = ROOT) -> int:
    root_resolved = root.resolve()
    removed = 0
    for current, dirnames, filenames in os.walk(root_resolved, topdown=True):
        dirnames[:] = [name for name in dirnames if name not in PROTECTED_DIRS]
        current_path = Path(current)

        cache_dirs = [name for name in dirnames if name in PATTERNS]
        for name in cache_dirs:
            path = current_path / name
            if not _inside_root(path, root_resolved):
                continue
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
        dirnames[:] = [name for name in dirnames if name not in PATTERNS]

        for filename in filenames:
            path = current_path / filename
            if path.suffix not in {".pyc", ".pyo"} or not _inside_root(path, root_resolved):
                continue
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def main() -> int:
    removed = clean_generated_artifacts()
    print(f"Removed {removed} generated artifact paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
