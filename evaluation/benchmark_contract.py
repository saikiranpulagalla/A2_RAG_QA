"""Immutable benchmark labels kept separate from model output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTATIONS_PATH = ROOT / "data" / "benchmark_expectations.json"
QUESTIONS_PATH = ROOT / "data" / "questions" / "nq_1000.json"


def load_expected_abstention_indexes() -> frozenset[int]:
    """Return fixture-owned one-based benchmark rows expected to abstain."""
    payload = json.loads(EXPECTATIONS_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported benchmark expectations schema")
    expected_hash = str(payload.get("questions_sha256", "")).lower()
    actual_hash = hashlib.sha256(QUESTIONS_PATH.read_bytes()).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError("Benchmark expectations do not match the bundled questions dataset")
    indexes = payload.get("expected_abstention_indexes", [])
    if not isinstance(indexes, list) or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in indexes):
        raise ValueError("Benchmark expected abstention indexes must be positive integers")
    return frozenset(indexes)
