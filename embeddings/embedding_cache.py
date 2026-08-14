"""Versioned, provider-scoped persistent embedding cache."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterator, List

from embeddings.embedder import embedding_cache_identity, get_embedder
from config import EMBEDDING_CACHE_MAX_VECTORS
from utils import RAGException, setup_logger

logger = setup_logger(__name__)
ROOT = Path(__file__).resolve().parents[1]
_configured_path = Path(os.getenv("A2_RAG_EMBED_CACHE", ".cache/embeddings_cache.json"))
CACHE_FILE = _configured_path if _configured_path.is_absolute() else ROOT / _configured_path
CACHE_VERSION = 2
LOCK_TIMEOUT_SECONDS = 10.0

_state_lock = threading.RLock()
_cache: Dict[str, Any] = {"version": CACHE_VERSION, "namespaces": {}}
_loaded_path: Path | None = None


def _empty_cache() -> Dict[str, Any]:
    return {"version": CACHE_VERSION, "namespaces": {}}


def _hash_text(text: str) -> str:
    return sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _valid_vector(value: Any, expected_dimension: int | None = None) -> bool:
    if not isinstance(value, list) or not value:
        return False
    if expected_dimension is not None and len(value) != expected_dimension:
        return False
    return all(
        isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item))
        for item in value
    )


def _read_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _empty_cache()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable embedding cache %s: %s", path, exc)
        return _empty_cache()

    # Version 1 was a flat text-hash map and had no model identity. Reusing it
    # would preserve the exact corruption this format is designed to prevent.
    if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
        logger.warning("Ignoring legacy or unsupported embedding cache format in %s", path)
        return _empty_cache()
    namespaces = payload.get("namespaces")
    if not isinstance(namespaces, dict):
        return _empty_cache()

    cleaned = _empty_cache()
    for identity, namespace in namespaces.items():
        if not isinstance(identity, str) or not isinstance(namespace, dict):
            continue
        dimension = namespace.get("dimension")
        vectors = namespace.get("vectors")
        if not isinstance(dimension, int) or dimension <= 0 or not isinstance(vectors, dict):
            continue
        valid_vectors = {
            key: [float(item) for item in vector]
            for key, vector in vectors.items()
            if isinstance(key, str) and _valid_vector(vector, dimension)
        }
        cleaned["namespaces"][identity] = {"dimension": dimension, "vectors": valid_vectors}
    return cleaned


def _ensure_loaded() -> None:
    global _cache, _loaded_path
    resolved = CACHE_FILE.resolve()
    if _loaded_path != resolved:
        _cache = _read_cache(resolved)
        _loaded_path = resolved


def load_cache_from_disk() -> None:
    global _cache, _loaded_path
    with _state_lock:
        resolved = CACHE_FILE.resolve()
        _cache = _read_cache(resolved)
        _loaded_path = resolved


@contextmanager
def _cache_file_lock() -> Iterator[None]:
    lock_path = CACHE_FILE.with_suffix(CACHE_FILE.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    fd: int | None = None
    token = uuid.uuid4().hex
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"token={token} pid={os.getpid()} created={time.time()}\n".encode("ascii"))
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for embedding cache lock: {lock_path}")
            time.sleep(0.05)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            if token in lock_path.read_text(encoding="ascii", errors="ignore"):
                lock_path.unlink()
        except FileNotFoundError:
            pass


def _merge_cache(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = _read_cache_payload(base)
    for identity, namespace in updates.get("namespaces", {}).items():
        current = merged["namespaces"].get(identity)
        if current is None or current.get("dimension") != namespace.get("dimension"):
            merged["namespaces"][identity] = {
                "dimension": namespace["dimension"],
                "vectors": dict(namespace.get("vectors", {})),
            }
        else:
            current["vectors"].update(namespace.get("vectors", {}))
    return merged


def _read_cache_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Make an isolated in-memory copy of an already validated payload."""
    return {
        "version": CACHE_VERSION,
        "namespaces": {
            identity: {"dimension": namespace["dimension"], "vectors": dict(namespace.get("vectors", {}))}
            for identity, namespace in payload.get("namespaces", {}).items()
        },
    }


def _atomic_write(payload: Dict[str, Any]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=CACHE_FILE.parent,
            prefix=CACHE_FILE.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        try:
            os.chmod(temp_path, 0o600)
        except OSError:
            pass
        os.replace(temp_path, CACHE_FILE)
        try:
            os.chmod(CACHE_FILE, 0o600)
        except OSError:
            pass
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def save_cache_to_disk() -> None:
    global _cache
    with _state_lock:
        _ensure_loaded()
        try:
            with _cache_file_lock():
                disk_cache = _read_cache(CACHE_FILE)
                _cache = _merge_cache(disk_cache, _cache)
                for namespace in _cache["namespaces"].values():
                    vectors = namespace.get("vectors", {})
                    while len(vectors) > EMBEDDING_CACHE_MAX_VECTORS:
                        vectors.pop(next(iter(vectors)))
                _atomic_write(_cache)
        except (OSError, TimeoutError) as exc:
            logger.warning("Failed to save embedding cache: %s", exc)


def embed_with_cache(texts: List[str], force_refresh: bool = False, *, persist: bool = True) -> List[List[float]]:
    if not texts:
        return []
    embedder = get_embedder()
    identity = embedding_cache_identity(embedder)

    if not persist:
        vectors = embedder.embed_documents(texts)
        if len(vectors) != len(texts):
            raise RAGException(f"Embedding backend returned {len(vectors)} vectors for {len(texts)} inputs")
        dimensions = {len(vector) for vector in vectors if isinstance(vector, list) and vector}
        if len(dimensions) != 1:
            raise RAGException("Embedding backend returned empty, invalid, or inconsistent vector dimensions")
        dimension = next(iter(dimensions))
        if not all(_valid_vector(vector, dimension) for vector in vectors):
            raise RAGException("Embedding backend returned empty, invalid, or inconsistent vector dimensions")
        return [[float(item) for item in vector] for vector in vectors]

    with _state_lock:
        _ensure_loaded()
        namespace = _cache["namespaces"].get(identity)
        expected_dimension = namespace.get("dimension") if namespace else None
        vectors = namespace.get("vectors", {}) if namespace else {}
        ordered: List[List[float] | None] = [None] * len(texts)
        misses = []
        for idx, text in enumerate(texts):
            key = _hash_text(text)
            vector = vectors.get(key)
            if not force_refresh and _valid_vector(vector, expected_dimension):
                ordered[idx] = list(vector)
            else:
                misses.append((idx, text, key))

    if misses:
        new_vectors = embedder.embed_documents([text for _, text, _ in misses])
        if len(new_vectors) != len(misses):
            raise RAGException(
                f"Embedding backend returned {len(new_vectors)} vectors for {len(misses)} cache misses"
            )
        if not all(isinstance(vector, list) and vector for vector in new_vectors):
            raise RAGException("Embedding backend returned empty, invalid, or inconsistent vector dimensions")
        dimensions = {len(vector) for vector in new_vectors}
        if len(dimensions) != 1:
            raise RAGException("Embedding backend returned empty, invalid, or inconsistent vector dimensions")
        dimension = dimensions.pop()
        if not all(_valid_vector(vector, dimension) for vector in new_vectors):
            raise RAGException("Embedding backend returned non-finite or non-numeric vector values")

        with _state_lock:
            _ensure_loaded()
            namespace = _cache["namespaces"].get(identity)
            if namespace is None or namespace.get("dimension") != dimension:
                namespace = {"dimension": dimension, "vectors": {}}
                _cache["namespaces"][identity] = namespace
            for vector, (idx, _text, key) in zip(new_vectors, misses):
                normalized_vector = [float(item) for item in vector]
                namespace["vectors"][key] = normalized_vector
                ordered[idx] = normalized_vector
            save_cache_to_disk()

    if any(vector is None for vector in ordered):
        raise RAGException("Embedding cache failed to populate every requested vector")
    return [vector for vector in ordered if vector is not None]


def clear_cache() -> None:
    global _cache, _loaded_path
    with _state_lock:
        _cache = _empty_cache()
        _loaded_path = CACHE_FILE.resolve()
        try:
            CACHE_FILE.unlink()
        except FileNotFoundError:
            pass
