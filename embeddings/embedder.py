"""Embedding provider factory.

Default path is local sentence-transformers to keep the repository runnable
without API quota. API providers remain available when explicitly configured.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import List, Optional

from config import (
    ALLOW_EMBEDDING_PROVIDER_FALLBACK,
    EMBEDDING_PROVIDER,
    GOOGLE_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_MODEL_REVISION,
    OPENAI_EMBEDDING_MODEL,
)
from utils import APIException, setup_logger

logger = setup_logger(__name__)
_embedder_cache = None


class DeterministicHashEmbeddings:
    """Tiny dependency-free embedding backend for no-network tests/demos.

    It is not a semantic model. It only provides stable vectors so the vector
    path can be integration-tested without model downloads or API keys.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.cache_identity = f"deterministic-hash:v1:dim={dim}"

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


def _offline_embeddings_enabled() -> bool:
    return os.getenv("A2_RAG_OFFLINE", "0") in {"1", "true", "True"} or os.getenv("A2_RAG_TEST_EMBEDDINGS", "0") in {"1", "true", "True"}


def embedding_cache_identity(embedder: object) -> str:
    """Return a stable identity for vectors produced by an embedding backend."""
    explicit = getattr(embedder, "cache_identity", None)
    if explicit:
        return str(explicit)
    details = {
        "class": f"{embedder.__class__.__module__}.{embedder.__class__.__qualname__}",
        "model": getattr(embedder, "model", None) or getattr(embedder, "model_name", None),
        "dimensions": getattr(embedder, "dimensions", None) or getattr(embedder, "dim", None),
        "revision": LOCAL_EMBEDDING_MODEL_REVISION if "HuggingFace" in embedder.__class__.__name__ else None,
        "normalized": getattr(embedder, "encode_kwargs", {}).get("normalize_embeddings")
        if isinstance(getattr(embedder, "encode_kwargs", {}), dict)
        else None,
    }
    encoded = json.dumps(details, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _provider_failover_enabled() -> bool:
    env_value = os.getenv("A2_RAG_ALLOW_EMBEDDING_FAILOVER")
    if env_value is not None:
        return env_value in {"1", "true", "True"}
    return bool(ALLOW_EMBEDDING_PROVIDER_FALLBACK)


def get_embedder(force_new: bool = False, provider: Optional[str] = None):
    global _embedder_cache
    if _embedder_cache is not None and not force_new and provider is None:
        return _embedder_cache

    if provider is None and _offline_embeddings_enabled():
        embedder = DeterministicHashEmbeddings()
        _embedder_cache = embedder
        logger.info("Embedding provider initialized: deterministic-hash")
        return embedder

    selected = (provider or EMBEDDING_PROVIDER or "local").lower()
    if selected not in {"local", "openai", "google"}:
        raise APIException(f"Unknown embedding provider: {selected}")
    attempts = []

    fallback_order = {
        "local": ["openai", "google"],
        "openai": ["local", "google"],
        "google": ["local", "openai"],
    }
    provider_order = [selected]
    if _provider_failover_enabled():
        provider_order.extend(fallback_order[selected])

    for candidate in provider_order:
        try:
            if candidate == "local":
                from langchain_huggingface import HuggingFaceEmbeddings

                embedder = HuggingFaceEmbeddings(
                    model_name=LOCAL_EMBEDDING_MODEL,
                    model_kwargs={"device": "cpu", "revision": LOCAL_EMBEDDING_MODEL_REVISION},
                    encode_kwargs={"normalize_embeddings": True},
                )
            elif candidate == "openai":
                if not os.getenv("OPENAI_API_KEY"):
                    raise RuntimeError("OPENAI_API_KEY is not set")
                from langchain_openai import OpenAIEmbeddings

                embedder = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
            else:
                if not os.getenv("GOOGLE_API_KEY"):
                    raise RuntimeError("GOOGLE_API_KEY is not set")
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                embedder = GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)

            if provider is None:
                _embedder_cache = embedder
            logger.info("Embedding provider initialized: %s", candidate)
            return embedder
        except Exception as exc:  # pragma: no cover - depends on optional deps/keys
            attempts.append(f"{candidate}: {exc}")
            logger.warning("Embedding provider failed: %s", attempts[-1])

    failover_hint = " Set A2_RAG_ALLOW_EMBEDDING_FAILOVER=1 to permit explicit cross-provider fallback." if not _provider_failover_enabled() else ""
    raise APIException(f"Configured embedding provider '{selected}' is unavailable. Attempts: " + " | ".join(attempts) + failover_hint)


def clear_embedder_cache() -> None:
    global _embedder_cache
    _embedder_cache = None
