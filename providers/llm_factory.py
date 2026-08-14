"""LLM provider factory.

All LLM construction is centralized here so the app does not mutate global
``OPENAI_API_KEY`` / ``OPENAI_API_BASE`` environment variables. OpenRouter is
OpenAI-compatible and is passed explicitly as ``base_url`` + ``api_key``.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from config import (
    ALLOW_OFFLINE_EXTRACTIVE_LLM,
    DECISION_LLM_MODEL,
    FALLBACK_LLM_MODEL,
    GEMINI_MODEL,
    LLM_CIRCUIT_BREAKER_FAILURES,
    LLM_CIRCUIT_BREAKER_RESET_SECONDS,
    LLM_TIMEOUT_SECONDS,
    MAX_RETRIES,
    MAX_TOKENS,
    OPENAI_MODEL,
    OPENROUTER_API_BASE,
    OPENROUTER_APP_TITLE,
    OPENROUTER_MODEL,
    OPENROUTER_SITE_URL,
    USE_FALLBACK_MODEL,
    USE_OPENROUTER,
)
from utils import RAGException, setup_logger

logger = setup_logger(__name__)
_circuit_lock = threading.Lock()
_circuit_state: dict[int, dict[str, float]] = {}


@dataclass(frozen=True)
class LLMInvocation:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


def _load_env_file() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception as exc:  # pragma: no cover - optional dependency/bootstrap path
        logger.debug("dotenv loading skipped: %s", exc)


_load_env_file()


class ExtractiveOfflineLLM:
    """Deterministic no-key fallback for demos and tests.

    It extracts a likely sentence/span from the provided evidence instead of
    pretending to be a capable generative LLM. Enable with A2_RAG_OFFLINE=1.
    Do not use outputs from this mode as benchmark claims.
    """

    def invoke(self, prompt: str):
        question_match = re.search(r"Question:\s*(.*?)\n\s*Answer:", prompt, flags=re.IGNORECASE | re.DOTALL)
        question = question_match.group(1).strip() if question_match else ""
        context_match = re.search(r"Retrieved context:\s*(.*?)\n\s*Question:", prompt, flags=re.IGNORECASE | re.DOTALL)
        context = context_match.group(1).strip() if context_match else ""

        class Response:
            content = ""

        if not context or "[No relevant context found]" in context:
            Response.content = "I cannot answer based on the provided context."
            return Response()

        q_terms = {t for t in re.findall(r"[a-z0-9]+", question.lower()) if len(t) > 2}
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", context) if s.strip()]
        evidence_sentences = [s for s in sentences if not s.startswith("[Evidence") and not s.startswith("[doc=") and not s.startswith("[source=")]
        ranked = sorted(
            evidence_sentences or sentences,
            key=lambda s: (-len(q_terms & set(re.findall(r"[a-z0-9]+", s.lower()))), len(s)),
        )
        Response.content = ranked[0][:300] if ranked else "I cannot answer based on the provided context."
        return Response()


def offline_mode_enabled() -> bool:
    """Return whether deterministic offline operation was explicitly requested."""
    return ALLOW_OFFLINE_EXTRACTIVE_LLM and os.getenv("A2_RAG_OFFLINE", "0") in {"1", "true", "True"}



def _make_chat_openai(**kwargs: Any):
    from langchain_openai import ChatOpenAI

    try:
        return ChatOpenAI(**kwargs)
    except TypeError:
        # Older langchain-openai used openai_api_key/openai_api_base aliases.
        if "api_key" in kwargs:
            kwargs["openai_api_key"] = kwargs.pop("api_key")
        if "base_url" in kwargs:
            kwargs["openai_api_base"] = kwargs.pop("base_url")
        return ChatOpenAI(**kwargs)


def create_llm(model: Optional[str] = None, *, purpose: str = "generation") -> Tuple[Any, str]:
    """Create the best available chat model and return ``(llm, provider_name)``."""
    errors = []

    if offline_mode_enabled():
        return ExtractiveOfflineLLM(), "offline:extractive"

    if USE_OPENROUTER and os.getenv("OPENROUTER_API_KEY"):
        selected_model = model or (DECISION_LLM_MODEL if purpose == "decision" else OPENROUTER_MODEL)
        headers = {}
        if OPENROUTER_SITE_URL:
            headers["HTTP-Referer"] = OPENROUTER_SITE_URL
        if OPENROUTER_APP_TITLE:
            headers["X-OpenRouter-Title"] = OPENROUTER_APP_TITLE
        try:
            llm = _make_chat_openai(
                model=selected_model,
                temperature=0,
                max_tokens=MAX_TOKENS,
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=OPENROUTER_API_BASE,
                default_headers=headers or None,
                timeout=LLM_TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
            )
            return llm, f"openrouter:{selected_model}"
        except Exception as exc:  # pragma: no cover - depends on external packages/keys
            errors.append(f"OpenRouter failed: {exc}")
            logger.warning(errors[-1])

    if os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            selected_model = model or GEMINI_MODEL
            llm = ChatGoogleGenerativeAI(
                model=selected_model,
                temperature=0,
                max_output_tokens=MAX_TOKENS,
                timeout=LLM_TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
            )
            return llm, f"google:{selected_model}"
        except Exception as exc:  # pragma: no cover
            errors.append(f"Gemini failed: {exc}")
            logger.warning(errors[-1])

    if USE_FALLBACK_MODEL and os.getenv("OPENAI_API_KEY"):
        try:
            selected_model = model or FALLBACK_LLM_MODEL or OPENAI_MODEL
            llm = _make_chat_openai(
                model=selected_model,
                temperature=0,
                max_tokens=MAX_TOKENS,
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=LLM_TIMEOUT_SECONDS,
                max_retries=MAX_RETRIES,
            )
            return llm, f"openai:{selected_model}"
        except Exception as exc:  # pragma: no cover
            errors.append(f"OpenAI failed: {exc}")
            logger.warning(errors[-1])

    raise RAGException(
        "No usable LLM provider found. Set OPENROUTER_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, "
        "or set A2_RAG_OFFLINE=1 for deterministic local demo mode. "
        + " | ".join(errors)
    )


def _usage_metadata(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return usage
    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        for key in ("token_usage", "usage", "usage_metadata"):
            value = response_metadata.get(key)
            if isinstance(value, dict):
                return value
    return {}


def _usage_int(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
    return 0


def invoke_llm_with_usage(llm: Any, prompt: str) -> LLMInvocation:
    """Invoke an LLM with a small in-process circuit breaker and token trace."""
    circuit_key = id(llm)
    now = time.monotonic()
    with _circuit_lock:
        state = _circuit_state.setdefault(circuit_key, {"failures": 0.0, "open_until": 0.0})
        if state["open_until"] > now:
            remaining = state["open_until"] - now
            raise RAGException(f"LLM circuit breaker is open for another {remaining:.1f}s")
        if state["open_until"]:
            state.update(failures=0.0, open_until=0.0)

    try:
        response = llm.invoke(prompt)
    except Exception:
        with _circuit_lock:
            state = _circuit_state.setdefault(circuit_key, {"failures": 0.0, "open_until": 0.0})
            state["failures"] += 1
            if state["failures"] >= LLM_CIRCUIT_BREAKER_FAILURES:
                state["open_until"] = time.monotonic() + LLM_CIRCUIT_BREAKER_RESET_SECONDS
        raise

    with _circuit_lock:
        _circuit_state[circuit_key] = {"failures": 0.0, "open_until": 0.0}
    usage = _usage_metadata(response)
    input_tokens = _usage_int(usage, "input_tokens", "prompt_tokens")
    output_tokens = _usage_int(usage, "output_tokens", "completion_tokens")
    total_tokens = _usage_int(usage, "total_tokens") or input_tokens + output_tokens
    return LLMInvocation(
        text=getattr(response, "content", str(response)).strip(),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def invoke_llm(llm: Any, prompt: str) -> str:
    """Backward-compatible text-only invocation helper."""
    return invoke_llm_with_usage(llm, prompt).text


def clear_llm_circuit_breakers() -> None:
    with _circuit_lock:
        _circuit_state.clear()
