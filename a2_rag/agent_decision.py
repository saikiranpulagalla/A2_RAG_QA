"""Adaptive retrieval decision module.

The decision path is deliberately heuristic-first. That avoids paying an LLM call
for obvious factual or obvious non-factual queries, then uses an LLM only for the
ambiguous middle.
"""

from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Optional, Tuple

from config import (
    FORCE_RETRIEVAL_KEYWORDS,
    HEURISTIC_DECISION_HIGH,
    HEURISTIC_DECISION_LOW,
    MAX_RETRIES,
    QUOTA_WAIT_SECONDS,
    RETRIEVAL_DECISION_CONFIDENCE_THRESHOLD,
    RETRY_BACKOFF_SECONDS,
    SKIP_RETRIEVAL_KEYWORDS,
)
from providers.llm_factory import create_llm, invoke_llm_with_usage, offline_mode_enabled
from utils import AgentDecision, RETRIEVAL_DECISION_PROMPT, setup_logger

logger = setup_logger(__name__)
_decision_llm = None


def _contains_any(query: str, keywords) -> bool:
    q = query.lower()
    return any(
        re.search(rf"(?<![a-z0-9]){re.escape(keyword.lower())}(?![a-z0-9])", q) is not None
        for keyword in keywords
    )


def _looks_source_bound(query: str) -> bool:
    q = query.lower()
    markers = ["in the document", "from the document", "according to", "in this paper", "from the report"]
    return any(m in q for m in markers)


def _looks_current_or_risky(query: str) -> bool:
    q = query.lower()
    current_terms = ["latest", "current", "today", "yesterday", "tomorrow", "recent", "price", "schedule"]
    risky_terms = ["medical", "legal", "financial", "diagnosis", "treatment", "investment", "law"]
    return any(t in q for t in current_terms + risky_terms)


def _heuristic_confidence(query: str) -> Tuple[float, str]:
    """Return confidence that retrieval is needed and a reason."""
    q = query.strip().lower()
    if not q:
        return 0.0, "Empty query"

    if _looks_source_bound(q):
        return 0.95, "Question is explicitly source/document-bound"
    if _looks_current_or_risky(q):
        return 0.92, "Question is current or high-stakes, so retrieval is safer"
    if _contains_any(q, FORCE_RETRIEVAL_KEYWORDS):
        return 0.82, "Question asks for specific factual evidence"
    if _contains_any(q, SKIP_RETRIEVAL_KEYWORDS):
        return 0.25, "Query appears to be writing/opinion/transformation-oriented"

    score = 0.48
    if re.search(r"\b\d{3,4}\b", q):
        score += 0.12
    if len(q) > 100:
        score += 0.08
    if any(word in q for word in ["compare", "difference", "rank", "best", "worst", "list"]):
        score += 0.08
    if q.startswith(("how did", "how does", "why did", "what caused")):
        score += 0.10
    if q.startswith(("rewrite", "summarize", "translate", "improve wording")):
        score -= 0.20

    score = max(0.05, min(0.95, score))
    return score, f"Heuristic retrieval confidence {score:.2f}"


def _parse_llm_decision(response: str) -> Optional[Tuple[bool, float, str]]:
    decision = None
    confidence = 0.5
    reasoning = "LLM decision parsed"
    for raw_line in (response or "").splitlines():
        line = raw_line.strip()
        if line.upper().startswith("DECISION:"):
            value = line.split(":", 1)[1].strip().upper()
            if value == "YES":
                decision = True
            elif value == "NO":
                decision = False
            else:
                return None
        elif line.upper().startswith("CONFIDENCE:"):
            match = re.search(r"\d+", line)
            if match:
                confidence = max(0.0, min(1.0, int(match.group()) / 100.0))
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()
    if decision is None:
        return None
    return decision, confidence, reasoning


def _get_decision_llm():
    global _decision_llm
    if _decision_llm is None:
        _decision_llm, _provider = create_llm(purpose="decision")
    return _decision_llm


def _llm_decision(query: str) -> Optional[Tuple[bool, float, str, int, int, int]]:
    prompt = RETRIEVAL_DECISION_PROMPT.format(query=query)
    backoff = RETRY_BACKOFF_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            invocation = invoke_llm_with_usage(_get_decision_llm(), prompt)
            parsed = _parse_llm_decision(invocation.text)
            if parsed is None:
                logger.warning("Decision LLM returned malformed routing output")
                return None
            needs, confidence, reasoning = parsed
            return needs, confidence, reasoning, invocation.input_tokens, invocation.output_tokens, invocation.total_tokens
        except Exception as exc:  # pragma: no cover - external API path
            msg = str(exc).lower()
            logger.warning("Decision LLM failed on attempt %s: %s", attempt, exc)
            if any(term in msg for term in ["rate", "quota", "timeout", "429"]):
                time.sleep(min(backoff * attempt, QUOTA_WAIT_SECONDS))
                continue
            return None
    return None


@lru_cache(maxsize=1000)
def needs_retrieval_cached(query: str, use_llm: bool = True, fallback_to_heuristic: bool = True):
    score, reason = _heuristic_confidence(query)

    if score <= HEURISTIC_DECISION_LOW:
        return False, score, reason, "heuristic", 0, 0, 0, 0
    if score >= HEURISTIC_DECISION_HIGH:
        return True, score, reason, "heuristic", 0, 0, 0, 0

    if offline_mode_enabled():
        return (
            True,
            score,
            "Offline mode routes ambiguous queries to retrieval conservatively.",
            "heuristic_offline",
            0,
            0,
            0,
            0,
        )

    if use_llm:
        llm_result = _llm_decision(query)
        if llm_result is not None:
            needs, conf, llm_reason, input_tokens, output_tokens, total_tokens = llm_result
            return needs, conf, llm_reason, "llm", 1, input_tokens, output_tokens, total_tokens

    if not fallback_to_heuristic:
        return True, 0.50, "Ambiguous and no fallback allowed; retrieving conservatively", "fallback", 0, 0, 0, 0

    needs = score >= RETRIEVAL_DECISION_CONFIDENCE_THRESHOLD
    return needs, score, reason, "heuristic", 0, 0, 0, 0


def needs_retrieval(query: str, use_llm: bool = True, fallback_to_heuristic: bool = True) -> AgentDecision:
    needs, confidence, reasoning, source, llm_calls, input_tokens, output_tokens, total_tokens = needs_retrieval_cached(
        query, use_llm, fallback_to_heuristic
    )
    return AgentDecision(
        needs_retrieval=needs,
        confidence=float(confidence),
        reasoning=reasoning,
        source=source,
        llm_calls=int(llm_calls),
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        total_tokens=int(total_tokens),
    )


def clear_decision_cache() -> None:
    global _decision_llm
    _decision_llm = None
    needs_retrieval_cached.cache_clear()
