"""
Agentic decision module for A²-RAG system.

Decides whether retrieval is necessary for a given query.
Uses LLM-based reasoning with heuristic confidence scoring.

Key Design:
- Explicit decision logic (yes/no)
- Confidence scores for explainability
- Fallback heuristics if LLM fails
"""

import re
import time
import os
from functools import lru_cache
from typing import Optional, Tuple, Union
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from config import (
    DECISION_LLM_MODEL,
    USE_FALLBACK_MODEL,
    RETRIEVAL_DECISION_CONFIDENCE_THRESHOLD,
    MAX_RETRIES,
    RETRY_BACKOFF_SECONDS,
    QUOTA_WAIT_SECONDS,
    HEURISTIC_DECISION_LOW,
    HEURISTIC_DECISION_HIGH,
    USE_OPENROUTER,
    OPENROUTER_MODEL,
    OPENROUTER_API_BASE,
    FORCE_RETRIEVAL_KEYWORDS,
    SKIP_RETRIEVAL_KEYWORDS,
    MAX_TOKENS,
)
from utils import setup_logger, AgentDecision, APIException, RETRIEVAL_DECISION_PROMPT

logger = setup_logger(__name__)

# Global LLM instance for caching
_llm_cache: Optional[Union[ChatOpenAI, ChatGoogleGenerativeAI]] = None


def get_decision_llm() -> Union[ChatOpenAI, ChatGoogleGenerativeAI]:
    """
    Get or create LLM instance for agentic decision.
    
    Prefers OpenRouter (if configured), falls back to Gemini candidates.
    
    Returns:
        ChatOpenAI or ChatGoogleGenerativeAI instance
    """
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache

    last_exc = None

    # Try OpenRouter first if enabled and API key present
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    openrouter_base = OPENROUTER_API_BASE or os.getenv('OPENROUTER_API_BASE')
    if USE_OPENROUTER and openrouter_key and not openrouter_base:
        # Use default OpenRouter base if only key is provided
        openrouter_base = "https://openrouter.ai/api/v1"
    
    if USE_OPENROUTER and openrouter_key and openrouter_base:
        prev_base = os.environ.get('OPENAI_API_BASE')
        prev_key = os.environ.get('OPENAI_API_KEY')
        try:
            os.environ['OPENAI_API_BASE'] = openrouter_base
            os.environ['OPENAI_API_KEY'] = openrouter_key
            logger.info(f"Initializing decision LLM with OpenRouter model: {OPENROUTER_MODEL}")
            _llm_cache = ChatOpenAI(model=OPENROUTER_MODEL, temperature=0)
            logger.info(f"Decision LLM initialized with OpenRouter: {OPENROUTER_MODEL}")
            return _llm_cache
        except Exception as e:
            last_exc = e
            logger.warning(f"Decision LLM OpenRouter init failed: {e}")
        finally:
            # Restore previous values
            if prev_base is not None:
                os.environ['OPENAI_API_BASE'] = prev_base
            else:
                os.environ.pop('OPENAI_API_BASE', None)
            if prev_key is not None:
                os.environ['OPENAI_API_KEY'] = prev_key
            else:
                os.environ.pop('OPENAI_API_KEY', None)

    # Fallback to Gemini candidates
    candidates = [DECISION_LLM_MODEL]
    # Add common Gemini fallbacks (do not assume availability)
    for alt in ("gemini-2.5-flash", "gemini-1.5-flash"):
        if alt not in candidates:
            candidates.append(alt)

    for m in candidates:
        if not m:
            continue
        try:
            logger.info(f"Initializing decision LLM with Gemini model candidate: {m}")
            _llm_cache = ChatGoogleGenerativeAI(model=m, temperature=0)
            logger.info(f"Decision LLM initialized with Gemini: {m}")
            return _llm_cache
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            logger.warning(f"Decision LLM Gemini init failed for {m}: {e}")
            # try next candidate
            continue

    # All candidates failed — raise the last exception to be handled by callers
    logger.error("All decision model candidates (OpenRouter + Gemini) failed to initialize")
    if last_exc:
        raise last_exc
    raise Exception("Could not initialize decision LLM")


def _extract_decision_from_response(response: str) -> Tuple[bool, float, str]:
    """
    Parse LLM response to extract decision, confidence, and reasoning.
    
    Expected format:
        DECISION: [YES/NO]
        CONFIDENCE: [0-100]
        REASONING: [explanation]
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Tuple of (needs_retrieval: bool, confidence: float [0, 1], reasoning: str)
    """
    try:
        lines = response.strip().split('\n')
        decision_bool = False
        confidence = 0.5
        reasoning = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith("DECISION:"):
                decision_str = line.replace("DECISION:", "").strip().upper()
                decision_bool = "YES" in decision_str
            elif line.startswith("CONFIDENCE:"):
                conf_str = line.replace("CONFIDENCE:", "").strip()
                # Extract number (0-100)
                conf_match = re.search(r'\d+', conf_str)
                if conf_match:
                    confidence = int(conf_match.group()) / 100.0
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()
        
        return decision_bool, confidence, reasoning
    except Exception as e:
        logger.warning(f"Failed to parse decision response: {str(e)}")
        return False, 0.5, "Parsing failed"


def _heuristic_decision(query: str) -> float:
    """
    Refined heuristic confidence score for retrieval decision.
    
    Strategy:
    - ALWAYS retrieve for factual, entity-based, definition, historical, medical questions
    - ONLY skip for opinion, reasoning-only, or conversational questions
    - Use keyword matching for quick decisions
    
    Returns confidence score [0, 1] that retrieval is needed:
    - 0.85+ → force retrieval (factual keyword detected)
    - 0.4–0.6 → heuristic ambiguous (use for final decision)
    - <0.4 → likely skip (opinion/reasoning-only)
    
    Args:
        query: Input query
        
    Returns:
        Confidence score [0, 1] that retrieval is needed
    """
    from config import FORCE_RETRIEVAL_KEYWORDS, SKIP_RETRIEVAL_KEYWORDS
    
    query_lower = query.lower()
    
    # ===== FORCE RETRIEVAL: Factual Keywords =====
    # If query contains factual keywords, always retrieve
    for keyword in FORCE_RETRIEVAL_KEYWORDS:
        if keyword in query_lower:
            logger.debug(f"Force-retrieval keyword '{keyword}' detected → retrieve")
            return 0.85  # High confidence for retrieval
    
    # ===== SKIP RETRIEVAL: Opinion/Reasoning Keywords =====
    # If query is opinion/reasoning-only, skip retrieval
    skip_count = sum(1 for kw in SKIP_RETRIEVAL_KEYWORDS if kw in query_lower)
    factual_count = len([kw for kw in FORCE_RETRIEVAL_KEYWORDS if kw in query_lower])
    
    if skip_count > 0 and factual_count == 0:
        logger.debug(f"Skip-retrieval keywords detected ({skip_count}) → likely skip")
        return 0.35  # Low confidence for retrieval (lean towards skip)
    
    # ===== HEURISTIC SCORING =====
    score = 0.45  # Neutral baseline (ambiguous)
    
    # Bonus for longer queries (complex questions need context)
    if len(query) > 60:
        score += 0.08
    if len(query) > 100:
        score += 0.05
    
    # Bonus for numbers/dates (temporal/factual info)
    if any(char.isdigit() for char in query):
        score += 0.10
    
    # Bonus for comparative/superlative (often need context)
    if any(word in query_lower for word in ["best", "worst", "most", "least", "compare"]):
        score += 0.08
    
    # Bonus for process/how questions (usually need context)
    if query_lower.startswith(("how do", "how does", "how can", "how to")):
        score += 0.10
    
    # Clamp to reasonable range [0.35, 0.65] for heuristic decisions
    score = max(0.35, min(0.65, score))
    logger.debug(f"Heuristic score: {score:.2f}")
    
    return score

    
    # Bonus for length (longer questions often need context)
    if len(query) > 50:
        score += 0.1
    if len(query) > 100:
        score += 0.05
    
    # Bonus for numbers/dates (factual info)
    if any(char.isdigit() for char in query):
        score += 0.1
    
    return min(score, 1.0)  # Cap at 1.0


def _is_opinion_query(query: str) -> bool:
    """Quick heuristic to detect opinion/subjective questions."""
    q = query.lower()
    opinion_markers = ["do you think", "should", "is it okay", "opinion", "what do you think", "would you", "recommend"]
    return any(marker in q for marker in opinion_markers)


def _is_short_common_fact(query: str) -> bool:
    """
    IMPROVED: Don't classify fact-based or medical questions as 'common-fact' to skip retrieval.
    
    This function now returns False for most queries (only skips for pure definition/concept Qs).
    Fact-based questions like "who is X", "where is Y", "when was Z" should ALWAYS retrieve.
    Medical/entity questions should also retrieve.
    
    Returns:
        True only for very short definition questions without medical/entity terms
    """
    from config import FORCE_RETRIEVAL_KEYWORDS
    
    q = query.strip()
    
    # Only skip very short, non-factual definition questions
    if len(q) > 80:
        return False
    
    lower = q.lower()
    
    # Skip only for pure definition/conceptual questions without wh-words
    definition_markers = ["what is", "define", "explain", "describe"]
    has_definition = any(lower.startswith(marker) for marker in definition_markers)
    
    # Check if it has any force-retrieval keywords (factual, medical, entity terms)
    has_force_keywords = any(kw in lower for kw in FORCE_RETRIEVAL_KEYWORDS)
    
    # Only return True (skip retrieval) if it's a definition AND has no force-retrieval keywords
    if has_definition and not has_force_keywords:
        return True
    
    # All other questions should retrieve
    return False



def _get_llm_decision_with_fallback(query: str) -> Optional[Tuple[bool, float, str]]:
    """
    Get LLM decision with automatic fallback to Gemini.
    
    Tries primary OpenAI LLM first, falls back to Google Gemini if it fails.
    
    Args:
        query: Input question
        
    Returns:
        Tuple of (needs_retrieval, confidence, reasoning) or None if all fail
    """
    prompt = RETRIEVAL_DECISION_PROMPT.format(query=query)
    
    llm = get_decision_llm()
    backoff = RETRY_BACKOFF_SECONDS
    for attempt in range(1, max(1, MAX_RETRIES) + 1):
        try:
            logger.debug(f"[DECISION] Using decision LLM ({DECISION_LLM_MODEL}), attempt {attempt}...")
            response = llm.invoke(prompt, max_tokens=MAX_TOKENS).content
            decision, conf, reasoning = _extract_decision_from_response(response)
            logger.debug(f"[DECISION] Decision LLM succeeded on attempt {attempt}")
            return decision, conf, reasoning
        except Exception as e:
            msg = str(e).lower()
            logger.warning(f"[DECISION] Decision LLM failed (attempt {attempt}): {str(e)}")
            # If quota exhausted, wait a bit and retry (exponential backoff)
            if "quota" in msg or "resource_exhausted" in msg or "rate limit" in msg:
                wait = backoff * (2 ** (attempt - 1))
                logger.info(f"[DECISION] Quota/rate limit detected, sleeping {wait}s before retry")
                time.sleep(min(wait, QUOTA_WAIT_SECONDS))
                continue
            # Non-quota errors: break and return None
            break

    logger.error(f"[DECISION] Decision LLM failed after {MAX_RETRIES} attempts")
    return None


# Cache decisions for repeated queries to reduce API calls
@lru_cache(maxsize=1000)
def needs_retrieval_cached(query: str, use_llm: bool = True, fallback_to_heuristic: bool = True) -> Tuple[bool, float, str, str]:
    """
    Cached decision with improved confidence calibration.
    
    Confidence ranges:
    - For SKIP decisions: return 0.35–0.65 (ambiguous, not confident)
    - For RETRIEVE decisions: return 0.7+ (confident retrieval helps)
    - Never return 1.0 for skipped retrieval (too confident)
    
    Returns: (needs_retrieval, confidence, reasoning, source)
    """
    # Heuristic-first: compute heuristic confidence
    heuristic_confidence = _heuristic_decision(query)

    # Quick deterministic rules
    if _is_opinion_query(query):
        return False, 0.40, "Opinion/subjective question - answers depend on perspective, not external facts", "heuristic"
    if _is_short_common_fact(query):
        # Only skip for pure definition questions
        return False, 0.50, "Definition/conceptual question - can be answered from model knowledge", "heuristic"

    # If heuristic is decisive, avoid LLM
    if heuristic_confidence <= HEURISTIC_DECISION_LOW:
        # Skip decision: return confidence in [0.35, 0.65] range
        return False, max(0.35, heuristic_confidence), f"Low confidence ({heuristic_confidence:.2f}) that retrieval is needed - using model knowledge", "heuristic"
    if heuristic_confidence >= HEURISTIC_DECISION_HIGH:
        # Retrieve decision: return confidence >= 0.7
        return True, max(0.7, heuristic_confidence), f"High confidence ({heuristic_confidence:.2f}) that retrieval is needed - query asks for specific facts", "heuristic"

    # Ambiguous: consult LLM if allowed
    if use_llm:
        llm_res = _get_llm_decision_with_fallback(query)
        if llm_res is not None:
            needs, conf, reasoning = llm_res
            # Calibrate LLM confidence: if retrieving, ensure >= 0.7
            if needs:
                conf = max(0.7, conf)
            else:
                conf = min(0.65, conf)  # If skipping, keep < 0.7
            return needs, conf, reasoning, "llm"
    
    # Fallback to heuristic: calibrate confidence
    needs = heuristic_confidence > RETRIEVAL_DECISION_CONFIDENCE_THRESHOLD
    if needs:
        # Retrieve: ensure confidence >= 0.7
        final_conf = max(0.7, heuristic_confidence)
        reasoning = f"Ambiguous query (confidence: {heuristic_confidence:.2f}) - performing retrieval"
    else:
        # Skip: keep confidence in [0.35, 0.65] range
        final_conf = min(0.65, heuristic_confidence)
        reasoning = f"Ambiguous query (confidence: {heuristic_confidence:.2f}) - using model knowledge"
    
    return needs, final_conf, reasoning, "heuristic"


def needs_retrieval(
    query: str,
    use_llm: bool = True,
    fallback_to_heuristic: bool = True
) -> AgentDecision:
    """
    Decide whether retrieval is needed for the given query.
    
    Strategy:
    1. Try LLM-based decision (explicit reasoning)
    2. If LLM fails, fall back to heuristic scoring
    3. Compare confidence against threshold
    
    Args:
        query: Input question
        use_llm: If True, use LLM for decision; if False, use heuristic only
        fallback_to_heuristic: If True, use heuristic as fallback if LLM fails
        
    Returns:
        AgentDecision with decision, confidence, and reasoning
        
    Example:
        >>> result = needs_retrieval("What is the capital of France?")
        >>> print(f"Needs retrieval: {result.needs_retrieval}")
        >>> print(f"Confidence: {result.confidence}")
        >>> print(f"Reasoning: {result.reasoning}")
    """
    # Compute decision (cached wrapper handles heuristics and LLM fallback)
    needs, confidence, reasoning, source = needs_retrieval_cached(
        query, use_llm=use_llm, fallback_to_heuristic=fallback_to_heuristic
    )

    # Log only final decision and confidence (RETRIEVE or SKIP)
    decision_str = "RETRIEVE" if needs else "SKIP"
    logger.info(f"[DECISION] Final decision: {decision_str} (confidence={confidence:.2f}, source={source})")

    return AgentDecision(
        needs_retrieval=needs,
        confidence=confidence,
        reasoning=reasoning,
        source=source,
    )

