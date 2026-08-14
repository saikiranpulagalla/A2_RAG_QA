"""Streamlit demo for A2-RAG."""

from __future__ import annotations

import os
import hmac
import sys
import threading
import time
from collections import deque
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from a2_rag.a2_pipeline import A2RAG
from baseline_rag.baseline_pipeline import BaselineRAG
from config import MAX_QUERY_CHARS, NUM_DOCS
from ui.security import access_policy
from data import load_document_objects
from utils import normalize_document, setup_logger

logger = setup_logger(__name__)

st.set_page_config(page_title="A2-RAG QA", layout="wide")
st.title("A2-RAG: Adaptive Retrieval QA")
st.caption("Adaptive retrieval router + parent-child retrieval compared with an always-retrieve baseline.")


def _available_llm_provider() -> str | None:
    if os.getenv("OPENROUTER_API_KEY"):
        return "OpenRouter"
    if os.getenv("GOOGLE_API_KEY"):
        return "Gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "OpenAI"
    if os.getenv("A2_RAG_OFFLINE", "0") in {"1", "true", "True"}:
        return "Offline extractive demo"
    return None


provider = _available_llm_provider()
if provider:
    st.success(f"LLM provider detected: {provider}")
else:
    st.error("No LLM key found. Set OPENROUTER_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, or A2_RAG_OFFLINE=1 for local extractive demo mode.")

required_access_token = os.getenv("A2_RAG_ACCESS_TOKEN", "")
security = access_policy()
if not security.allowed:
    st.error(security.message)
    st.stop()
if security.requires_token:
    supplied_access_token = st.sidebar.text_input("Access token", type="password")
    if not hmac.compare_digest(supplied_access_token, required_access_token):
        st.info("Enter the configured access token to use this demo.")
        st.stop()
else:
    st.sidebar.warning("Explicit local demo mode. Do not expose this process beyond localhost.")

num_docs = st.sidebar.slider("Documents to index", min_value=25, max_value=1000, value=min(NUM_DOCS, 300), step=25)
mode = st.sidebar.radio("System", ["A2-RAG", "Baseline"])


@st.cache_resource(show_spinner=True, max_entries=4)
def load_model(n_docs: int, selected_mode: str):
    docs = load_document_objects(limit=n_docs)
    return A2RAG(docs) if selected_mode == "A2-RAG" else BaselineRAG(docs)


@st.cache_resource
def _shared_request_state():
    return {"timestamps": deque(), "lock": threading.Lock(), "semaphore": None, "semaphore_limit": None}


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(1, value)


def _nonnegative_float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(0.0, value)


def _allow_process_request() -> bool:
    state = _shared_request_state()
    limit = _positive_int_env("A2_RAG_MAX_REQUESTS_PER_MINUTE", 60)
    now = time.monotonic()
    with state["lock"]:
        timestamps = state["timestamps"]
        while timestamps and now - timestamps[0] >= 60.0:
            timestamps.popleft()
        if len(timestamps) >= limit:
            return False
        timestamps.append(now)
    return True


def _acquire_request_slot():
    state = _shared_request_state()
    limit = _positive_int_env("A2_RAG_MAX_CONCURRENT_REQUESTS", 2)
    with state["lock"]:
        if state["semaphore"] is None or state["semaphore_limit"] != limit:
            state["semaphore"] = threading.BoundedSemaphore(limit)
            state["semaphore_limit"] = limit
        return state["semaphore"].acquire(blocking=False), state["semaphore"]


query = st.text_area("Question", value="Who wrote Pride and Prejudice?", height=90)
if len(query) > MAX_QUERY_CHARS:
    st.warning(f"Query is longer than {MAX_QUERY_CHARS} characters; shorten it for stable evaluation.")

if st.button("Answer", disabled=not provider or not query.strip() or len(query) > MAX_QUERY_CHARS):
    try:
        now = time.monotonic()
        minimum_interval = _nonnegative_float_env("A2_RAG_MIN_REQUEST_INTERVAL_SECONDS", 1.0)
        previous_request = float(st.session_state.get("last_request_at", 0.0))
        if now - previous_request < minimum_interval:
            st.warning("Please wait briefly before sending another request.")
            st.stop()
        acquired, semaphore = _acquire_request_slot()
        if not acquired:
            st.warning("This demo is handling its maximum number of requests. Try again shortly.")
            st.stop()
        try:
            if not _allow_process_request():
                st.warning("This demo has reached its temporary request budget. Try again shortly.")
                st.stop()
            st.session_state["last_request_at"] = now
            model = load_model(num_docs, mode)
            result = model.answer(query.strip(), return_metadata=True)
        finally:
            semaphore.release()
        st.subheader("Answer")
        st.write(result["answer"])

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Retrieval used", str(result.get("decision", {}).get("needs_retrieval")))
        c2.metric("LLM calls", result.get("usage", {}).get("total_llm_calls", 0))
        c3.metric("Vector queries", result.get("usage", {}).get("vector_queries", 0))
        c4.metric("Sparse queries", result.get("usage", {}).get("sparse_queries", 0))
        c5.metric("Grounding", (result.get("grounding") or {}).get("status", "n/a"))

        with st.expander("Decision trace"):
            st.json(result.get("decision", {}))
        with st.expander("Grounding trace"):
            st.json(result.get("grounding", {}))
        with st.expander("Retrieval trace"):
            st.json({k: v for k, v in result.get("retrieval", {}).items() if k != "documents"})
            for i, doc in enumerate(result.get("retrieval", {}).get("documents", []), 1):
                normalized = normalize_document(doc)
                st.markdown(f"**Retrieved chunk {i}**")
                source = normalized.metadata.get("source") or normalized.metadata.get("title") or normalized.metadata.get("index")
                if source is not None:
                    st.caption(f"Source: {source}")
                st.write(normalized.content[:1000])
    except Exception:
        logger.exception("UI request failed")
        st.error("The request could not be completed. Check server logs for the internal error details.")
