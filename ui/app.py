import streamlit as st
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports like `baseline_rag` resolve when
# Streamlit runs from a different CWD.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd
import time
from dotenv import load_dotenv

load_dotenv()

from utils import setup_logger, structured_log
from config import EVAL_SAMPLE_SIZE

logger = setup_logger(__name__, verbose=True)

st.set_page_config(page_title="A²-RAG Explorer", layout="wide")

st.title("A²-RAG Explorer — Adaptive & Agentic RAG")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    model_mode = st.selectbox("Mode", ["A²-RAG", "Baseline RAG"])
    show_decision = st.checkbox("Show decision reasoning", value=True)
    show_retrieved = st.checkbox("Show retrieved documents", value=True)
    show_parent_child = st.checkbox("Show parent vs child chunks", value=True)
    load_models = st.button("Load models")
    run_eval = st.button("Run sample evaluation")

# Session state for models
if 'baseline' not in st.session_state:
    st.session_state['baseline'] = None
if 'a2rag' not in st.session_state:
    st.session_state['a2rag'] = None

# Lazy model loading
if load_models:
    st.info("Initializing models (this may take a few seconds)...")
    try:
        # Lazy import to avoid heavy library imports at Streamlit startup
        from baseline_rag.baseline_pipeline import BaselineRAG
        from a2_rag.a2_pipeline import A2RAG

        docs = ["Placeholder document: load real docs via evaluation run."]
        st.session_state['baseline'] = BaselineRAG(docs)
        st.session_state['a2rag'] = A2RAG(docs)
        st.success("Models initialized (placeholder). You can now run queries.")
    except Exception as e:
        st.error(f"Model init failed: {e}")

# Query input
st.subheader("Interactive Query")
query = st.text_input("Enter a question to ask the system:")

col1, col2 = st.columns([2, 1])
with col2:
    run_query = st.button("Run Query")

if run_query and query:
    model = st.session_state['a2rag'] if model_mode == "A²-RAG" else st.session_state['baseline']
    if model is None:
        st.warning("Models not initialized. Click 'Load models' first.")
    else:
        with st.spinner("Running..."):
            start = time.time()
            try:
                if model_mode == "A²-RAG":
                    # model is an A2RAG instance — ask it for metadata-aware answer
                    result = model.answer(query, return_metadata=True)
                    # Structured logging for decision stage
                    structured_log("Decision", {
                        "query": query,
                        "decision": result.get('decision')
                    }, logger_name=__name__)

                    st.markdown("**Decision**")
                    dec = result.get('decision', {})
                    st.write(dec if show_decision else {k: dec[k] for k in ('needs_retrieval', 'confidence') if k in dec})

                    st.markdown("**Answer**")
                    st.write(result.get('answer', ''))

                    if show_retrieved:
                        st.markdown("**Retrieval Metadata**")
                        st.write(result.get('retrieval', {}))

                else:
                    # Baseline returns context when requested
                    result = model.answer(query, return_context=True)
                    st.markdown("**Answer**")
                    st.write(result.get('answer', result))
                    if show_retrieved:
                        st.markdown("**Context**")
                        st.write(result.get('context', ''))

                elapsed = time.time() - start
                st.success(f"Query completed in {elapsed:.2f}s")

            except Exception as e:
                st.error(f"Query failed: {e}")

# Sample evaluation trigger
if run_eval:
    st.info("Running a short sample evaluation (this will call models and may incur API usage).")
    from evaluation.evaluator import run_sample_evaluation, load_results_df
    try:
        run_sample_evaluation(sample_size=EVAL_SAMPLE_SIZE)
        df = load_results_df()
        st.success("Evaluation complete — results loaded")
        st.subheader("Per-question results (first 200 rows)")
        st.dataframe(df.head(200))

        st.subheader("EM / F1 Comparison")
        summary_df = df.groupby('model')[['em','f1']].mean().reset_index()
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Exact Match (EM)**")
            st.bar_chart(summary_df.set_index('model')['em'])
        with cols[1]:
            st.markdown("**F1 Score**")
            st.bar_chart(summary_df.set_index('model')['f1'])

        st.subheader("Retrieval Skips (A²-RAG)")
        if 'needs_retrieval' in df.columns:
            pie = df[df['model']=='A²-RAG']['needs_retrieval'].value_counts(normalize=True)
            st.write(pie)

    except Exception as e:
        st.error(f"Evaluation failed: {e}")
