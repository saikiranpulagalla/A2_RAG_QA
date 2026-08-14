from a2_rag.agent_decision import _contains_any, _parse_llm_decision, clear_decision_cache, needs_retrieval


def setup_function():
    clear_decision_cache()


def test_source_bound_query_retrieves_without_llm():
    decision = needs_retrieval("According to the document, who founded the company?", use_llm=False)
    assert decision.needs_retrieval is True
    assert decision.source == "heuristic"
    assert decision.llm_calls == 0


def test_rewrite_query_skips_retrieval_without_llm():
    decision = needs_retrieval("Rewrite this sentence in a better tone", use_llm=False)
    assert decision.needs_retrieval is False
    assert decision.llm_calls == 0


def test_offline_ambiguous_query_uses_conservative_heuristic_routing(monkeypatch):
    monkeypatch.setenv("A2_RAG_OFFLINE", "1")
    clear_decision_cache()

    decision = needs_retrieval("Tell me something interesting about science.")

    assert decision.needs_retrieval is True
    assert decision.source == "heuristic_offline"
    assert decision.llm_calls == 0


def test_router_requires_an_exact_yes_or_no_decision():
    assert _parse_llm_decision("DECISION: MAYBE") is None
    assert _parse_llm_decision("DECISION: NOPE") is None
    assert _parse_llm_decision("DECISION: YES PLEASE") is None
    assert _parse_llm_decision("DECISION: YES") == (True, 0.5, "LLM decision parsed")


def test_skip_keywords_do_not_match_inside_other_words():
    assert _contains_any("Tell me about Stonehenge", ["tone"]) is False
