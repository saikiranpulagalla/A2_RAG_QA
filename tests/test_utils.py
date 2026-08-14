import pytest

from utils import (
    EVIDENCE_ABSTENTION,
    documents_to_texts,
    enforce_evidence_grounding,
    lexical_relevance,
    normalize_document,
    normalize_documents,
    rerank_by_lexical_signal,
    static_corpus_policy,
    validate_query,
)


def test_dict_documents_extract_text_not_repr():
    docs = [{"text": "Clean document text", "title": "T"}]
    assert documents_to_texts(docs) == ["Clean document text"]
    assert "{'text'" not in documents_to_texts(docs)[0]


def test_normalize_document_preserves_metadata():
    doc = normalize_document({"text": "Alpha", "title": "Source title", "id": 1})
    assert doc.content == "Alpha"
    assert doc.metadata["title"] == "Source title"


def test_lexical_rerank_moves_relevant_doc_up():
    docs = ["bananas and apples", "Ada Lovelace wrote about the analytical engine"]
    ranked = rerank_by_lexical_signal("Who wrote about the analytical engine?", docs, weight=1.0)
    assert ranked[0].startswith("Ada Lovelace")


def test_lexical_relevance_nonzero_for_overlap():
    assert lexical_relevance("capital france", "Paris is the capital of France") > 0


def test_validate_query_trims_and_rejects_empty():
    assert validate_query("  hello  ") == "hello"
    with pytest.raises(Exception, match="non-empty"):
        validate_query("   ")


def test_validate_query_rejects_overlong_input():
    with pytest.raises(Exception, match="MAX_QUERY_CHARS"):
        validate_query("x" * 6, max_chars=5)


def test_single_document_string_is_not_split_into_characters():
    assert [doc.content for doc in normalize_documents("one complete document")] == ["one complete document"]


def test_grounding_replaces_abstention_prefix_with_canonical_abstention():
    answer, grounding = enforce_evidence_grounding(
        EVIDENCE_ABSTENTION + " Secret is 42.",
        "The supplied evidence contains no secret.",
    )

    assert answer == EVIDENCE_ABSTENTION
    assert grounding["status"] == "abstained_unsupported_evidence"
    assert grounding["original_grounding"]["is_supported"] is False


def test_static_policy_uses_intent_aware_matching():
    assert static_corpus_policy("Where was the Beauty and the Beast live action filmed?")["blocked"] is False
    assert static_corpus_policy("According to you, what is the current Bitcoin price?")["blocked"] is True
    assert static_corpus_policy("Can I mix my meds?")["blocked"] is True
    assert static_corpus_policy("Should I buy Tesla shares?")["blocked"] is True
    assert static_corpus_policy("According to the document, what was the Bitcoin price?")["blocked"] is False
