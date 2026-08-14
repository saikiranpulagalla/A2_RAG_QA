from retrieval.diversity import duplicate_context_rate, select_diverse_contexts
from retrieval.hybrid import sparse_search
from utils import answer_support_report, enforce_evidence_grounding, is_abstention
from providers.llm_factory import ExtractiveOfflineLLM, invoke_llm


def test_sparse_search_recovers_exact_rare_term():
    docs = [
        "Generic document about fruit and weather.",
        "The ZXQ-417 calibrator was introduced by Ada Labs.",
        "Another unrelated passage.",
    ]
    assert sparse_search("Who introduced ZXQ-417?", docs, k=2)[0].content.startswith("The ZXQ-417")


def test_diversity_selector_reduces_duplicate_contexts():
    docs = [
        "Ada Lovelace wrote notes about the Analytical Engine.",
        "Ada Lovelace wrote notes about the Analytical Engine.",
        "Charles Babbage designed the Analytical Engine.",
    ]
    selected = select_diverse_contexts("Analytical Engine", docs, k=2, lambda_mult=0.65)
    assert len(selected) == 2
    assert len({doc.content for doc in selected}) == 2
    assert duplicate_context_rate(docs) > duplicate_context_rate(selected)


def test_answer_support_report_supported_and_unsupported():
    supported = answer_support_report("Ada Lovelace", ["Ada Lovelace wrote the notes."])
    unsupported = answer_support_report("Charles Darwin", ["Ada Lovelace wrote the notes."])
    assert supported["is_supported"] is True
    assert unsupported["is_supported"] is False


def test_abstention_is_treated_as_safe_grounding():
    answer = "I cannot answer based on the provided context."
    report = answer_support_report(answer, [])
    assert is_abstention(answer)
    assert report["status"] == "abstained"
    assert report["is_supported"] is True


def test_grounding_enforcement_uses_exact_prompt_evidence_and_abstains():
    answer, report = enforce_evidence_grounding(
        "Charles Darwin", "[Evidence 1]\nAda Lovelace wrote the notes."
    )
    assert is_abstention(answer)
    assert report["status"] == "abstained_unsupported_evidence"


def test_offline_extractive_llm_extracts_from_context():
    prompt = """Retrieved context:\n[Evidence 1]\nAda Lovelace wrote notes for the Analytical Engine.\n\nQuestion: Who wrote notes?\n\nAnswer:"""
    assert "Ada Lovelace" in invoke_llm(ExtractiveOfflineLLM(), prompt)
