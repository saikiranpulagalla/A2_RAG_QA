from evaluation.evaluate import evaluate_rag
from utils import Document, STATIC_CORPUS_ABSTENTION


class _PolicyBlockedModel:
    documents = [Document("The canonical context.", {"doc_id": "corpus-doc-0000"})]

    def answer(self, _question, return_metadata=False):
        assert return_metadata is True
        return {
            "answer": STATIC_CORPUS_ABSTENTION,
            "policy": {"blocked": True},
            "retrieval": {"documents": []},
            "decision": {},
            "usage": {},
            "grounding": {"status": "abstained", "is_supported": True},
        }


class _GoldPassageModel:
    documents = [Document("The canonical context.", {"doc_id": "corpus-doc-0000"})]

    def answer(self, _question, return_metadata=False):
        assert return_metadata is True
        return {
            "answer": "answer",
            "policy": {"blocked": False},
            "retrieval": {
                "documents": [{"content": "The canonical context.", "metadata": {"doc_id": "corpus-doc-0000"}}],
                "quality": {"status": "good"},
            },
            "decision": {"needs_retrieval": True},
            "usage": {},
            "grounding": {"status": "supported", "is_supported": True},
        }


def test_expected_abstentions_are_not_counted_as_qa_failures():
    result = evaluate_rag(
        _PolicyBlockedModel(),
        [{"question": "What is the current rate?", "context": "The canonical context.", "answers": {"text": ["42"]}}],
        expected_abstention_indexes={1},
    )

    summary = result.summary()
    assert summary["num_examples"] == 1
    assert summary["num_qa_examples"] == 0
    assert summary["num_expected_abstentions"] == 1
    assert summary["abstention_success_rate"] == 1.0


def test_model_policy_does_not_change_default_benchmark_labels():
    result = evaluate_rag(
        _PolicyBlockedModel(),
        [{"question": "A paired benchmark question", "context": "The canonical context.", "answers": {"text": ["42"]}}],
    )

    row = result.rows[0]
    assert row["expected_abstention"] is False
    assert row["scored_qa"] is True
    assert row["evaluation_category"] == "unexpected_model_policy_abstention"
    assert result.summary()["num_qa_examples"] == 1


def test_paired_evaluation_reports_gold_passage_recall_and_rank():
    result = evaluate_rag(
        _GoldPassageModel(),
        [{"question": "Question", "context": "The canonical context.", "answers": {"text": ["answer"]}}],
    )

    row = result.rows[0]
    assert row["gold_document_ids"] == ["corpus-doc-0000"]
    assert row["recall_at_k"] == 1.0
    assert row["reciprocal_rank"] == 1.0
