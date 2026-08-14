from evaluation.evaluate import best_reference_metrics, exact_match, f1_score, retrieval_hit_rate, short_answer


def test_exact_match_normalizes_articles_and_punctuation():
    assert exact_match("The Paris.", "Paris")


def test_f1_uses_token_counts_not_sets():
    score = f1_score("new new york", "new york")
    assert 0 < score < 1


def test_retrieval_hit_rate_finds_answer_in_context():
    assert retrieval_hit_rate("Ada Lovelace", ["The notes were written by Ada Lovelace."])


def test_short_answer_preserves_abbreviation_periods():
    assert short_answer("St. Louis") == "St. Louis"


def test_metrics_use_best_alternative_reference():
    em, f1 = best_reference_metrics("Saint Louis", ["St. Louis", "Saint Louis"])
    assert em is True
    assert f1 == 1.0
    assert retrieval_hit_rate(["St. Louis", "Saint Louis"], ["The city is Saint Louis."])


def test_retrieval_hit_rate_uses_token_boundaries_for_short_answers():
    assert retrieval_hit_rate("US", ["This discussion is unrelated."]) is False


def test_retrieval_hit_rate_prefers_gold_document_identifiers():
    docs = [
        {"content": "The answer appears here.", "metadata": {"doc_id": "gold"}},
        {"content": "Another answer appears here.", "metadata": {"doc_id": "other"}},
    ]
    assert retrieval_hit_rate("answer", docs, gold_document_ids=["gold"]) is True
    assert retrieval_hit_rate("answer", docs[1:], gold_document_ids=["gold"]) is False


def test_call_model_does_not_retry_internal_type_errors():
    from evaluation.evaluate import _call_model

    class BrokenModel:
        calls = 0

        def answer(self, _question, return_metadata=False):
            self.calls += 1
            raise TypeError("internal model failure")

    model = BrokenModel()
    try:
        _call_model(model, "question")
    except TypeError:
        pass
    else:
        raise AssertionError("internal TypeError was swallowed")
    assert model.calls == 1


def test_csv_formula_values_are_escaped(tmp_path):
    from evaluation.evaluate import EvaluationResult, export_per_question_csv

    result = EvaluationResult()
    result.add_row({"question": " =HYPERLINK('https://example.invalid')", "prediction": "@SUM(A1:A2)"})
    output = tmp_path / "results.csv"
    export_per_question_csv(result, "=untrusted-model", str(output))

    contents = output.read_text(encoding="utf-8")
    assert "'=untrusted-model" in contents
    assert "' =HYPERLINK" in contents
    assert "'@SUM" in contents


def test_comparison_export_labels_closed_corpus_retrieval_metrics(tmp_path):
    from evaluation.evaluate import export_comparison_csv

    output = tmp_path / "comparison.csv"
    export_comparison_csv(
        {
            "model": {
                "benchmark_type": "paired_closed_corpus",
                "hit_rate": 0.5,
                "avg_reciprocal_rank": 0.25,
                "num_qa_examples": 2,
                "num_expected_abstentions": 1,
            }
        },
        str(output),
    )

    contents = output.read_text(encoding="utf-8")
    assert "Gold Recall@k" in contents
    assert "Gold MRR" in contents
    assert "Hit Rate" not in contents
