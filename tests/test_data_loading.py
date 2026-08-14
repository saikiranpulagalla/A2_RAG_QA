import json
from pathlib import Path

import pytest

from data import get_reference_answer, get_reference_answers, load_document_objects, load_documents, load_questions


def test_load_documents_from_dict_shape(tmp_path: Path):
    path = tmp_path / "docs.json"
    path.write_text(json.dumps([{"text": "Doc one"}, {"content": "Doc two"}]), encoding="utf-8")
    assert load_documents(path) == ["Doc one", "Doc two"]


def test_load_questions_limit(tmp_path: Path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps([{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}]), encoding="utf-8")
    assert len(load_questions(path, limit=1)) == 1


def test_get_reference_answer_nq_shape():
    assert get_reference_answer({"answers": {"text": ["Paris"]}}) == "Paris"


def test_all_reference_answers_are_preserved():
    example = {"answers": {"text": ["St. Louis", "Saint Louis", "St. Louis"]}}
    assert get_reference_answers(example) == ["St. Louis", "Saint Louis"]


def test_limits_have_explicit_zero_and_negative_semantics(tmp_path: Path):
    path = tmp_path / "docs.json"
    path.write_text(json.dumps([{"text": "one"}, {"text": "two"}]), encoding="utf-8")
    assert load_documents(path, limit=0) == []
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        load_documents(path, limit=-1)
    with pytest.raises(TypeError, match="integer or None"):
        load_documents(path, limit=True)


def test_document_object_loader_rejects_non_list_json(tmp_path: Path):
    path = tmp_path / "docs.json"
    path.write_text(json.dumps({"text": "not a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a list"):
        load_document_objects(path)


def test_document_object_loader_assigns_stable_corpus_document_ids(tmp_path: Path):
    path = tmp_path / "docs.json"
    path.write_text(json.dumps([{"text": "First"}, {"text": "Second"}]), encoding="utf-8")

    docs = load_document_objects(path)

    assert [doc.metadata["doc_id"] for doc in docs] == ["corpus-doc-0000", "corpus-doc-0001"]


def test_default_dataset_paths_do_not_depend_on_the_process_working_directory(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)

    assert load_documents(limit=1)
    assert load_questions(limit=1)


def test_numeric_zero_reference_answers_are_preserved():
    assert get_reference_answers({"answers": {"text": [0]}}) == ["0"]


def test_benchmark_expectation_fixture_matches_the_bundled_questions():
    from evaluation.benchmark_contract import load_expected_abstention_indexes

    assert load_expected_abstention_indexes() == frozenset()
