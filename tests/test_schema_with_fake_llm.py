from baseline_rag.baseline_pipeline import BaselineRAG
from a2_rag.a2_pipeline import A2RAG


class FakeLLM:
    def invoke(self, prompt):
        class Response:
            content = "Ada Lovelace"
        return Response()


def test_baseline_metadata_schema(monkeypatch):
    monkeypatch.setattr("baseline_rag.baseline_pipeline.create_llm", lambda *a, **k: (FakeLLM(), "fake"))
    monkeypatch.setattr("baseline_rag.baseline_pipeline.chunk_documents", lambda docs, **k: list(docs))
    fake_store = type("Store", (), {"count": lambda self: 1})()
    monkeypatch.setattr("baseline_rag.baseline_pipeline.build_local_store", lambda *a, **k: fake_store)
    monkeypatch.setattr("baseline_rag.baseline_pipeline.similarity_search_local", lambda *a, **k: ["Ada Lovelace wrote notes."])
    rag = BaselineRAG(["Ada Lovelace wrote notes."])
    out = rag.answer("Who wrote notes?", return_metadata=True)
    assert set(["answer", "decision", "retrieval", "usage", "model"]).issubset(out)
    assert out["decision"]["needs_retrieval"] is True


def test_a2_metadata_schema(monkeypatch):
    monkeypatch.setattr("a2_rag.a2_pipeline.create_llm", lambda *a, **k: (FakeLLM(), "fake"))
    monkeypatch.setattr("a2_rag.a2_pipeline.needs_retrieval", lambda *a, **k: type("D", (), {"needs_retrieval": False, "confidence": 0.2, "reasoning": "skip", "source": "heuristic", "llm_calls": 0})())
    rag = A2RAG(["Ada Lovelace wrote notes."])
    out = rag.answer("Rewrite this", return_metadata=True)
    assert set(["answer", "decision", "retrieval", "usage", "model"]).issubset(out)
    assert out["usage"]["vector_queries"] == 0
