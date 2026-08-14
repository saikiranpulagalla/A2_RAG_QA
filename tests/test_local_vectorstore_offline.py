from pathlib import Path


def _enable_offline_embeddings(monkeypatch, tmp_path):
    monkeypatch.setenv("A2_RAG_OFFLINE", "1")
    monkeypatch.setenv("A2_RAG_TEST_EMBEDDINGS", "1")
    from a2_rag.parent_child_retrieval import clear_parent_cache
    from embeddings import embedding_cache
    from embeddings.embedder import clear_embedder_cache

    embedding_cache.CACHE_FILE = Path(tmp_path) / "embeddings_cache.json"
    embedding_cache.clear_cache()
    clear_embedder_cache()
    clear_parent_cache()


def test_local_cosine_index_preserves_provenance(monkeypatch, tmp_path):
    _enable_offline_embeddings(monkeypatch, tmp_path)

    from utils import Document
    from vectorstore.local_store import build_local_store, similarity_search_local

    docs = [
        Document("Ada Lovelace wrote notes about the Analytical Engine.", {"source": "history"}),
        Document("Grace Hopper worked on compilers and COBOL.", {"source": "computing"}),
    ]
    vector_store = build_local_store(docs, index_name="test_local")
    results = similarity_search_local(
        vector_store,
        "Who wrote notes about the Analytical Engine?",
        k=1,
        fallback_empty=False,
    )
    assert results
    assert "Lovelace" in results[0].content
    assert results[0].metadata["source"] == "history"


def test_offline_end_to_end_pipelines_use_local_index(monkeypatch, tmp_path):
    _enable_offline_embeddings(monkeypatch, tmp_path)

    from a2_rag.a2_pipeline import A2RAG
    from baseline_rag.baseline_pipeline import BaselineRAG

    docs = [
        "Ada Lovelace wrote notes about the Analytical Engine.",
        "Grace Hopper worked on compilers and COBOL.",
    ]

    baseline_out = BaselineRAG(docs, k=1).answer(
        "Who wrote notes about the Analytical Engine?", return_metadata=True
    )
    assert baseline_out["retrieval"]["num_documents"] >= 1
    assert baseline_out["retrieval"]["documents"][0]["metadata"]

    a2 = A2RAG(docs)
    a2.prepare()
    a2_out = a2.answer("Who wrote notes about the Analytical Engine?", return_metadata=True)
    assert a2_out["decision"]["needs_retrieval"] is True
    assert a2_out["retrieval"]["num_documents"] >= 1
    assert a2_out["usage"]["vector_queries"] >= 1


def test_zero_query_embeddings_do_not_return_arbitrary_first_documents(monkeypatch):
    import numpy as np
    import vectorstore.local_store as local_store
    from utils import Document

    store = local_store.LocalVectorStore(
        documents=[Document("first"), Document("second")],
        vectors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    monkeypatch.setattr(local_store, "embed_with_cache", lambda *_args, **_kwargs: [[0.0, 0.0]])

    assert local_store.similarity_search_local(store, "non-lexical query", fallback_empty=False) == []
