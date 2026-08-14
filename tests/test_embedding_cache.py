import json


class FakeEmbedder:
    def __init__(self, identity, vector):
        self.cache_identity = identity
        self.vector = vector
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [list(self.vector) for _ in texts]


def test_cache_is_namespaced_by_embedding_identity(monkeypatch, tmp_path):
    from embeddings import embedding_cache

    embedding_cache.CACHE_FILE = tmp_path / "cache.json"
    embedding_cache.clear_cache()
    first = FakeEmbedder("model-a", [1.0, 0.0])
    second = FakeEmbedder("model-b", [0.0, 1.0])

    monkeypatch.setattr(embedding_cache, "get_embedder", lambda: first)
    assert embedding_cache.embed_with_cache(["same text"]) == [[1.0, 0.0]]
    monkeypatch.setattr(embedding_cache, "get_embedder", lambda: second)
    assert embedding_cache.embed_with_cache(["same text"]) == [[0.0, 1.0]]
    assert first.calls == 1
    assert second.calls == 1

    payload = json.loads(embedding_cache.CACHE_FILE.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert set(payload["namespaces"]) == {"model-a", "model-b"}


def test_legacy_flat_cache_is_not_reused(monkeypatch, tmp_path):
    from embeddings import embedding_cache

    embedding_cache.CACHE_FILE = tmp_path / "cache.json"
    embedding_cache.CACHE_FILE.write_text(json.dumps({"old-text-hash": [99.0]}), encoding="utf-8")
    fake = FakeEmbedder("model-a", [1.0])
    monkeypatch.setattr(embedding_cache, "get_embedder", lambda: fake)
    embedding_cache.load_cache_from_disk()
    assert embedding_cache.embed_with_cache(["text"]) == [[1.0]]


def test_query_embeddings_can_skip_persistent_cache(monkeypatch, tmp_path):
    from embeddings import embedding_cache

    embedding_cache.CACHE_FILE = tmp_path / "cache.json"
    embedding_cache.clear_cache()
    fake = FakeEmbedder("model-a", [1.0])
    monkeypatch.setattr(embedding_cache, "get_embedder", lambda: fake)

    assert embedding_cache.embed_with_cache(["unique user query"], persist=False) == [[1.0]]
    assert not embedding_cache.CACHE_FILE.exists()


def test_persistent_cache_is_bounded_per_embedding_identity(monkeypatch, tmp_path):
    from embeddings import embedding_cache

    embedding_cache.CACHE_FILE = tmp_path / "cache.json"
    embedding_cache.clear_cache()
    monkeypatch.setattr(embedding_cache, "EMBEDDING_CACHE_MAX_VECTORS", 2)
    fake = FakeEmbedder("model-a", [1.0])
    monkeypatch.setattr(embedding_cache, "get_embedder", lambda: fake)

    for text in ("one", "two", "three"):
        embedding_cache.embed_with_cache([text])

    payload = json.loads(embedding_cache.CACHE_FILE.read_text(encoding="utf-8"))
    assert len(payload["namespaces"]["model-a"]["vectors"]) == 2
    assert fake.calls == 3


def test_cache_lock_does_not_delete_a_lock_replaced_by_another_owner(monkeypatch, tmp_path):
    from embeddings import embedding_cache

    embedding_cache.CACHE_FILE = tmp_path / "cache.json"
    lock_path = embedding_cache.CACHE_FILE.with_suffix(".json.lock")
    with embedding_cache._cache_file_lock():
        lock_path.write_text("token=replaced-owner\n", encoding="ascii")

    assert lock_path.exists()
    lock_path.unlink()


def test_cache_rejects_boolean_vector_values(monkeypatch, tmp_path):
    from embeddings import embedding_cache
    from utils import RAGException

    embedding_cache.CACHE_FILE = tmp_path / "cache.json"
    fake = FakeEmbedder("model-a", [True])
    monkeypatch.setattr(embedding_cache, "get_embedder", lambda: fake)

    try:
        embedding_cache.embed_with_cache(["text"], persist=False)
    except RAGException:
        pass
    else:
        raise AssertionError("boolean values must not be accepted as embedding dimensions")
