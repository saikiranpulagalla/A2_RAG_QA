import builtins
import sys
import types

import pytest


def test_explicit_offline_mode_ignores_ambient_cloud_credentials(monkeypatch):
    from providers import llm_factory

    monkeypatch.setenv("A2_RAG_OFFLINE", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ambient-openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "ambient-google-key")

    def fail_if_cloud_client_is_built(**_kwargs):
        raise AssertionError("offline mode attempted to construct a cloud client")

    monkeypatch.setattr(llm_factory, "_make_chat_openai", fail_if_cloud_client_is_built)
    llm, provider = llm_factory.create_llm()

    assert isinstance(llm, llm_factory.ExtractiveOfflineLLM)
    assert provider == "offline:extractive"


def test_local_embedding_failure_does_not_silently_call_cloud(monkeypatch):
    from embeddings import embedder

    embedder.clear_embedder_cache()
    monkeypatch.delenv("A2_RAG_OFFLINE", raising=False)
    monkeypatch.delenv("A2_RAG_TEST_EMBEDDINGS", raising=False)
    monkeypatch.delenv("A2_RAG_ALLOW_EMBEDDING_FAILOVER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-that-must-not-be-used")
    imported = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "langchain_huggingface":
            raise ImportError("local backend unavailable")
        if name == "langchain_openai":
            imported.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(Exception, match="Configured embedding provider 'local' is unavailable"):
        embedder.get_embedder(force_new=True, provider="local")
    assert imported == []


def test_static_corpus_policy_blocks_dynamic_and_personalized_advice():
    from utils import static_corpus_policy

    assert static_corpus_policy("What is the current exchange rate?")["blocked"] is True
    assert static_corpus_policy("Should I take this medicine for my symptoms?")["blocked"] is True
    assert static_corpus_policy("According to the report, what was the current rate?")["blocked"] is False


def test_static_corpus_policy_blocks_medication_market_and_election_bypasses():
    from utils import static_corpus_policy

    assert static_corpus_policy("What amount of warfarin should an adult take?")["blocked"] is True
    assert static_corpus_policy("What is the Bitcoin price?")["blocked"] is True
    assert static_corpus_policy("Who won the last election?")["blocked"] is True
    assert static_corpus_policy("According to the document, what was the Bitcoin price?")["blocked"] is False


def test_google_provider_honors_explicit_model_override(monkeypatch):
    from providers import llm_factory

    captured = {}

    class FakeGoogle:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.ModuleType("langchain_google_genai")
    fake_module.ChatGoogleGenerativeAI = FakeGoogle
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)
    monkeypatch.delenv("A2_RAG_OFFLINE", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")

    _llm, provider = llm_factory.create_llm(model="gemini-test-override")

    assert captured["model"] == "gemini-test-override"
    assert provider == "google:gemini-test-override"


def test_openai_provider_honors_explicit_model_override(monkeypatch):
    from providers import llm_factory

    captured = {}

    def fake_openai(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(llm_factory, "_make_chat_openai", fake_openai)
    monkeypatch.delenv("A2_RAG_OFFLINE", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    _llm, provider = llm_factory.create_llm(model="gpt-test-override")

    assert captured["model"] == "gpt-test-override"
    assert provider == "openai:gpt-test-override"


def test_openai_provider_can_be_explicitly_disabled(monkeypatch):
    from providers import llm_factory
    from utils import RAGException

    monkeypatch.setattr(llm_factory, "USE_FALLBACK_MODEL", False)
    monkeypatch.delenv("A2_RAG_OFFLINE", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    with pytest.raises(RAGException, match="No usable LLM provider"):
        llm_factory.create_llm()


def test_pipelines_do_not_force_an_openrouter_model_on_other_providers(monkeypatch):
    import a2_rag.a2_pipeline as a2_pipeline
    import baseline_rag.baseline_pipeline as baseline_pipeline

    requested_models = []

    def fake_factory(model=None, **_kwargs):
        requested_models.append(model)
        return object(), "fake-provider"

    monkeypatch.setattr(a2_pipeline, "create_llm", fake_factory)
    monkeypatch.setattr(baseline_pipeline, "create_llm", fake_factory)

    a2_pipeline.A2RAG(["evidence"])
    baseline_pipeline.BaselineRAG(["evidence"])

    assert requested_models == [None, None]


def test_pipelines_reject_documents_that_normalize_to_empty(monkeypatch):
    import a2_rag.a2_pipeline as a2_pipeline
    import baseline_rag.baseline_pipeline as baseline_pipeline

    monkeypatch.setattr(a2_pipeline, "create_llm", lambda **_kwargs: (object(), "fake"))
    monkeypatch.setattr(baseline_pipeline, "create_llm", lambda **_kwargs: (object(), "fake"))

    with pytest.raises(Exception, match="non-empty document"):
        a2_pipeline.A2RAG([{}])
    with pytest.raises(Exception, match="non-empty document"):
        baseline_pipeline.BaselineRAG([{}])
