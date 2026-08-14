from ui.security import access_policy


def test_ui_fails_closed_without_token_or_explicit_local_mode():
    policy = access_policy({})
    assert policy.allowed is False


def test_ui_allows_explicit_local_demo_only_when_not_public():
    assert access_policy({"A2_RAG_LOCAL_DEMO": "1"}).allowed is True
    assert access_policy({"A2_RAG_LOCAL_DEMO": "1", "A2_RAG_PUBLIC": "1"}).allowed is False


def test_ui_requires_token_when_public():
    policy = access_policy({"A2_RAG_PUBLIC": "1", "A2_RAG_ACCESS_TOKEN": "token"})
    assert policy.allowed is True
    assert policy.requires_token is True
