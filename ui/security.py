"""Pure access-policy helpers for the Streamlit private demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


_TRUE_VALUES = {"1", "true", "True"}


@dataclass(frozen=True)
class AccessPolicy:
    allowed: bool
    requires_token: bool
    message: str


def access_policy(environ: Mapping[str, str] | None = None) -> AccessPolicy:
    """Fail closed unless an access token or explicit local-demo mode is set."""
    env = os.environ if environ is None else environ
    has_token = bool(env.get("A2_RAG_ACCESS_TOKEN", ""))
    public_mode = env.get("A2_RAG_PUBLIC", "0") in _TRUE_VALUES
    local_demo = env.get("A2_RAG_LOCAL_DEMO", "0") in _TRUE_VALUES

    if public_mode and local_demo:
        return AccessPolicy(False, False, "Public mode cannot use A2_RAG_LOCAL_DEMO.")
    if has_token:
        return AccessPolicy(True, True, "Access token required.")
    if local_demo and not public_mode:
        return AccessPolicy(True, False, "Explicit localhost/private demo mode.")
    return AccessPolicy(
        False,
        False,
        "Set A2_RAG_ACCESS_TOKEN, or explicitly set A2_RAG_LOCAL_DEMO=1 for a localhost-only demo.",
    )
