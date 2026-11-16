"""Opt-in provider smoke test; excluded from the default tests/ collection."""

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("KAMIDSO_RUN_LIVE_LLM_TEST") != "1",
    reason="Set KAMIDSO_RUN_LIVE_LLM_TEST=1 explicitly to enable provider traffic",
)
def test_live_https_provider_returns_chat_completion():
    from app.agent import EnvChatCompletionsClient

    url = os.environ.get("LLM_CHAT_COMPLETIONS_URL", "")
    if not url.startswith("https://"):
        pytest.skip("An HTTPS chat-completions endpoint is required")
    if not os.environ.get("LLM_MODEL") or not os.environ.get("LLM_API_KEY"):
        pytest.skip("Provider configuration is unavailable")

    response = EnvChatCompletionsClient().create(
        messages=[{"role": "user", "content": "Reply briefly in Persian."}],
        tools=[],
    )
    assert isinstance(response.get("choices"), list) and response["choices"]
    assert response["choices"][0]["finish_reason"] in ("stop", "tool_calls")
