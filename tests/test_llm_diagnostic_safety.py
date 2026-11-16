"""Offline checks for lossless binary payloads and capture failures."""

import base64

import pytest

from app.agent import EnvChatCompletionsClient
from live_tests import llm_diagnostics
from test_agent_loop import structured_turn
from test_llm_io_diagnostics_red import (
    ALPHA_NODE, CAPTURE_FLAG, CAPTURE_PATH, DEMO_SECRET, QUESTION, chat, fake_provider,
    isolated_provider_environment, read_capture, runtime_client, wire,
)


def test_non_utf8_error_body_is_lossless_except_credential_redaction(tmp_path, monkeypatch):
    raw = b"\xff\x80\r\n" + DEMO_SECRET.encode("ascii") + b"\x00"
    replies = [(503, raw), (200, wire(structured_turn("پاسخ آماده شد."), cost=0.00011))]
    response, _, sent, path, _ = chat(tmp_path, monkeypatch, replies, enabled="1")
    assert response.status_code == 200 and len(sent) == 2
    first = read_capture(path)["provider_calls"][0]
    body = first["response"]["body"]
    assert body["encoding"] == "base64" and body["credential_redactions"] == 1
    assert base64.b64decode(body["data"], validate=True) == raw.replace(
        DEMO_SECRET.encode("ascii"), b"[REDACTED]",
    )


@pytest.mark.parametrize("fail_capture", [False, True])
def test_file_is_retained_and_capture_failure_does_not_change_chat(
    tmp_path, monkeypatch, fail_capture,
):
    path = tmp_path / "retain.json"
    monkeypatch.setenv(CAPTURE_FLAG, "1")
    monkeypatch.setenv(CAPTURE_PATH, str(path))
    raw = wire(structured_turn("پاسخ آماده شد."), cost=0.00011)
    sent = fake_provider(monkeypatch, [(200, raw)])
    if fail_capture:
        monkeypatch.setattr(llm_diagnostics, "_MAX_BYTES", 1024)
    expected = llm_diagnostics.DiagnosticCaptureError if fail_capture else AssertionError
    with runtime_client(tmp_path, monkeypatch, EnvChatCompletionsClient()) as client:
        with pytest.raises(expected):
            with llm_diagnostics.capture_run(node_id=ALPHA_NODE, monkeypatch=monkeypatch):
                response = client.post(
                    "/chat", headers={"X-Demo-Token": "demo-alpha"}, json={"question": QUESTION},
                )
                if not fail_capture:
                    raise AssertionError("Deliberate test assertion after the provider response")
    assert response.status_code == 200 and len(sent) == 1
    assert response.json()["answer"] == "پاسخ آماده شد."
    result = read_capture(path)
    if fail_capture:
        assert result["capture_errors"]
    else:
        assert result["capture_errors"] == []
        assert result["provider_calls"][0]["response"]["body"]["data"].encode("utf-8") == raw
