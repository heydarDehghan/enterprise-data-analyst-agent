"""Q2-E RED contracts at the one-shot evaluation recorder's JSON boundary."""

import json
from contextlib import nullcontext
from dataclasses import asdict

import httpx
import pytest

from app.observations import observer
from app.schemas import ToolTrace
from live_tests.llm_diagnostics import DiagnosticCaptureError, _private_output, _Recorder
from test_llm_io_diagnostics_red import DEMO_SECRET, body_bytes
from test_state_recovery import state_store, tenant


def test_q2_serializes_persisted_tool_traces_in_order_and_redacts_credentials(tmp_path):
    store = state_store(tmp_path)
    alpha = tenant()
    conversation_id = store.create_conversation(alpha)
    store.save_turn(
        alpha, conversation_id, trace_id="q2-trace-test", status="failed", answer=None,
        trace=[
            ToolTrace(step=1, tool_name="query_sql", status="success", duration_ms=4),
            ToolTrace(step=2, tool_name="get_tickets", status="error", duration_ms=0,
                      error=f"Tool call failed: {DEMO_SECRET}"),
        ],
        messages=[], artifacts=[],
    )
    turn, = state_store(tmp_path).get_conversation(alpha, conversation_id).turns
    assert all(isinstance(step, ToolTrace) for step in turn.trace)
    recorder = _Recorder("Q2-E/trace", DEMO_SECRET)
    # Use the same summary handoff as the Q2-C evaluation wrapper.
    recorder.data["application"] = {"persisted_turns": [asdict(turn)]}

    capture = tmp_path / "trace-capture.json"
    with _private_output(capture) as output:
        recorder.finish(output)

    raw = capture.read_text(encoding="utf-8")
    assert DEMO_SECRET not in raw
    data = json.loads(raw)
    assert data["capture_errors"] == []
    saved, = data["application"]["persisted_turns"]
    assert saved["trace_id"] == "q2-trace-test" and saved["status"] == "failed"
    assert saved["trace"] == [
        {"step": 1, "tool_name": "query_sql", "status": "success", "duration_ms": 4, "error": None},
        {"step": 2, "tool_name": "get_tickets", "status": "error", "duration_ms": 0,
         "error": "Tool call failed: [REDACTED]"},
    ]
    assert all(type(step["duration_ms"]) is int for step in saved["trace"])


@pytest.mark.parametrize("invalid_summary", [False, True], ids=["serializable", "invalid_summary"])
def test_q2_preserves_raw_provider_capture_when_optional_summary_fails(tmp_path, invalid_summary):
    request_body = f'\n{{ "input": "{DEMO_SECRET}" }}\r\n'.encode("utf-8")
    response_body = f' {{ "reply": "{DEMO_SECRET}" }}\n'.encode("utf-8")

    def provider(request):
        assert request.headers["authorization"] == f"Bearer {DEMO_SECRET}"
        assert request.content == request_body
        return httpx.Response(200, content=response_body, headers={"x-request-id": "q2-provider-test"})

    recorder = _Recorder("Q2-E/provider", DEMO_SECRET)
    recorder.sink = recorder.observe
    client_factory = recorder.instrument(httpx.Client)
    token = observer.set(recorder.sink)
    try:
        recorder.observe("turn", conversation_id="q2-conversation", trace_id="q2-trace",
                         max_calls=8, max_total_tokens=40000)
        recorder.observe("provider_start", index=1)
        with client_factory(transport=httpx.MockTransport(provider)) as client:
            response = client.post("https://provider.example/chat/completions", content=request_body,
                                   headers={"Authorization": f"Bearer {DEMO_SECRET}"})
        recorder.observe("provider_end")
        assert response.status_code == 200 and response.content == response_body
    finally:
        observer.reset(token)

    recorder.data["application"] = {"summary": object() if invalid_summary else "complete"}
    capture = tmp_path / "provider-capture.json"
    outcome = pytest.raises(DiagnosticCaptureError) if invalid_summary else nullcontext()
    with _private_output(capture) as output, outcome:
        recorder.finish(output)

    raw = capture.read_text(encoding="utf-8")
    assert DEMO_SECRET not in raw
    data = json.loads(raw)
    assert bool(data["capture_errors"]) is invalid_summary
    attempt, = data["provider_calls"]
    assert attempt["index"] == 1 and attempt["trace_id"] == "q2-trace"
    assert attempt["request"]["headers"]["authorization"] == "[REDACTED]"
    assert attempt["response"]["status_code"] == 200
    assert attempt["response"]["provider_request_id"] == "q2-provider-test"
    for side, original in (("request", request_body), ("response", response_body)):
        assert attempt[side]["body"]["credential_redactions"] == 1
        assert body_bytes(attempt[side]) == original.replace(DEMO_SECRET.encode("utf-8"), b"[REDACTED]")
