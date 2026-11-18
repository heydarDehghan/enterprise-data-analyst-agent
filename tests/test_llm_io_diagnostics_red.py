"""Phase 4-C-C.3-A contract. The real adapter uses MockTransport only."""

import json
from contextlib import nullcontext
from importlib.util import find_spec

import httpx
import pytest

from app import config
from app.agent import EnvChatCompletionsClient
from app.main import app
from app.secure_data import TenantContext
from test_agent_loop import call, final_turn, numeric_slot, structured_turn, tool_turn
from test_live_llm_integration import SQL, provider_transport, runtime_client


CAPTURE_FLAG = "KAMIDSO_CAPTURE_LLM_DIAGNOSTICS"
CAPTURE_PATH = "KAMIDSO_LLM_DIAGNOSTICS_PATH"
LIVE_FLAG = "KAMIDSO_RUN_LIVE_AGENT_TESTS"
ALPHA_NODE = (
    "live_tests/test_agent_end_to_end_live.py::"
    "test_live_chat_calls_paid_sql_and_tenant_tickets[demo-alpha-alpha.db-alpha_tickets.csv-A001-B]"
)
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.8-27b"
DEMO_SECRET = "test-only-provider-auth-sentinel"
REJECTED_MARKER = "rejected-diagnostic-text-sentinel"
QUESTION = "درآمد پرداخت‌شده سپتامبر ۲۰۲۶ و تعداد تیکت‌های مشتری را گزارش کن."


def diagnostic_run(monkeypatch, node_id=ALPHA_NODE):
    # RED scaffolding: keep exercising the real, uninstrumented path before
    # GREEN exists. The enabled tests still require a real file with real data.
    module = "live_tests.llm_diagnostics"
    if find_spec(module) is None:
        return nullcontext()
    from live_tests.llm_diagnostics import capture_run

    return capture_run(node_id=node_id, monkeypatch=monkeypatch)


@pytest.fixture(autouse=True)
def isolated_provider_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_ENV_FILE", tmp_path / "unused.env")
    monkeypatch.delenv(CAPTURE_FLAG, raising=False)
    monkeypatch.delenv(CAPTURE_PATH, raising=False)
    monkeypatch.setenv(LIVE_FLAG, "1")  # Simulated approval for the mocked target.
    monkeypatch.setenv("LLM_CHAT_COMPLETIONS_URL", ENDPOINT)
    monkeypatch.setenv("LLM_MODEL", MODEL)
    monkeypatch.setenv("LLM_API_KEY", DEMO_SECRET)
    monkeypatch.setenv("LLM_MAX_COMPLETION_TOKENS", "2048")
    monkeypatch.setenv("LLM_MAX_CALLS_PER_REQUEST", "8")
    monkeypatch.setenv("LLM_MAX_TOTAL_TOKENS_PER_REQUEST", "24000")


def wire(completion, *, cost):
    # Deliberately different from json.dumps(response.json()) defaults, so a
    # recorder that reserializes provider JSON cannot pass the byte check.
    completion["usage"] = {
        "prompt_tokens": 91, "completion_tokens": 27, "total_tokens": 118, "cost": cost,
    }
    return ("\n" + json.dumps(completion, ensure_ascii=False, indent=2) + "\r\n").encode("utf-8")


def structured_exchange(*, reject=False):
    selected = tool_turn(
        call("query_sql", {"sql": SQL, "params": ["2026-09"]}, "sql-choice"),
        call("get_tickets", {"customer_id": "A001"}, "tickets-choice"),
    )
    selected["choices"][0]["message"]["content"] = "ابتدا منابع داده را بررسی می‌کنم."
    template = "در {{period}}، تعداد تیکت {{count}} و درآمد {{revenue}} تومان بود."
    if reject:
        template += f" {REJECTED_MARKER}"
    final = structured_turn(
        template,
        count=numeric_slot("۲", "tickets-choice", "ticket_count", "count"),
        revenue=numeric_slot("۵۳٬۲۵۰٬۰۰۱" if reject else "۵۳٬۲۵۰٬۰۰۰", "sql-choice", "revenue"),
        period={"source": "user_request", "value": "۲۰۲۶-۰۹"},
    )
    return [
        (200, wire(selected, cost=0.00017)),
        (200, wire(final, cost=0.00019)),
    ], final["choices"][0]["message"]["content"]


def fake_provider(monkeypatch, replies):
    seen = []

    def handle(request):
        assert str(request.url) == ENDPOINT
        assert request.headers["authorization"] == f"Bearer {DEMO_SECRET}"
        seen.append(request.content)
        assert len(seen) <= len(replies), "Diagnostic capture must not add provider calls"
        reply = replies[len(seen) - 1]
        if isinstance(reply, Exception):
            raise reply
        status, data = reply
        return httpx.Response(status, content=data, headers={"Content-Type": "application/json"})

    provider_transport(monkeypatch, handle)
    return seen


def chat(tmp_path, monkeypatch, replies, *, enabled, node_id=ALPHA_NODE):
    path = tmp_path / "human-review.json"
    with monkeypatch.context() as patch:
        if enabled is not None:
            patch.setenv(CAPTURE_FLAG, enabled)
        patch.setenv(CAPTURE_PATH, str(path))
        sent = fake_provider(patch, replies)
        with runtime_client(tmp_path, patch, EnvChatCompletionsClient()) as client:
            tenant = app.state.secure_data.authenticate("demo-alpha")
            store = app.state.conversation_store
            conversation = store.create_conversation(tenant)
            with diagnostic_run(patch, node_id):
                response = client.post(
                    "/chat", headers={"X-Demo-Token": "demo-alpha"},
                    json={"question": QUESTION, "conversation_id": conversation},
                )
            snapshot = store.get_conversation(tenant, conversation)
    return response, snapshot, sent, path, conversation


def read_capture(path):
    assert path.is_file(), "Explicitly enabled run did not retain a diagnostic JSON file for review"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["test_node_id"] == ALPHA_NODE
    assert data["run_id"]
    return data


def body_bytes(message):
    body = message["body"]
    assert body["encoding"] == "utf-8"
    return body["data"].encode("utf-8")


@pytest.mark.parametrize("flag", [None, "0", "true"])
def test_default_and_nonexplicit_flags_leave_no_capture(tmp_path, monkeypatch, flag):
    replies, _ = structured_exchange()
    response, _, sent, path, _ = chat(tmp_path, monkeypatch, replies, enabled=flag)
    assert response.status_code == 200
    assert len(sent) == 2
    assert not path.exists()


def test_diagnostic_flag_alone_does_not_enable_capture(tmp_path, monkeypatch):
    monkeypatch.delenv(LIVE_FLAG)
    replies, _ = structured_exchange()
    result, _, sent, path, _ = chat(tmp_path, monkeypatch, replies, enabled="1")
    assert result.status_code == 200 and len(sent) == 2
    assert not path.exists()


@pytest.mark.parametrize("misconfiguration", ["missing_path", "relative_path", "existing_path", "wrong_node"])
def test_enabled_capture_requires_unused_path_and_exact_approved_case(tmp_path, monkeypatch, misconfiguration):
    monkeypatch.setenv(CAPTURE_FLAG, "1")
    path = tmp_path / "human-review.json"
    if misconfiguration != "missing_path":
        monkeypatch.setenv(CAPTURE_PATH, path.name if misconfiguration == "relative_path" else str(path))
    if misconfiguration == "existing_path":
        path.write_text("previous human review\n", encoding="utf-8")
    selected = ALPHA_NODE.replace("-A001-B]", "-A002-B]") if misconfiguration == "wrong_node" else ALPHA_NODE
    sent = fake_provider(monkeypatch, [])
    with pytest.raises(ValueError):
        with diagnostic_run(monkeypatch, selected):
            pass
    assert sent == []
    if misconfiguration == "existing_path":
        assert path.read_text(encoding="utf-8") == "previous human review\n"
    else:
        assert not path.exists()


def test_exact_provider_bodies_and_correlated_grounding_leave_execution_unchanged(tmp_path, monkeypatch):
    replies, final = structured_exchange()
    recorded_tools = []
    original_sql, original_tickets = TenantContext.agent_query, TenantContext.tickets

    def query(tenant, sql, params=()):
        recorded_tools.append(("query_sql", sql, tuple(params)))
        return original_sql(tenant, sql, params)

    def tickets(tenant, *args, **kwargs):
        recorded_tools.append(("get_tickets", args, kwargs))
        return original_tickets(tenant, *args, **kwargs)

    monkeypatch.setattr(TenantContext, "agent_query", query)
    monkeypatch.setattr(TenantContext, "tickets", tickets)
    baseline, baseline_state, baseline_sent, _, _ = chat(
        tmp_path / "baseline", monkeypatch, replies, enabled=None,
    )
    baseline_tools = list(recorded_tools)
    recorded_tools.clear()
    result, state, sent, path, conversation = chat(tmp_path / "diagnostic", monkeypatch, replies, enabled="1")
    assert baseline.status_code == result.status_code == 200
    assert sent == baseline_sent and len(sent) == 2
    assert recorded_tools == baseline_tools and len(recorded_tools) == 2
    assert state.messages == baseline_state.messages
    assert result.json()["answer"] == baseline.json()["answer"]

    def semantic_trace(response):
        trace = response.json()["trace"]
        assert len(trace) == len(recorded_tools)
        for step in trace:
            duration_ms = step.pop("duration_ms")
            assert type(duration_ms) is int and duration_ms >= 0
        return trace

    assert semantic_trace(result) == semantic_trace(baseline)

    data = read_capture(path)
    first, second = calls = data["provider_calls"]
    trace_id = result.json()["trace_id"]
    assert [item["index"] for item in calls] == [1, 2]
    assert len({item["call_id"] for item in calls}) == len(calls)
    for item, outbound, inbound in zip(calls, sent, replies):
        assert item["conversation_id"] == conversation and item["trace_id"] == trace_id
        assert item["request"]["method"] == "POST"
        assert item["request"]["url"] == ENDPOINT
        assert body_bytes(item["request"]) == outbound
        assert item["response"]["status_code"] == inbound[0]
        assert body_bytes(item["response"]) == inbound[1]
        assert item["request"]["headers"]["authorization"] == "[REDACTED]"
    assert data["runtime_limits"] == {"max_calls": 8, "max_total_tokens": 24_000}
    outbound = json.loads(body_bytes(second["request"]))
    assert outbound["model"] == MODEL and outbound["max_completion_tokens"] == 2048
    assert [message["role"] for message in outbound["messages"]] == [
        "system", "user", "assistant", "tool", "tool",
    ]
    assert outbound["messages"][1]["content"] == QUESTION
    assert {tool["function"]["name"] for tool in outbound["tools"]} >= {"query_sql", "get_tickets"}
    returned = {msg["tool_call_id"]: msg["content"] for msg in outbound["messages"] if msg["role"] == "tool"}
    assert set(returned) == {"sql-choice", "tickets-choice"}
    executions = data["tool_executions"]
    assert [(entry["step"], entry["tool_name"], entry["tool_call_id"], entry["status"]) for entry in executions] == [
        (1, "query_sql", "sql-choice", "success"), (2, "get_tickets", "tickets-choice", "success"),
    ]
    for entry in executions:
        assert entry["provider_call_id"] == first["call_id"]
        assert entry["trace_id"] == trace_id
        assert entry["result_text"] == returned[entry["tool_call_id"]]

    decision, = data["grounding"]
    assert decision["provider_call_id"] == second["call_id"] and decision["trace_id"] == trace_id
    assert decision["raw_final_content"] == final
    assert decision["template"] == json.loads(final)["template"]
    assert decision["slots"] == json.loads(final)["slots"]
    assert decision["rendered_answer"] == result.json()["answer"]
    assert decision["rejection_reason"] is None
    slots = {slot["slot"]: slot for slot in decision["slot_decisions"]}
    assert {key: (slot["decision"], slot["authoritative_value"]) for key, slot in slots.items()} == {
        "count": ("accepted", 2), "revenue": ("accepted", 53_250_000), "period": ("accepted", "2026-09"),
    }
    assert slots["revenue"]["evidence_ref"] == "sql-choice"
    assert slots["revenue"]["evidence_field"] == "revenue"
    assert DEMO_SECRET not in path.read_text(encoding="utf-8")


def test_rejected_content_is_only_in_capture_and_slot_outcomes_are_faithful(tmp_path, monkeypatch, caplog):
    replies, final = structured_exchange(reject=True)
    baseline, _, baseline_sent, _, _ = chat(tmp_path / "baseline", monkeypatch, replies, enabled=None)
    result, state, sent, path, _ = chat(tmp_path / "diagnostic", monkeypatch, replies, enabled="1")
    assert baseline.status_code == result.status_code == 502
    assert baseline.json() == result.json() == {"detail": "Analysis answer rejected"}
    assert sent == baseline_sent and len(sent) == 2
    turn, = state.turns
    assert turn.status == "failed" and turn.answer is None and turn.failure_code == "unsupported_number"
    assert REJECTED_MARKER not in result.text
    assert REJECTED_MARKER not in caplog.text
    assert REJECTED_MARKER not in json.dumps(state.messages, ensure_ascii=False)

    data = read_capture(path)
    last = data["provider_calls"][-1]
    assert body_bytes(last["response"]) == replies[-1][1]
    decision, = data["grounding"]
    assert decision["provider_call_id"] == last["call_id"]
    assert decision["trace_id"] == turn.trace_id
    assert decision["raw_final_content"] == final
    assert decision["rendered_answer"] is None and decision["rejection_reason"] == "unsupported_number"
    assert REJECTED_MARKER in decision["template"]
    slots = {slot["slot"]: slot for slot in decision["slot_decisions"]}
    assert {name: slot["decision"] for name, slot in slots.items()} == {
        "count": "accepted", "revenue": "rejected", "period": "not_evaluated",
    }
    revenue = slots["revenue"]
    assert revenue["evidence_ref"] == "sql-choice" and revenue["evidence_field"] == "revenue"
    assert revenue["proposed_value"] == "۵۳٬۲۵۰٬۰۰۱"
    assert revenue["authoritative_value"] == 53_250_000
    assert revenue["reason_code"] == "unsupported_number"


def test_provider_failures_are_captured_before_retry_with_only_auth_redacted(tmp_path, monkeypatch, caplog):
    failure = f" \r\nprovider says: auth={DEMO_SECRET}; جزئیات خطا\n".encode("utf-8")
    success = wire(structured_turn("پاسخ تکمیل شد."), cost=0.00011)
    replies = [
        (429, failure),
        httpx.ReadTimeout(f"socket timeout {DEMO_SECRET}"),
        (200, success),
    ]
    result, _, sent, path, _ = chat(tmp_path, monkeypatch, replies, enabled="1")
    assert result.status_code == 200
    assert len(sent) == 3 and sent[0] == sent[1] == sent[2]
    data = read_capture(path)
    first, second, third = data["provider_calls"]
    assert [first["index"], second["index"], third["index"]] == [1, 2, 3]
    assert [body_bytes(item["request"]) for item in (first, second, third)] == sent
    assert first["response"]["status_code"] == 429
    assert body_bytes(first["response"]) == failure.replace(DEMO_SECRET.encode(), b"[REDACTED]")
    assert second["response"] is None and second["transport_error"]["type"] == "ReadTimeout"
    assert second["transport_error"]["message"] == "socket timeout [REDACTED]"
    assert body_bytes(third["response"]) == success
    assert DEMO_SECRET not in path.read_text(encoding="utf-8")
    assert DEMO_SECRET not in caplog.text


def test_malformed_answer_is_preserved_without_inventing_parsed_slots(tmp_path, monkeypatch):
    final = '{"template":"پاسخ ناقص ' + REJECTED_MARKER
    reply = wire(final_turn(final), cost=0.00011)
    result, snapshot, sent, path, _ = chat(tmp_path, monkeypatch, [(200, reply)], enabled="1")
    assert result.status_code == 502 and len(sent) == 1
    assert snapshot.turns[0].failure_code == "invalid_template"
    assert REJECTED_MARKER not in json.dumps(snapshot.messages, ensure_ascii=False)
    data = read_capture(path)
    decision, = data["grounding"]
    assert decision["raw_final_content"] == final
    assert decision["rejection_reason"] == "invalid_template"
    assert decision["template"] is None and decision["slots"] is None
    assert decision["slot_decisions"] == [] and decision["rendered_answer"] is None
