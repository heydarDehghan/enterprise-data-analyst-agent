"""Phase 4-C-C.5-L-A. Recover once from an empty provider length completion."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import config
from app.secure_data import TenantContext
from test_agent_loop import ScriptedLLM, call, final_turn, structured_turn, tool_result
from test_evidence_reference_contract_red import manifest
from test_evidence_reference_repair_red import (
    REPAIRED_TEMPLATE, SQL_REF, UNKNOWN_REF, answer, chat as submit, script, service_at, snapshot,
    sql_turn, usage,
)
from test_llm_io_diagnostics_red import (
    CAPTURE_FLAG, DEMO_SECRET, ENDPOINT, LIVE_FLAG, MODEL, body_bytes, chat as capture_chat, read_capture,
)
from test_structured_answer_grounding_red import PAID_SQL, QUESTION, september_total


PARTIAL_MARKER = "UNFINISHED_PROVIDER_DRAFT"
PARTIAL_CALL_ID = "unfinished-tool"
PROVIDER_UNAVAILABLE = {"detail": "Analysis provider unavailable"}
BUDGET_EXCEEDED = {"detail": "Request budget exceeded"}
ANSWER_REJECTED = {"detail": "Analysis answer rejected"}


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_ENV_FILE", tmp_path / "unused.env")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MAX_CALLS_PER_REQUEST", "8")
    monkeypatch.setenv("LLM_MAX_TOTAL_TOKENS_PER_REQUEST", "40000")


@pytest.fixture
def mocked_wire(monkeypatch):
    # The capture helper always installs an HTTPX MockTransport before opening the app.
    monkeypatch.setenv(LIVE_FLAG, "1")
    monkeypatch.delenv(CAPTURE_FLAG, raising=False)
    monkeypatch.setenv("LLM_CHAT_COMPLETIONS_URL", ENDPOINT)
    monkeypatch.setenv("LLM_MODEL", MODEL)
    monkeypatch.setenv("LLM_API_KEY", DEMO_SECRET)
    monkeypatch.setenv("LLM_MAX_COMPLETION_TOKENS", "8192")


def length_completion(*, content=None, tool_calls=None):
    return {
        "choices": [{
            "finish_reason": "length",
            "message": {"role": "assistant", "content": content,
                        "tool_calls": [] if tool_calls is None else tool_calls},
        }],
        "usage": {
            "prompt_tokens": 2524, "completion_tokens": 8192, "total_tokens": 10716,
            "completion_tokens_details": {"reasoning_tokens": 8196},
        },
    }


def post(service, *, tenant="demo-alpha", conversation_id=None):
    with TestClient(service) as client:
        return submit(client, tenant, conversation_id)


def seed_conversation(service):
    owner = service.state.secure_data.authenticate("demo-alpha")
    return service.state.conversation_store.create_conversation(owner)


def assert_fresh_request(initial, regenerated):
    assert regenerated["tools"] == initial["tools"]
    assert regenerated["tools"]
    assert regenerated["messages"][1:] == initial["messages"][1:]
    assert manifest(regenerated["messages"]) == manifest(initial["messages"])
    assert PARTIAL_MARKER not in json.dumps(regenerated["messages"])


def assert_failed_turn(service, conversation_id, *, successful_tools=0):
    state = snapshot(service, conversation_id=conversation_id)
    turn, = state.turns
    assert turn.status == "failed" and turn.answer is None
    assert len(turn.trace) == successful_tools
    assert state.evidence == []
    assert not any(item["role"] == "assistant" and not item.get("tool_calls") for item in state.messages)
    transcript = json.dumps(state.messages)
    assert PARTIAL_MARKER not in transcript and PARTIAL_CALL_ID not in transcript


def test_captured_empty_length_recovers_with_one_real_sql_execution_and_grounded_answer(
    tmp_path, monkeypatch, september_total,
):
    queried = []
    actual_query = TenantContext.agent_query

    def record_query(tenant, sql, params=()):
        queried.append((tenant.display_label, sql))
        return actual_query(tenant, sql, params)

    monkeypatch.setattr(TenantContext, "agent_query", record_query)
    llm = script(length_completion(), sql_turn(), answer(september_total, SQL_REF))
    service = service_at(tmp_path, llm)
    conversation = seed_conversation(service)
    response = post(service, conversation_id=conversation)

    assert response.status_code == 200, response.text
    assert len(llm.calls) == 3
    first, retry, final = llm.calls
    assert_fresh_request(first, retry)
    assert retry["messages"][-1] == {"role": "user", "content": QUESTION}
    assert queried == [("Alpha", PAID_SQL)]
    assert tool_result(final["messages"], SQL_REF) == {"rows": [{"total_revenue": september_total}]}
    body = response.json()
    assert body["conversation_id"] == conversation
    assert body["answer"] == REPAIRED_TEMPLATE.replace("{{revenue}}", str(september_total))
    assert [(step["tool_name"], step["status"]) for step in body["trace"]] == [("query_sql", "success")]
    state = snapshot(service_at(tmp_path, ScriptedLLM()), conversation_id=conversation)
    turn, = state.turns
    assert (turn.status, turn.answer, turn.trace_id) == ("success", body["answer"], body["trace_id"])
    evidence, = state.evidence
    assert evidence.call_id == SQL_REF and evidence.scope.conversation_id == conversation
    assert [msg["content"] for msg in state.messages if msg["role"] == "assistant" and not msg.get("tool_calls")] == [
        body["answer"],
    ]


@pytest.mark.parametrize("tool_between", [False, True], ids=["consecutive", "after_a_tool"])
def test_second_empty_length_is_terminal_for_the_chat(tmp_path, tool_between):
    middle = [sql_turn()] if tool_between else []
    llm = script(length_completion(), *middle, length_completion(), structured_turn("این پاسخ اضافی است."))
    service = service_at(tmp_path, llm)
    conversation = seed_conversation(service)
    response = post(service, conversation_id=conversation)

    assert response.status_code == 503 and response.json() == PROVIDER_UNAVAILABLE
    assert len(llm.calls) == 2 + len(middle)
    assert_fresh_request(*llm.calls[:2])
    assert_failed_turn(service, conversation, successful_tools=len(middle))


@pytest.mark.parametrize("kind", ["text", "malformed_arguments", "complete_looking_tool", "invalid_tool_array"])
def test_length_with_partial_output_fails_without_recovery_or_tool_execution(
    tmp_path, monkeypatch, kind, caplog,
):
    def no_query(*args, **kwargs):
        pytest.fail("A length completion must never execute a tool")

    monkeypatch.setattr(TenantContext, "agent_query", no_query)
    tool = call("query_sql", {"sql": PAID_SQL}, PARTIAL_CALL_ID)
    if kind == "malformed_arguments":
        tool["function"]["arguments"] = '{"sql":"' + PARTIAL_MARKER
    if kind == "text":
        truncated = length_completion(content='{"template":"' + PARTIAL_MARKER)
    else:
        truncated = length_completion(tool_calls=tool if kind == "invalid_tool_array" else [tool])

    llm = script(truncated, structured_turn("این پاسخ اضافی است."))
    service = service_at(tmp_path, llm)
    conversation = seed_conversation(service)
    response = post(service, conversation_id=conversation)

    assert response.status_code == 503 and response.json() == PROVIDER_UNAVAILABLE
    assert len(llm.calls) == 1
    assert_failed_turn(service, conversation)
    assert PARTIAL_MARKER not in caplog.text + response.text


@pytest.mark.parametrize("scenario", ["cumulative", "initial_exhausted", "usage_missing"])
def test_length_recovery_charges_the_original_reported_tokens(tmp_path, scenario):
    truncated = length_completion()
    if scenario == "initial_exhausted":
        truncated = usage(truncated, 40_000)
    if scenario == "usage_missing":
        truncated["usage"] = {}
    llm = script(truncated, usage(structured_turn("پاسخ آماده شد."), 29_285))
    service = service_at(tmp_path, llm)
    conversation = seed_conversation(service)
    response = post(service, conversation_id=conversation)

    assert response.status_code == 429 and response.json() == BUDGET_EXCEEDED
    assert len(llm.calls) == (2 if scenario == "cumulative" else 1)
    assert_failed_turn(service, conversation)


@pytest.mark.parametrize("call_limit", [1, 2])
def test_length_recovery_shares_the_chat_call_limit(tmp_path, monkeypatch, call_limit, september_total):
    monkeypatch.setenv("LLM_MAX_CALLS_PER_REQUEST", str(call_limit))
    llm = script(length_completion(), sql_turn(), answer(september_total, SQL_REF))
    service = service_at(tmp_path, llm)
    conversation = seed_conversation(service)
    response = post(service, conversation_id=conversation)

    assert len(llm.calls) == call_limit
    if call_limit == 1:
        assert response.status_code == 503 and response.json() == PROVIDER_UNAVAILABLE
    else:
        assert response.status_code == 429 and response.json() == BUDGET_EXCEEDED
        assert_fresh_request(*llm.calls)
    assert_failed_turn(service, conversation, successful_tools=call_limit - 1)


@pytest.mark.parametrize("invalid", ["json", "value"])
def test_recovered_final_answer_still_requires_strict_grounding(tmp_path, september_total, invalid):
    final = final_turn('{"template":"ناقص') if invalid == "json" else answer(september_total + 1, SQL_REF)
    llm = script(sql_turn(), length_completion(), final)
    service = service_at(tmp_path, llm)
    conversation = seed_conversation(service)
    response = post(service, conversation_id=conversation)

    assert response.status_code == 502 and response.json() == ANSWER_REJECTED
    assert len(llm.calls) == 3
    assert_fresh_request(*llm.calls[1:])
    assert tool_result(llm.calls[2]["messages"], SQL_REF)["rows"] == [{"total_revenue": september_total}]
    assert_failed_turn(service, conversation, successful_tools=1)
    turn, = snapshot(service, conversation_id=conversation).turns
    assert turn.failure_code == ("invalid_template" if invalid == "json" else "unsupported_number")


@pytest.mark.parametrize("repair_result", ["valid", "unknown", "budget", "no_call_left"])
def test_length_recovery_and_evidence_repair_have_separate_bounds_and_shared_budgets(
    tmp_path, monkeypatch, september_total, repair_result,
):
    limit = 3 if repair_result == "no_call_left" else 4
    monkeypatch.setenv("LLM_MAX_CALLS_PER_REQUEST", str(limit))
    accepted = answer(september_total, SQL_REF)
    invalid = answer(september_total, UNKNOWN_REF)
    repair = invalid if repair_result == "unknown" else accepted
    if repair_result == "budget":
        repair = usage(repair, 29_085)  # 10716 + 100 + 100 + 29085 = 40001
    llm = script(length_completion(), sql_turn(), invalid, repair, accepted)
    service = service_at(tmp_path, llm)
    conversation = seed_conversation(service)
    response = post(service, conversation_id=conversation)

    expected_status = {"valid": 200, "unknown": 502, "budget": 429, "no_call_left": 502}
    assert response.status_code == expected_status[repair_result], response.text
    assert len(llm.calls) == limit
    assert_fresh_request(*llm.calls[:2])
    if limit == 4:
        assert llm.calls[3]["tools"] == []
        assert llm.calls[3]["messages"][1:] == llm.calls[2]["messages"][1:]
        assert UNKNOWN_REF not in json.dumps(llm.calls[3]["messages"])
    if repair_result == "valid":
        stored = snapshot(service, conversation_id=conversation)
        assert [record.call_id for record in stored.evidence] == [SQL_REF]
        assert len(stored.turns[0].trace) == 1
        assert UNKNOWN_REF not in json.dumps(stored.messages)
    else:
        assert response.json() == (BUDGET_EXCEEDED if repair_result == "budget" else ANSWER_REJECTED)
        assert_failed_turn(service, conversation, successful_tools=1)


def test_empty_length_inside_answer_only_repair_does_not_open_tool_path(tmp_path, september_total):
    llm = script(sql_turn(), answer(september_total, UNKNOWN_REF), length_completion(), sql_turn())
    service = service_at(tmp_path, llm)
    conversation = seed_conversation(service)
    response = post(service, conversation_id=conversation)

    assert response.status_code == 502 and response.json() == ANSWER_REJECTED
    assert len(llm.calls) == 3 and llm.calls[-1]["tools"] == []
    assert_failed_turn(service, conversation, successful_tools=1)


def test_recovery_keeps_owned_conversation_and_verified_evidence_after_restart(tmp_path, september_total):
    initial = service_at(tmp_path, script(sql_turn(), answer(september_total, SQL_REF)))
    seeded = post(initial)
    assert seeded.status_code == 200
    conversation = seeded.json()["conversation_id"]
    llm = script(length_completion(), answer(september_total, SQL_REF))
    reopened = service_at(tmp_path, llm)

    denied = post(reopened, tenant="demo-beta", conversation_id=conversation)
    assert denied.status_code == 404 and llm.calls == []
    response = post(reopened, conversation_id=conversation)

    assert response.status_code == 200, response.text
    assert response.json()["conversation_id"] == conversation and response.json()["trace"] == []
    assert len(llm.calls) == 2
    assert_fresh_request(*llm.calls)
    advertised = manifest(llm.calls[1]["messages"])
    assert advertised["context"]["tenant_context"] == {"value": "Alpha"}
    assert [entry["evidence_ref"] for entry in advertised["tools"]] == [SQL_REF]
    stored = snapshot(reopened, conversation_id=conversation)
    assert [turn.status for turn in stored.turns] == ["success", "success"]
    assert [record.call_id for record in stored.evidence] == [SQL_REF]


def encoded(completion):
    return ("\n" + json.dumps(completion, ensure_ascii=False, indent=2) + "\r\n").encode("utf-8")


@pytest.mark.parametrize("first_failure", ["transport", "invalid_structure"])
def test_ordinary_provider_failure_retries_without_entering_length_recovery(
    tmp_path, monkeypatch, mocked_wire, first_failure,
):
    failure = httpx.ConnectError("test transport interruption") if first_failure == "transport" else (200, b'{"choices": null}')
    replies = [failure, (200, encoded(usage(structured_turn("پاسخ ثبت شد."), 100)))]
    response, _, sent, _, _ = capture_chat(tmp_path, monkeypatch, replies, enabled=None)

    assert response.status_code == 200
    assert len(sent) == 2 and sent[0] == sent[1]


@pytest.mark.parametrize("recovery_failure", ["transport", "invalid_structure"])
def test_failed_recovery_attempt_is_not_retried_as_another_fresh_generation(
    tmp_path, monkeypatch, mocked_wire, recovery_failure,
):
    if recovery_failure == "transport":
        failure = httpx.ConnectError("test transport interruption")
    else:
        failure = (200, b'{"choices": null}')
    replies = [
        (200, encoded(length_completion())),
        failure,
        (200, encoded(usage(structured_turn("این پاسخ اضافی است."), 100))),
    ]
    response, state, sent, _, _ = capture_chat(tmp_path, monkeypatch, replies, enabled=None)

    assert response.status_code == 503 and response.json() == PROVIDER_UNAVAILABLE
    assert len(sent) == 2
    assert_fresh_request(*(json.loads(payload) for payload in sent))
    turn, = state.turns
    assert turn.status == "failed" and turn.answer is None


def test_non_length_finish_with_null_content_is_not_eligible(tmp_path):
    blocked = length_completion()
    blocked["choices"][0]["finish_reason"] = "content_filter"
    llm = script(blocked, structured_turn("پاسخ اضافی است."))
    response = post(service_at(tmp_path, llm))
    assert response.status_code == 503 and response.json() == PROVIDER_UNAVAILABLE
    assert len(llm.calls) == 1


@pytest.mark.parametrize("eligible", [True, False], ids=["recovered", "partial_discarded"])
def test_private_capture_links_original_length_and_recovery_outcome(
    tmp_path, monkeypatch, mocked_wire, caplog, eligible,
):
    truncated = length_completion(content=None if eligible else PARTIAL_MARKER)
    final = usage(structured_turn("پاسخ تازه ثبت شد."), 100)
    replies = [(200, encoded(truncated)), (200, encoded(final))]
    response, state, sent, path, conversation = capture_chat(tmp_path, monkeypatch, replies, enabled="1")
    recorded = read_capture(path)
    calls = recorded["provider_calls"]

    assert response.status_code == (200 if eligible else 503)
    assert len(sent) == len(calls) == (2 if eligible else 1)
    for record, outbound, (_, inbound) in zip(calls, sent, replies[:len(calls)], strict=True):
        assert body_bytes(record["request"]) == outbound
        assert body_bytes(record["response"]) == inbound
        assert record["request"]["headers"]["authorization"] == "[REDACTED]"
        payload = json.loads(outbound)
        assert payload["model"] == MODEL and payload["max_completion_tokens"] == 8192
        assert not {"reasoning", "reasoning_effort", "include_reasoning"} & payload.keys()
    assert [json.loads(body_bytes(item["response"]))["usage"]["total_tokens"] for item in calls] == (
        [10716, 100] if eligible else [10716]
    )
    if eligible:
        assert_fresh_request(*(json.loads(payload) for payload in sent))

    events = recorded.get("length_recovery", [])
    assert len(events) == 1
    event, = events
    turn, = state.turns
    assert event["original_provider_call_id"] == calls[0]["call_id"]
    assert event["recovery_provider_call_id"] == (calls[1]["call_id"] if eligible else None)
    assert event["conversation_id"] == conversation and event["trace_id"] == turn.trace_id
    assert event["eligible"] is eligible and event["attempts"] == int(eligible)
    assert event["outcome"] == turn.status == ("success" if eligible else "failed")
    assert recorded["tool_executions"] == [] and recorded["capture_errors"] == []
    assert len(recorded["grounding"]) == int(eligible)
    assert PARTIAL_MARKER not in json.dumps(state.messages) + response.text + caplog.text
    assert DEMO_SECRET not in path.read_text(encoding="utf-8") + caplog.text
