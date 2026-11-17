"""Phase 4-C-C.5-S-A. Exercise the canonical citation contract after restart."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agent import EnvChatCompletionsClient
from live_tests.llm_diagnostics import PYTHON_PNG_NODE, capture_run
from live_tests.test_agent_end_to_end_live import paid_september_total
from test_agent_loop import numeric_slot, structured_turn, tool_result
from test_evidence_reference_contract_red import EVIDENCE_PREFIX, manifest
from test_evidence_reference_repair_red import answer, assert_rejected, script, service_at, snapshot, sql_turn, usage
from test_final_json_format_recovery_red import content_of
from test_followup_period_context_red import (
    CAPTURED_QUESTION, LIVE_FOLLOWUP, SEPTEMBER_CONTEXT, period_slot, request_date, sandbox_result,
)
from test_length_completion_recovery_red import encoded, isolated_settings, mocked_wire
from test_llm_io_diagnostics_red import CAPTURE_FLAG, CAPTURE_PATH, DEMO_SECRET, body_bytes, fake_provider
from test_python_png_accepted_response_red import FOREIGN, OWNER, python_turn
from test_python_png_diagnostic_contract_red import assert_decoded_chart
from test_python_sandbox import docker_ready
from test_structured_answer_grounding_red import september_total


CANONICAL = "5889ea83-a043-9257-9ebe-3da1d6bf7a33_0"
OTHER = "b1ec8f32-073a-486b-a4fc-1c1f31fff932_0"
FOREIGN_REF = "foreign-issued-september"
CURRENT_SQL = "followup-new-september-sql"
ALIASES = ("call_2", "execute_python_call_1")
PERIOD = SEPTEMBER_CONTEXT["period"]
FIELD = {"evidence_field": "revenue", "type": "numeric", "unit": "scalar"}
RULES_PREFIX = "KAMIDSO_CITATION_RULES_V1 "


def report(total, ref):
    return structured_turn(
        "در دورهٔ {{period}}، از {{start}} تا {{end}}، مبلغ {{revenue}} تومان بود.",
        period=period_slot(PERIOD),
        start=request_date("period_start", SEPTEMBER_CONTEXT["period_start"]),
        end=request_date("period_end", SEPTEMBER_CONTEXT["period_end"]),
        revenue=numeric_slot(total, ref, "revenue", "scalar"),
    )


@pytest.fixture
def restarted_report(tmp_path, monkeypatch, mocked_wire, september_total):
    def run(*replies, refs=(CANONICAL,), foreign=False, real_docker=False, call_limit=8):
        def submit(client, headers, question, conversation=None):
            body = {"question": question}
            if conversation is not None:
                body["conversation_id"] = conversation
            return client.post("/chat", headers=headers, json=body)

        if foreign:
            beta_total = paid_september_total("beta.db")
            beta = service_at(tmp_path, script(sql_turn(FOREIGN_REF), answer(beta_total, FOREIGN_REF)))
            with TestClient(beta) as client:
                response = submit(client, FOREIGN, CAPTURED_QUESTION)
            assert response.status_code == 200
            issued, = snapshot(beta, response, tenant="demo-beta").evidence
            assert issued.call_id == FOREIGN_REF and issued.succeeded

        first = service_at(tmp_path, script(*(python_turn(ref) for ref in refs), report(september_total, refs[0])))
        if not real_docker:
            first.state.sandbox = sandbox_result(revenue=str(september_total))
        with TestClient(first) as client:
            initial = submit(client, OWNER, CAPTURED_QUESTION)
            assert initial.status_code == 200
            initial_body = initial.json()
            initial_png = None
            if real_docker:
                artifact, = initial_body["artifacts"]
                path_url = f"/artifacts/{artifact['artifact_id']}"
                image = client.get(path_url, headers=OWNER)
                assert image.status_code == 200
                assert_decoded_chart(image.content, september_total)
                initial_png = image.content

        persisted = snapshot(first, initial)
        conversation = initial_body["conversation_id"]
        assert {record.call_id for record in persisted.evidence} == set(refs)
        monkeypatch.setenv("LLM_MAX_CALLS_PER_REQUEST", str(call_limit))
        mock_replies = [(200, encoded(reply if "usage" in reply else usage(reply, 100))) for reply in replies]
        sent = fake_provider(monkeypatch, mock_replies)
        monkeypatch.setenv(CAPTURE_FLAG, "1")
        path = tmp_path / "followup-capture.json"
        monkeypatch.setenv(CAPTURE_PATH, str(path))
        restarted = service_at(tmp_path, EnvChatCompletionsClient())
        if not real_docker:
            restarted.state.sandbox = sandbox_result(revenue=str(september_total))
        before = snapshot(restarted, conversation_id=conversation)
        assert before.evidence == persisted.evidence and restarted is not first

        with capture_run(node_id=PYTHON_PNG_NODE, monkeypatch=monkeypatch), TestClient(restarted) as client:
            response = submit(client, OWNER, LIVE_FOLLOWUP, conversation)
            attempts = len(sent)
            assert submit(client, FOREIGN, LIVE_FOLLOWUP, conversation).status_code == 404
            if real_docker:
                image = client.get(path_url, headers=OWNER)
                assert image.status_code == 200 and image.content == initial_png
                assert client.get(path_url, headers=FOREIGN).status_code == 404
            assert len(sent) == attempts

        stored = snapshot(restarted, conversation_id=conversation)
        assert stored.turns[0] == persisted.turns[0]
        assert len(stored.turns) == 2 and stored.turns[1].trace_id != initial_body["trace_id"]
        assert stored.artifacts == persisted.artifacts
        assert not any(step.tool_name == "execute_python" for step in stored.turns[1].trace)
        assert not any(
            call["function"]["name"] == "execute_python"
            for message in stored.messages[len(persisted.messages):]
            if message["role"] == "assistant" for call in message.get("tool_calls") or []
        )
        if not real_docker:
            restarted.state.sandbox.execute.assert_not_called()

        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        assert DEMO_SECRET not in raw and data["capture_errors"] == []
        assert data["runtime_limits"] == {"max_calls": call_limit, "max_total_tokens": 40_000}
        assert len(data["provider_calls"]) == len(sent)
        for recorded, request_bytes, (_, response_bytes) in zip(data["provider_calls"], sent, mock_replies[:len(sent)], strict=True):
            assert recorded["trace_id"] == stored.turns[-1].trace_id and recorded["conversation_id"] == conversation
            assert recorded["request"]["headers"]["authorization"] == "[REDACTED]"
            assert body_bytes(recorded["request"]) == request_bytes
            assert body_bytes(recorded["response"]) == response_bytes
        return SimpleNamespace(
            response=response, state=stored, requests=[json.loads(request) for request in sent],
            capture=data, transcript=json.dumps(stored.messages, ensure_ascii=False),
        )

    return run


def assert_canonical_manifest(request, *refs):
    evidence = manifest(request["messages"])
    assert evidence["context"]["user_request"] == SEPTEMBER_CONTEXT
    assert evidence["context"]["tenant_context"] == {"value": "Alpha"}
    assert not evidence["truncated"]
    python_entries = [item for item in evidence["tools"] if item["tool_name"] == "execute_python"]
    assert len(python_entries) == len(refs)
    assert {item["evidence_ref"]: item["fields"] for item in python_entries} == {
        ref: [FIELD] for ref in refs
    }
    for ref in refs:
        assert any(call["id"] == ref for message in request["messages"] if message["role"] == "assistant"
                   for call in message.get("tool_calls") or [])
        assert json.loads(tool_result(request["messages"], ref)["stdout"])["metrics"]["revenue"].isdecimal()


@pytest.mark.parametrize("attempt_index", [0, 1], ids=["initial", "repair"])
def test_system_prompt_explicitly_teaches_verbatim_canonical_citation(restarted_report, september_total, attempt_index):
    rejected = report(september_total, ALIASES[0])
    accepted = report(september_total, CANONICAL)
    run = restarted_report(rejected, accepted)
    assert run.response.status_code == 200 and len(run.requests) == 2
    request = run.requests[attempt_index]
    assert_canonical_manifest(request, CANONICAL)
    lines = request["messages"][0]["content"].splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(EVIDENCE_PREFIX))
    rules = [line.removeprefix(RULES_PREFIX) for line in lines[index + 1:] if line.startswith(RULES_PREFIX)]
    assert len(rules) == 1, "Expected explicit canonical citation rules beside the evidence manifest"
    text = rules[0].lower()
    for requirement in (
        "copy the complete evidence_ref verbatim",
        "evidence_field, type, unit",
        "never emit <actual_tool_call_id>",
        "tool name, call order, or chat history",
    ):
        assert requirement in text
    assert all(alias not in json.dumps(request["messages"]) for alias in ALIASES)


def assert_failed_turn(run, code):
    assert (run.state.turns[-1].status, run.state.turns[-1].answer, run.state.turns[-1].failure_code) == (
        "failed", None, code,
    )
    assert [record.call_id for record in run.state.evidence] == [CANONICAL]


def test_phase_r_aliases_fail_closed_after_one_repair_with_real_docker(
    restarted_report, september_total, docker_ready,
):
    drafts = [report(september_total, alias) for alias in ALIASES]
    run = restarted_report(*drafts, real_docker=True)
    assert_rejected(run.response)
    assert_failed_turn(run, "unsupported_number")
    original, repair = run.requests
    for request in (original, repair):
        assert_canonical_manifest(request, CANONICAL)
        # Safe fallback while this adapter and route have no verified enum contract.
        assert set(request) == {"model", "messages", "tools", "max_completion_tokens"}
        assert request["max_completion_tokens"] == 8192
        assert all(alias not in json.dumps(request["messages"]) for alias in ALIASES)
    assert repair["messages"][1:] == original["messages"][1:] and repair["tools"] == []
    first, second = run.capture["grounding"]
    assert first["repair"] == {
        "reason": "unknown_evidence_reference", "eligible": True, "attempted": True, "blocked_by_call_limit": False,
    }
    assert [item["raw_final_content"] for item in (first, second)] == [content_of(reply) for reply in drafts]
    assert [item["is_repair"] for item in (first, second)] == [False, True]
    assert all(item["rejection_reason"] == "unsupported_number" for item in (first, second))
    assert all(item["slot_decisions"][-1]["decision"] == "rejected" for item in (first, second))
    assert not run.capture["tool_executions"]
    assert all(alias not in run.transcript for alias in ALIASES)


def test_repair_accepts_exact_issued_reference_without_promoting_or_executing_a_tool(restarted_report, september_total):
    rejected = report(september_total, ALIASES[0])
    accepted = report(september_total, CANONICAL)
    run = restarted_report(rejected, accepted)
    assert run.response.status_code == 200
    original, repair = run.requests
    assert_canonical_manifest(original, CANONICAL)
    assert_canonical_manifest(repair, CANONICAL)
    assert repair["tools"] == []
    first, second = run.capture["grounding"]
    assert first["repair"]["attempted"] and first["rejection_reason"] == "unsupported_number"
    assert second["is_repair"] and second["raw_final_content"] == content_of(accepted)
    assert second["rejection_reason"] is None and second["slot_decisions"][-1]["decision"] == "accepted"
    assert [record.call_id for record in run.state.evidence] == [CANONICAL]
    assert run.state.turns[-1].status == "success" and run.state.turns[-1].answer == second["rendered_answer"]
    assert ALIASES[0] not in run.transcript and not run.capture["tool_executions"]


@pytest.mark.parametrize("ref", [CANONICAL, OTHER, ALIASES[0]], ids=["first", "second", "alias"])
def test_multiple_same_value_references_are_distinguishable_without_heuristic_rewrite(restarted_report, september_total, ref):
    run = restarted_report(report(september_total, ref), refs=(CANONICAL, OTHER))
    request, = run.requests
    assert_canonical_manifest(request, CANONICAL, OTHER)
    assert {record.call_id for record in run.state.evidence} == {CANONICAL, OTHER}
    grounding, = run.capture["grounding"]
    if ref == ALIASES[0]:
        assert_rejected(run.response)
        assert grounding["rejection_reason"] == "unsupported_number"
        assert not grounding.get("repair", {}).get("attempted", False)
    else:
        assert run.response.status_code == 200
        assert grounding["slots"]["revenue"]["evidence_ref"] == ref
        assert grounding["rejection_reason"] is None and not grounding["is_repair"]


def test_foreign_issued_reference_is_rejected_before_repair(restarted_report, september_total):
    run = restarted_report(report(september_total, FOREIGN_REF), foreign=True)
    assert_rejected(run.response)
    assert_failed_turn(run, "unsupported_number")
    request, = run.requests
    assert_canonical_manifest(request, CANONICAL)
    assert FOREIGN_REF not in json.dumps(request["messages"])
    grounding, = run.capture["grounding"]
    assert grounding["rejection_reason"] == "unsupported_number"
    assert grounding["repair"]["eligible"] is False and grounding["repair"]["attempted"] is False


@pytest.mark.parametrize("limit", ["calls", "tokens"])
def test_repair_shares_existing_per_chat_budgets(restarted_report, september_total, limit):
    rejected = report(september_total, ALIASES[0])
    if limit == "calls":
        run = restarted_report(rejected, call_limit=1)
        assert_rejected(run.response)
        assert len(run.requests) == 1
        assert run.capture["grounding"][0]["repair"]["blocked_by_call_limit"] is True
        assert_failed_turn(run, "unsupported_number")
    else:
        run = restarted_report(usage(rejected, 39_999), usage(report(september_total, CANONICAL), 1))
        assert run.response.status_code == 429 and run.response.json() == {"detail": "Request budget exceeded"}
        assert len(run.requests) == 2 and run.requests[-1]["tools"] == []
        assert len(run.capture["grounding"]) == 1 and run.capture["grounding"][0]["repair"]["attempted"] is True
        assert_failed_turn(run, None)


def test_failed_repair_does_not_promote_new_sql_from_failed_turn(restarted_report, september_total):
    run = restarted_report(sql_turn(CURRENT_SQL), *(report(september_total, ref) for ref in ALIASES))
    assert_rejected(run.response)
    assert_failed_turn(run, "unsupported_number")
    execution, = run.capture["tool_executions"]
    assert (execution["tool_call_id"], execution["tool_name"], execution["status"]) == (CURRENT_SQL, "query_sql", "success")
    assert run.requests[-1]["tools"] == [] and len(run.requests) == 3
    assert all(alias not in run.transcript for alias in ALIASES)
