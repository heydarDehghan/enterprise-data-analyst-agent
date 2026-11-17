"""Phase 4-C-C.5-N-A. Bounded regeneration for a prose-framed structured answer."""

import json

import pytest
from fastapi.testclient import TestClient

from app.structured_grounding import (
    EvidenceScope, GroundingContext, StructuredGroundingError, validate_and_render,
)
from live_tests.test_agent_end_to_end_live import paid_september_total
from test_agent_loop import ScriptedLLM, call, final_turn, numeric_slot, structured_turn, tool_result, tool_turn
from test_evidence_reference_contract_red import manifest
from test_evidence_reference_repair_red import (
    SQL_REF, UNKNOWN_REF, answer, script, service_at, snapshot, sql_turn, usage,
)
from test_gregorian_month_boundary_red import request_date
from test_length_completion_recovery_red import (
    encoded, isolated_settings, length_completion, mocked_wire, post, seed_conversation,
)
from test_llm_io_diagnostics_red import DEMO_SECRET, body_bytes, chat as capture_chat, read_capture
from test_python_png_diagnostic_contract_red import PYTHON_CODE, assert_decoded_chart
from test_python_sandbox import docker_ready
from test_structured_answer_grounding_red import PAID_SQL, september_total


PHASE_M_PREFIX = (
    "\n\nبرنامه با موفقیت اجرا شد: فایل /data/paid_orders.csv خوانده شد، درآمد "
    "سفارش‌های پرداخت‌شده سپتامبر ۲۰۲۶ محاسبه شد، نمودار PNG در "
    "/output/paid_revenue.png ساخته شد و خروجی stdout فقط شیء JSON خواسته‌شده بود.\n\n"
)
PYTHON_REF = "call_915fff4f89b242848249726f"
PHASE_M_TEMPLATE = (
    "برای مستأجر {{tenant_label}}، مجموع درآمد سفارش‌های پرداخت‌شده در ماه "
    "{{user_period}} میلادی (از {{period_start}} تا {{period_end}}) برابر با {{revenue}} تومان "
    "محاسبه شد. این مبلغ با یک برنامهٔ Python با کتابخانهٔ استاندارد، مستقیم از فایل "
    "/data/paid_orders.csv به دست آمد و نمودار تصویری آن در مسیر /output/paid_revenue.png ذخیره شد."
)
NEW_TEMPLATE = "  مجموع درآمد پرداخت‌شده {{revenue}} تومان است.  "
PROSE = "محاسبه پایان یافت."
REJECTED = {"detail": "Analysis answer rejected"}
BUDGET_EXCEEDED = {"detail": "Request budget exceeded"}


def content_of(completion):
    return completion["choices"][0]["message"]["content"]


def assert_format_request(original, fresh, rejected):
    assert fresh["tools"] == []
    assert fresh["messages"][1:] == original["messages"][1:]
    assert fresh["messages"][0]["content"].startswith(original["messages"][0]["content"])
    assert manifest(fresh["messages"]) == manifest(original["messages"])
    outbound = "\n".join(item.get("content") or "" for item in fresh["messages"])
    assert rejected not in outbound
    assert PROSE not in outbound and PHASE_M_PREFIX.strip() not in outbound


def assert_failed(service, conversation_id, *, failure_code="invalid_template", tool_count=1):
    state = snapshot(service, conversation_id=conversation_id)
    turn, = state.turns
    assert (turn.status, turn.answer, turn.failure_code) == ("failed", None, failure_code)
    assert len(turn.trace) == tool_count and state.evidence == []
    assert not any(m["role"] == "assistant" and not m.get("tool_calls") for m in state.messages)
    return state


def test_phase_m_prose_is_still_rejected_by_parser_and_fresh_answer_preserves_docker_evidence(
    tmp_path, docker_ready, september_total,
):
    captured_shape = structured_turn(
        PHASE_M_TEMPLATE,
        tenant_label={"source": "tenant_context", "value": "Alpha"},
        user_period={"source": "user_request", "value": "2026-09"},
        period_start=request_date("period_start", "2026-09-01"),
        period_end=request_date("period_end", "2026-09-30"),
        revenue=numeric_slot(september_total, PYTHON_REF, "revenue", "scalar"),
    )
    rejected = PHASE_M_PREFIX + content_of(captured_shape)
    fresh = answer(september_total, PYTHON_REF, "revenue", "scalar", NEW_TEMPLATE)
    llm = script(tool_turn(call("execute_python", {"code": PYTHON_CODE}, PYTHON_REF)), final_turn(rejected), fresh)
    service = service_at(tmp_path, llm)
    conversation = seed_conversation(service)
    owner = service.state.secure_data.authenticate("demo-alpha")
    context = GroundingContext(EvidenceScope(owner.state_key, conversation), "Alpha")

    with pytest.raises(StructuredGroundingError, match="^invalid_template$"):
        validate_and_render(rejected, context=context)

    response = post(service, conversation_id=conversation)
    state = snapshot(service, conversation_id=conversation)
    output = tool_result(state.messages, PYTHON_REF)
    assert output["exit_code"] == 0 and output["stderr"] == ""
    assert json.loads(output["stdout"]) == {"metrics": {"revenue": str(september_total)}}
    artifact, = state.artifacts

    assert response.status_code == 200, response.text
    assert len(llm.calls) == 3
    assert_format_request(llm.calls[1], llm.calls[2], rejected)
    body = response.json()
    rendered = NEW_TEMPLATE.replace("{{revenue}}", str(september_total))
    assert body["answer"] == rendered and body["conversation_id"] == conversation
    assert [item["artifact_id"] for item in body["artifacts"]] == [artifact.artifact_id]
    assert [(item["tool_name"], item["status"]) for item in body["trace"]] == [("execute_python", "success")]

    offline = ScriptedLLM()
    reopened = service_at(tmp_path, offline)
    state = snapshot(reopened, conversation_id=conversation)
    turn, = state.turns
    evidence, = state.evidence
    assert turn.status == "success" and turn.answer == rendered and turn.trace_id == body["trace_id"]
    assert evidence.call_id == PYTHON_REF and evidence.succeeded and evidence.result == output
    assert [m["content"] for m in state.messages if m["role"] == "assistant" and not m.get("tool_calls")] == [rendered]
    with TestClient(reopened) as client:
        path = f"/artifacts/{artifact.artifact_id}"
        image = client.get(path, headers={"X-Demo-Token": "demo-alpha"})
        assert image.status_code == 200 and image.headers["content-type"] == "image/png"
        assert_decoded_chart(image.content, september_total)
        assert client.get(path, headers={"X-Demo-Token": "demo-beta"}).status_code == 404
    assert post(reopened, tenant="demo-beta", conversation_id=conversation).status_code == 404
    assert offline.calls == []


@pytest.mark.parametrize("frame", ["suffix", "after_length"])
def test_scoped_format_repair_keeps_its_own_bound_after_normal_or_length_recovery(
    tmp_path, monkeypatch, september_total, frame,
):
    earlier = [length_completion()] if frame == "after_length" else []
    clean = content_of(answer(september_total, SQL_REF))
    rejected = clean + "\n\n" + PROSE if frame == "suffix" else PROSE + "\n\n" + clean
    limit = 3 + len(earlier)
    monkeypatch.setenv("LLM_MAX_CALLS_PER_REQUEST", str(limit))
    new = answer(september_total, SQL_REF, template=NEW_TEMPLATE)
    llm = script(*earlier, sql_turn(), final_turn(rejected), new)
    service = service_at(tmp_path, llm)
    conversation = seed_conversation(service)
    response = post(service, conversation_id=conversation)

    assert response.status_code == 200, response.text
    assert len(llm.calls) == limit
    if earlier:
        assert llm.calls[0]["tools"] == llm.calls[1]["tools"] and llm.calls[1]["tools"]
    assert_format_request(*llm.calls[-2:], rejected)
    state = snapshot(service, conversation_id=conversation)
    assert [item.call_id for item in state.evidence] == [SQL_REF]
    assert len(state.turns[0].trace) == 1 and state.turns[0].status == "success"
    assert PROSE not in json.dumps(state.messages, ensure_ascii=False)


@pytest.mark.parametrize("invalid", [
    "no_template", "no_slots", "placeholder_mismatch", "wrong_value", "wrong_unit", "wrong_date",
    "wrong_month_sql", "unknown_evidence", "wrong_source", "multiple_objects", "truncated_json",
    "duplicate_names", "array", "fenced",
])
def test_invalid_framed_answers_remain_terminal(tmp_path, september_total, invalid):
    proper = content_of(answer(september_total, SQL_REF))
    structured = {
        "no_template": '{"slots":{}}',
        "no_slots": '{"template":"درآمد ثبت شد."}',
        "placeholder_mismatch": content_of(structured_turn("درآمد {{missing}} ثبت شد.")),
        "wrong_value": content_of(answer(september_total + 1, SQL_REF)),
        "wrong_unit": content_of(answer(september_total, SQL_REF, unit="scalar")),
        "wrong_date": content_of(answer(september_total, SQL_REF, template="{{end}}، درآمد {{revenue}} بود.",
                                        end=request_date("period_end", "2026-10-01"))),
        "unknown_evidence": content_of(answer(september_total, UNKNOWN_REF)),
        "wrong_source": content_of(structured_turn("دورهٔ {{period}} است.", period={
            "source": "provider_claim", "value": "2026-09",
        })),
        "multiple_objects": proper + "\n" + content_of(answer(september_total + 1, SQL_REF)),
        "truncated_json": proper[:-1],
        "duplicate_names": '{"template":"پاسخ","template":"جواب","slots":{}}',
        "array": "[" + proper + "]",
        "fenced": "\x60\x60\x60json\n" + proper + "\n\x60\x60\x60",
    }.get(invalid, proper)
    sql = PAID_SQL
    if invalid == "wrong_month_sql":
        sql = sql.replace("2026-09-01", "2026-10-01").replace("order_date < '2026-10-01'", "order_date < '2026-11-01'")
    llm = script(sql_turn(sql=sql), final_turn(PROSE + "\n" + structured), answer(september_total, SQL_REF))
    service = service_at(tmp_path, llm)
    conversation = seed_conversation(service)
    response = post(service, conversation_id=conversation)

    assert response.status_code == 502 and response.json() == REJECTED
    assert len(llm.calls) == 2
    assert_failed(service, conversation, failure_code="invalid_date_range" if invalid == "wrong_month_sql" else "invalid_template")


@pytest.mark.parametrize("result", [
    "framed", "unknown_ref", "wrong_value", "length", "tool", "transport",
    "over_budget", "over_budget_after_length",
])
def test_failed_format_attempt_is_terminal_and_shares_the_token_budget(tmp_path, september_total, result):
    accepted = answer(september_total, SQL_REF)
    rejected = PROSE + "\n" + content_of(accepted)
    followup = {
        "framed": final_turn(rejected),
        "unknown_ref": answer(september_total, UNKNOWN_REF),
        "wrong_value": answer(september_total + 1, SQL_REF),
        "length": length_completion(),
        "tool": sql_turn("forbidden-repeat"),
        "transport": RuntimeError("offline transport interruption"),
        "over_budget": usage(accepted, 39_801),
        "over_budget_after_length": usage(accepted, 29_085),
    }[result]
    earlier = [length_completion()] if result == "over_budget_after_length" else []
    llm = script(*earlier, sql_turn(), final_turn(rejected), followup, accepted)
    service = service_at(tmp_path, llm)
    conversation = seed_conversation(service)
    response = post(service, conversation_id=conversation)

    exhausted = result.startswith("over_budget")
    assert response.status_code == (429 if exhausted else 502)
    assert response.json() == (BUDGET_EXCEEDED if exhausted else REJECTED)
    assert len(llm.calls) == 3 + len(earlier)
    assert_format_request(*llm.calls[-2:], rejected)
    assert_failed(service, conversation, failure_code=None if exhausted else "invalid_template")


def test_format_failure_inside_evidence_repair_does_not_open_another_attempt(tmp_path, september_total):
    correct = answer(september_total, SQL_REF)
    llm = script(sql_turn(), answer(september_total, UNKNOWN_REF),
                 final_turn(PROSE + "\n" + content_of(correct)), correct)
    service = service_at(tmp_path, llm)
    conversation = seed_conversation(service)
    response = post(service, conversation_id=conversation)

    assert response.status_code == 502 and response.json() == REJECTED
    assert len(llm.calls) == 3 and llm.calls[-1]["tools"] == []
    assert_failed(service, conversation, failure_code="unsupported_number")


def test_exhausted_call_allowance_preserves_original_format_rejection(tmp_path, monkeypatch, september_total):
    monkeypatch.setenv("LLM_MAX_CALLS_PER_REQUEST", "2")
    correct = answer(september_total, SQL_REF)
    llm = script(sql_turn(), final_turn(PROSE + "\n" + content_of(correct)), correct)
    service = service_at(tmp_path, llm)
    conversation = seed_conversation(service)
    response = post(service, conversation_id=conversation)

    assert response.status_code == 502 and response.json() == REJECTED
    assert len(llm.calls) == 2
    assert_failed(service, conversation)


@pytest.mark.parametrize("tenant", ["demo-alpha", "demo-beta"], ids=["unrelated_conversation", "foreign_tenant"])
def test_framed_reference_from_other_conversation_cannot_qualify(tmp_path, september_total, tenant):
    foreign_ref = "call_unrelated_format_evidence"
    owner = service_at(tmp_path, script(sql_turn(foreign_ref), answer(september_total, foreign_ref)))
    seed = post(owner)
    assert seed.status_code == 200
    original_id = seed.json()["conversation_id"]
    local_total = september_total if tenant == "demo-alpha" else paid_september_total("beta.db")
    draft = PROSE + "\n" + content_of(answer(local_total, foreign_ref))
    llm = script(sql_turn(), final_turn(draft), answer(local_total, SQL_REF))
    current = service_at(tmp_path, llm)
    if tenant == "demo-beta":
        assert post(current, tenant=tenant, conversation_id=original_id).status_code == 404
        assert llm.calls == []
    owner_context = current.state.secure_data.authenticate(tenant)
    new_id = current.state.conversation_store.create_conversation(owner_context)
    response = post(current, tenant=tenant, conversation_id=new_id)

    assert response.status_code == 502 and response.json() == REJECTED and len(llm.calls) == 2
    assert [entry["evidence_ref"] for entry in manifest(llm.calls[1]["messages"])["tools"]] == [SQL_REF]
    turn, = snapshot(current, tenant=tenant, conversation_id=new_id).turns
    assert (turn.status, turn.failure_code, turn.answer) == ("failed", "invalid_template", None)


@pytest.mark.parametrize("eligible", [True, False], ids=["recovered", "ambiguous"])
def test_private_diagnostics_link_format_rejection_and_answer_only_attempt(
    tmp_path, monkeypatch, mocked_wire, caplog, eligible,
):
    clean = content_of(structured_turn("پاسخ آماده است."))
    draft = PROSE + "\n" + clean
    if not eligible:
        draft += "\n" + content_of(structured_turn("پاسخ دیگری ثبت شد."))
    fresh = usage(structured_turn("پاسخ تازه ثبت شد."), 200)
    replies = [(200, encoded(usage(final_turn(draft), 100))), (200, encoded(fresh))]
    response, state, sent, path, conversation = capture_chat(tmp_path, monkeypatch, replies, enabled="1")
    recorded = read_capture(path)
    calls = recorded["provider_calls"]

    assert response.status_code == (200 if eligible else 502)
    assert len(sent) == len(calls) == (2 if eligible else 1)
    for call_record, outbound, (_, inbound) in zip(calls, sent, replies[:len(calls)], strict=True):
        assert body_bytes(call_record["request"]) == outbound
        assert body_bytes(call_record["response"]) == inbound
        assert call_record["request"]["headers"]["authorization"] == "[REDACTED]"
        request = json.loads(outbound)
        assert request["max_completion_tokens"] == 8192
        assert not {"reasoning", "reasoning_effort", "include_reasoning"} & request.keys()

    original = recorded["grounding"][0]
    assert original["raw_final_content"] == draft and original["rejection_reason"] == "invalid_template"
    assert original["slot_decisions"] == [] and original["rendered_answer"] is None
    report = original.get("format_repair")
    assert report is not None
    assert report["eligible"] is eligible and report["attempted"] is eligible
    assert original["provider_call_id"] == calls[0]["call_id"]
    assert report["recovery_provider_call_id"] == (calls[1]["call_id"] if eligible else None)
    assert report["outcome"] == ("success" if eligible else "failed")
    assert [json.loads(body_bytes(item["response"]))["usage"]["total_tokens"] for item in calls] == (
        [100, 200] if eligible else [100]
    )
    if eligible:
        assert_format_request(*(json.loads(body) for body in sent), draft)
        regenerated = recorded["grounding"][1]
        assert regenerated["provider_call_id"] == calls[1]["call_id"]
        assert regenerated["rejection_reason"] is None and regenerated["rendered_answer"] == response.json()["answer"]
    assert len(recorded["grounding"]) == (2 if eligible else 1)
    turn, = state.turns
    assert original["conversation_id"] == conversation and original["trace_id"] == turn.trace_id
    assert PROSE not in json.dumps(state.messages, ensure_ascii=False) + response.text + caplog.text
    assert recorded["tool_executions"] == [] and recorded["capture_errors"] == []
    assert DEMO_SECRET not in path.read_text(encoding="utf-8") + caplog.text
