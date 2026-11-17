"""Phase 4-C-C.5-U-A. Exercise the template contract on mocked provider requests."""

import json

import pytest
from fastapi.testclient import TestClient

from app.agent import EnvChatCompletionsClient
from test_agent_loop import final_turn, numeric_slot, structured_turn
from test_canonical_evidence_citation_red import (
    ALIASES, CANONICAL, FOREIGN_REF, RULES_PREFIX, assert_canonical_manifest, report, restarted_report,
)
from test_evidence_reference_contract_red import EVIDENCE_PREFIX, manifest
from test_evidence_reference_repair_red import assert_rejected, service_at, snapshot, sql_turn, usage
from test_final_json_format_recovery_red import PROSE, assert_format_request, content_of
from test_followup_period_context_red import CAPTURED_QUESTION, SEPTEMBER_CONTEXT, period_slot, request_date
from test_length_completion_recovery_red import encoded, isolated_settings, length_completion, mocked_wire
from test_llm_io_diagnostics_red import fake_provider
from test_python_png_accepted_response_red import OWNER
from test_python_sandbox import docker_ready
from test_structured_answer_grounding_red import PAID_SQL, september_total


TEMPLATE_PREFIX = "KAMIDSO_TEMPLATE_RULES_V1 "
PHASE_T_REF = "b575ac83-6846-92db-8a59-b8bea11ad4fb_0"
# Exact model content from the Phase T follow-up capture, independent of local capture availability.
PHASE_T_DRAFT = """{
  "template": "مبلغ دقیق گزارش قبلی مجدداً اعلام می‌شود: مجموع درآمد سفارش‌های پرداخت‌شدهٔ سپتامبر 2026 برابر است با {{revenue}} تومان، و نمودار قبلی بدون تغییر در مسیر /output/paid_revenue.png حفظ شده است.",
  "slots": {
    "revenue": {
      "type": "numeric",
      "value": "53250000",
      "evidence_ref": "b575ac83-6846-92db-8a59-b8bea11ad4fb_0",
      "evidence_field": "revenue",
      "unit": "scalar"
    }
  }
}"""


def assert_template_rules(request):
    system = request["messages"][0]["content"]
    lines = system.splitlines()
    for prefix in (RULES_PREFIX, EVIDENCE_PREFIX):
        assert sum(line.startswith(prefix) for line in lines) == 1
    rules = [line.removeprefix(TEMPLATE_PREFIX) for line in lines if line.startswith(TEMPLATE_PREFIX)]
    assert len(rules) == 1, "Expected exactly one explicit KAMIDSO_TEMPLATE_RULES_V1 instruction"
    text = rules[0].lower()
    for part in (
        "exactly one json object", "template", "slots", "no literal numeric characters",
        "years", "dates", "amounts", "counts", "percentages", "numeric identifiers", "unicode",
        "{{user_period}}", "{{period_start}}", "{{period_end}}", "{{revenue}}",
        "each placeholder", "exactly one grounded slot", "never invent slot values", "user_request",
        "verified tool evidence",
        "malformed placeholders", "unsupported braces", "omit unsupported claims",
    ):
        assert part in text
    assert "<actual_tool_call_id>" not in text
    assert not any(alias in rules[0] for alias in ALIASES)
    return rules[0]


def test_ordinary_request_contains_explicit_template_rules(tmp_path, monkeypatch, mocked_wire):
    month = SEPTEMBER_CONTEXT["period"]
    valid = structured_turn("دورهٔ {{user_period}}", user_period=period_slot(month))
    requests = fake_provider(monkeypatch, [(200, encoded(usage(valid, 100)))])
    service = service_at(tmp_path, EnvChatCompletionsClient())
    with TestClient(service) as client:
        response = client.post("/chat", headers=OWNER, json={"question": CAPTURED_QUESTION})
    assert response.status_code == 200 and response.json()["answer"] == f"دورهٔ {month}"
    assert snapshot(service, response).turns[0].status == "success"
    raw, = requests
    request = json.loads(raw)
    assert manifest(request["messages"])["context"]["user_request"] == SEPTEMBER_CONTEXT
    assert_template_rules(request)


def test_restarted_followup_contains_explicit_template_rules(restarted_report, september_total):
    run = restarted_report(report(september_total, CANONICAL))
    assert run.response.status_code == 200
    request, = run.requests
    assert_canonical_manifest(request, CANONICAL)
    assert_template_rules(request)


@pytest.mark.parametrize("recovery", ["evidence", "format", "length"])
def test_template_rules_are_preserved_during_existing_recovery(restarted_report, september_total, recovery):
    valid = report(september_total, CANONICAL)
    if recovery == "evidence":
        rejected = report(september_total, ALIASES[0])
    elif recovery == "format":
        rejected = final_turn(PROSE + "\n\n" + content_of(valid))
    else:
        rejected = length_completion()

    run = restarted_report(rejected, valid)
    assert run.response.status_code == 200
    assert run.state.turns[-1].status == "success"
    initial, recovered = run.requests
    for request in (initial, recovered):
        assert_canonical_manifest(request, CANONICAL)
        assert request["max_completion_tokens"] == 8192
    assert manifest(initial["messages"]) == manifest(recovered["messages"])
    assert not run.capture["tool_executions"]
    if recovery == "length":
        length, = run.capture["length_recovery"]
        assert length["attempts"] == 1 and length["recovery_finish_reason"] == "stop"
        assert initial["tools"] == recovered["tools"] and recovered["tools"]
    else:
        assert recovered["tools"] == [] and recovered["messages"][1:] == initial["messages"][1:]
        first, second = run.capture["grounding"]
        flag = "repair" if recovery == "evidence" else "format_repair"
        assert first[flag]["attempted"] is True and second["rejection_reason"] is None
        assert second["raw_final_content"] == content_of(valid)

    assert assert_template_rules(recovered) == assert_template_rules(initial)


def assert_terminal(run, draft, *, code="invalid_template", original_ref=CANONICAL):
    assert_rejected(run.response)
    assert len(run.requests) == 1
    assert [turn.status for turn in run.state.turns] == ["success", "failed"]
    assert run.state.turns[-1].failure_code == code and run.state.turns[-1].answer is None
    assert [item.call_id for item in run.state.evidence] == [original_ref]
    observed, = run.capture["grounding"]
    assert observed["rejection_reason"] == code and observed["raw_final_content"] == content_of(draft)
    assert observed["rendered_answer"] is None and content_of(draft) not in run.transcript
    assert not any(observed.get(key, {}).get("attempted") for key in ("format_repair", "repair"))
    assert not run.capture["tool_executions"]
    return observed


def test_phase_t_complete_json_literal_year_fails_before_slot_evaluation(
    restarted_report, september_total, docker_ready,
):
    assert int(json.loads(PHASE_T_DRAFT)["slots"]["revenue"]["value"]) == september_total
    draft = final_turn(PHASE_T_DRAFT)
    run = restarted_report(draft, refs=(PHASE_T_REF,), real_docker=True)
    assert_canonical_manifest(run.requests[0], PHASE_T_REF)
    observed = assert_terminal(run, draft, original_ref=PHASE_T_REF)
    assert observed["format_repair"]["eligible"] is False
    slot, = observed["slot_decisions"]
    assert slot["evidence_ref"] == PHASE_T_REF and slot["decision"] == "not_evaluated"


def test_trusted_period_date_and_exact_python_slots_render_after_restart(restarted_report, september_total):
    ctx = SEPTEMBER_CONTEXT
    answer = structured_turn(
        "دورهٔ {{user_period}} از {{period_start}} تا {{period_end}}، درآمد {{revenue}} تومان بود.",
        user_period=period_slot(ctx["period"]),
        period_start=request_date("period_start", ctx["period_start"]),
        period_end=request_date("period_end", ctx["period_end"]),
        revenue=numeric_slot(september_total, CANONICAL, "revenue", "scalar"),
    )
    run = restarted_report(answer)
    assert run.response.status_code == 200 and len(run.requests) == 1
    rendered = f"دورهٔ {ctx['period']} از {ctx['period_start']} تا {ctx['period_end']}، درآمد {september_total} تومان بود."
    assert run.response.json()["answer"] == run.state.turns[-1].answer == rendered
    assert run.state.turns[-1].status == "success"
    grounding, = run.capture["grounding"]
    assert grounding["raw_final_content"] == content_of(answer) and grounding["rejection_reason"] is None
    assert {slot["slot"]: slot["decision"] for slot in grounding["slot_decisions"]} == {
        name: "accepted" for name in ("user_period", "period_start", "period_end", "revenue")
    }


@pytest.mark.parametrize("literal", ["۲۰۲۶", "٢٠٢٦", "Ⅸ", "²", "2026-09-01", "53,250,000", "12%", "AT042"])
def test_numeric_characters_in_literal_spans_are_terminal(restarted_report, september_total, literal):
    draft = structured_turn(
        f"گزارش {literal} با درآمد {{{{revenue}}}} تومان.",
        revenue=numeric_slot(september_total, CANONICAL, "revenue", "scalar"),
    )
    run = restarted_report(draft)
    observed = assert_terminal(run, draft)
    assert observed["slot_decisions"][0]["decision"] == "not_evaluated"


@pytest.mark.parametrize("defect", ["missing", "extra", "malformed", "nonslot-braces", "unsupported-source"])
def test_placeholder_and_source_errors_remain_terminal(restarted_report, september_total, defect):
    revenue = numeric_slot(september_total, CANONICAL, "revenue", "scalar")
    template = "گزارش {{revenue}} تومان است."
    slots = {"revenue": revenue}
    code = "invalid_template"
    if defect == "missing":
        slots.clear()
    elif defect == "extra":
        slots["extra"] = revenue
    elif defect == "malformed":
        template = "گزارش {{revenue} تومان است."
    elif defect == "nonslot-braces":
        template = "گزارش {ناشناخته} با درآمد {{revenue}} تومان است."
    else:
        slots["revenue"] = {"source": "model", "value": str(september_total)}
        code = "unsupported_number"
    draft = structured_turn(template, **slots)
    assert_terminal(restarted_report(draft), draft, code=code)


def test_framing_recovery_cannot_promote_an_invalid_numeric_literal(restarted_report, september_total):
    complete = structured_turn(
        "در سپتامبر 2026، درآمد {{revenue}} تومان بود.",
        revenue=numeric_slot(september_total, CANONICAL, "revenue", "scalar"),
    )
    draft = final_turn(PROSE + "\n\n" + content_of(complete))
    observed = assert_terminal(restarted_report(draft), draft)
    assert observed["format_repair"]["eligible"] is False


@pytest.mark.parametrize("defect", ["foreign-id", "wrong-unit", "wrong-field", "wrong-period"])
def test_well_formed_template_still_requires_scoped_values(restarted_report, september_total, defect):
    ref = FOREIGN_REF if defect == "foreign-id" else CANONICAL
    field = "unverified" if defect == "wrong-field" else "revenue"
    unit = "toman" if defect == "wrong-unit" else "scalar"
    period = "2026-10" if defect == "wrong-period" else SEPTEMBER_CONTEXT["period"]
    draft = structured_turn(
        "در {{user_period}}، درآمد {{revenue}} تومان بود.",
        user_period=period_slot(period),
        revenue=numeric_slot(september_total, ref, field, unit),
    )
    run = restarted_report(draft, foreign=defect == "foreign-id")
    assert_terminal(run, draft, code="unsupported_number")


def test_wrong_month_sql_stays_terminal_and_its_result_is_not_promoted(restarted_report, september_total):
    # Change the entire window at once to retain a valid half-open interval.
    sql = PAID_SQL.replace("2026-10-01", "2026-11-01").replace("2026-09-01", "2026-10-01")
    run = restarted_report(sql_turn(ref="wrong-month-sql", sql=sql), report(september_total, CANONICAL))
    assert_rejected(run.response)
    assert len(run.requests) == 2
    assert run.state.turns[-1].status == "failed" and run.state.turns[-1].failure_code == "invalid_date_range"
    assert [item.call_id for item in run.state.evidence] == [CANONICAL]
    assert len(run.capture["tool_executions"]) == 1 and run.capture["tool_executions"][0]["status"] == "success"


def test_length_recovery_keeps_normal_tools_and_stops_after_one_attempt(restarted_report):
    run = restarted_report(length_completion(), length_completion())
    assert run.response.status_code == 503
    initial, recovered = run.requests
    assert initial["tools"] == recovered["tools"] and initial["tools"]
    result, = run.capture["length_recovery"]
    assert result["attempts"] == 1 and result["recovery_finish_reason"] == "length"
    assert run.state.turns[-1].status == "failed" and not run.capture["tool_executions"]


def test_format_regeneration_consumes_the_existing_shared_token_budget(restarted_report, september_total):
    accepted = report(september_total, CANONICAL)
    rejected = PROSE + "\n\n" + content_of(accepted)
    run = restarted_report(usage(final_turn(rejected), 39_999), usage(accepted, 1))
    assert run.response.status_code == 429 and run.response.json() == {"detail": "Request budget exceeded"}
    initial, recovered = run.requests
    assert_format_request(initial, recovered, rejected)
    assert run.state.turns[-1].status == "failed" and run.state.turns[-1].answer is None
    assert [item.call_id for item in run.state.evidence] == [CANONICAL]
    observed, = run.capture["grounding"]
    assert observed["format_repair"]["attempted"] is True
