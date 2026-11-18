"""Q3-D RED contracts for trusted three-month request context."""

import hashlib
import json

import pytest

from app.agent import UngroundedAnswerError
from app.schemas import ChatRequest
from test_agent_loop import ScriptedLLM, agent, call, final_turn, numeric_slot, structured_turn, tool_result, tool_turn
from test_evidence_reference_contract_red import manifest
from test_structured_answer_grounding_red import alpha, forbid_live_provider


QUESTION = "نمودار روند فروش پرداخت‌شده به تفکیک دسته محصول در اوت، سپتامبر و اکتبر ۲۰۲۶ را تولید کن."
PERIOD = "اوت، سپتامبر و اکتبر ۲۰۲۶"
START, END = "2026-08-01", "2026-10-31"
REQUEST_CONTEXT = {"period": PERIOD, "period_start": START, "period_end": END}

# The original assistant content from Q3-C, including both leading newlines.
CAPTURED_FINAL = """

{
  "template": "نمودار روند فروش پرداخت‌شدهٔ بنگاه {{tenant_label}} به تفکیک دستهٔ محصول برای دورهٔ درخواستی {{user_period}} (از {{period_start}} تا {{period_end}}) به‌صورت تصویر PNG تولید و ذخیره شد. در این نمودار، مجموع درآمدهای پرداخت‌شدهٔ هر دسته — hardware، software، services و subscription — در سه ماه به‌صورت خطوط مجزا نشان داده شده است. بر اساس داده‌های سفارش‌های پرداخت‌شده، درآمد هر سه ماهه به‌تدریج از نخستین ماه دوره تا ماه آخر افزایش می‌یابد و در ماه پایانی دوره، همهٔ دسته‌ها به سطحی نزدیک به یکدیگر می‌رسند.",
  "slots": {
    "tenant_label": {"source": "tenant_context", "value": "Alpha"},
    "user_period": {"source": "user_request", "value": "اوت، سپتامبر و اکتبر ۲۰۲۶"},
    "period_start": {"source": "user_request", "type": "date", "field": "period_start", "value": "2026-08-01"},
    "period_end": {"source": "user_request", "type": "date", "field": "period_end", "value": "2026-10-31"}
  }
}"""
CAPTURED_SHA256 = "cf986a89adff45557877122d4b0f0376c22bef900a8a748be2f37ba0398cdc74"


def test_exact_q3_final_reply_renders_with_trusted_request_slots(alpha):
    assert hashlib.sha256(CAPTURED_FINAL.encode("utf-8")).hexdigest() == CAPTURED_SHA256
    original = json.loads(CAPTURED_FINAL)
    expected = original["template"]
    for name, value in {
        "tenant_label": "Alpha", "user_period": PERIOD, "period_start": START, "period_end": END,
    }.items():
        expected = expected.replace("{{" + name + "}}", value)

    llm = ScriptedLLM(final_turn(CAPTURED_FINAL))
    response = agent(llm, tenant=alpha).run(ChatRequest(question=QUESTION))

    assert response.answer == expected
    assert manifest(llm.calls[0]["messages"])["context"]["user_request"] == REQUEST_CONTEXT


def test_explicit_english_three_months_advertise_request_label_and_inclusive_boundaries(alpha):
    label = "August, September, and October 2026"
    llm = ScriptedLLM(structured_turn("بازهٔ نمودار مشخص است."))
    agent(llm, tenant=alpha).run(ChatRequest(question=f"Plot paid revenue for {label}."))

    assert manifest(llm.calls[0]["messages"])["context"].get("user_request") == {
        "period": label, "period_start": START, "period_end": END,
    }


@pytest.mark.parametrize(("slot_name", "value"), [
    ("user_period", "سپتامبر، اکتبر و نوامبر ۲۰۲۶"),
    ("period_start", "2026-08-02"),
    ("period_end", "2026-11-01"),
    ("period_end", "2026-10-32"),
], ids=["invented_period", "altered_start", "exclusive_cutoff", "invalid_calendar_date"])
def test_authorized_three_month_context_still_rejects_altered_slots(alpha, slot_name, value):
    slot = json.loads(CAPTURED_FINAL)["slots"][slot_name]
    llm = ScriptedLLM(structured_turn("بازهٔ گزارش {{value}} است.", value={**slot, "value": value}))

    with pytest.raises(UngroundedAnswerError) as failure:
        agent(llm, tenant=alpha).run(ChatRequest(question=QUESTION))

    assert failure.value.reason_code == "unsupported_number"
    # Rejection must happen despite having a valid request scope.
    assert manifest(llm.calls[0]["messages"])["context"].get("user_request") == REQUEST_CONTEXT


@pytest.mark.parametrize("question", [
    "نمودار فروش اوت، سپتامبر و اکتبر را تولید کن.",
    "نمودار فروش اوت ۲۰۲۵، سپتامبر و اکتبر ۲۰۲۶ را تولید کن.",
    QUESTION + " فقط از 2026-09-01 تا 2026-10-31 شامل.",
    "عدد دقیق گزارش قبلی را دوباره بگو و نمودار اوت، سپتامبر و اکتبر ۲۰۲۶ را تولید کن.",
    "نمودار فروش اوت، سپتامبر و شهریور ۲۰۲۶ را تولید کن.",
], ids=["missing_year", "mixed_years", "conflicting_dates", "previous_report_conflict", "mixed_calendar"])
def test_ambiguous_requests_do_not_authorize_three_month_scope(alpha, question):
    llm = ScriptedLLM(structured_turn("بازهٔ گزارش نیاز به توضیح دارد."))
    agent(llm, tenant=alpha).run(ChatRequest(question=question))

    assert "user_request" not in manifest(llm.calls[0]["messages"])["context"]


@pytest.mark.parametrize("reference", ["q3-issued-sql", "q3-invented-sql"], ids=["unverified_projection", "unissued_id"])
def test_chart_period_cannot_authorize_unverified_numeric_evidence(alpha, reference):
    sql = (
        "SELECT substr(order_date,1,7) AS month, category, SUM(amount_toman) AS revenue FROM paid_orders "
        "WHERE order_date >= '2026-08-01' AND order_date < '2026-11-01' "
        "GROUP BY month, category ORDER BY month, category"
    )

    def cite_result(messages, tools):
        rows = tool_result(messages, "q3-issued-sql")["rows"]
        assert len(rows) == 12
        assert manifest(messages)["tools"] == []
        return structured_turn(
            "فروش دستهٔ محصول {{revenue}} تومان بود.",
            revenue={**numeric_slot(rows[0]["revenue"], reference, "revenue"), "row": 0},
        )

    llm = ScriptedLLM(tool_turn(call("query_sql", {"sql": sql}, "q3-issued-sql")), cite_result)
    with pytest.raises(UngroundedAnswerError) as failure:
        agent(llm, tenant=alpha).run(ChatRequest(question=QUESTION))

    assert failure.value.reason_code == "unsupported_number"
    assert len(llm.calls) == 2
