"""Q1-A RED: authorize explicit comparison periods without weakening evidence checks."""

from decimal import Decimal
from uuid import uuid4

import pytest

from app.agent import UngroundedAnswerError
from app.schemas import ChatRequest
from test_agent_loop import ScriptedLLM, agent, call, structured_turn, tool_result, tool_turn
from test_evidence_reference_contract_red import manifest
from test_followup_period_context_red import period_slot, request_date, sandbox_result
from test_structured_answer_grounding_red import alpha, forbid_live_provider, numeric, run, source_paid_orders


QUESTION = "فروش پرداخت‌شده در سپتامبر و اکتبر\n۲۰۲۶ را مقایسه کن.\nمبلغ و درصد تغییر را بگو."
PERIOD = "سپتامبر و اکتبر ۲۰۲۶"
TEMPLATE = (
    "در بازه‌ی {{user_period}} (از {{period_start}} تا {{period_end}}) فروش پرداخت‌شده به این صورت "
    "مقایسه شد: در ماه سپتامبر {{sept_revenue}} تومان و در ماه اکتبر {{oct_revenue}} تومان. با مقایسه‌ی "
    "این دو ماه، فروش اکتبر نسبت به سپتامبر {{change_amount}} تومان بیشتر شد؛ به بیان دیگر تغییر فروش "
    "از سپتامبر به اکتبر {{change_percent}} درصد افزایش است."
)
MONTHS = (
    ("2026-09-01", "2026-10-01"),
    ("2026-10-01", "2026-11-01"),
)


def paid_sql(start, cutoff):
    return (
        "SELECT SUM(amount_toman) AS revenue FROM paid_orders "
        f"WHERE order_date >= '{start}' AND order_date < '{cutoff}'"
    )


@pytest.fixture
def q1_replay():
    # Keep the captured question, template and slot contract portable. Derive
    # values independently and issue fresh IDs for the scripted provider calls.
    orders = source_paid_orders()
    september, october = (
        sum(amount for date, _, amount in orders if start <= date < cutoff) for start, cutoff in MONTHS
    )
    change = october - september
    percent = str(float((Decimal(change) * 100 / september).quantize(Decimal("0.01"))))
    metrics = {"change_amount": str(change), "change_percent": percent}
    sept_ref, oct_ref, python_ref = (uuid4().hex for _ in range(3))

    def calculate(messages, tools):
        for ref, expected in ((sept_ref, september), (oct_ref, october)):
            assert tool_result(messages, ref) == {"rows": [{"revenue": expected}]}
        return tool_turn(call("execute_python", {"code": "print(calculation_json)"}, python_ref))

    def check_calculation(messages, tools):
        result = tool_result(messages, python_ref)
        assert result["exit_code"] == 0 and result["stderr"] == ""
        assert result["stdout"] == sandbox_stdout

    sandbox = sandbox_result(**metrics)
    sandbox_stdout = sandbox.execute.return_value.stdout
    calls = (
        tool_turn(call("get_schema", {}, uuid4().hex)),
        tool_turn(*(
            call("query_sql", {"sql": paid_sql(*window)}, ref)
            for window, ref in zip(MONTHS, (sept_ref, oct_ref), strict=True)
        )),
        calculate,
    )
    slots = {
        "user_period": period_slot(PERIOD),
        "period_start": request_date("period_start", "2026-09-01"),
        "period_end": request_date("period_end", "2026-10-31"),
        "sept_revenue": numeric(str(september), sept_ref, "revenue"),
        "oct_revenue": numeric(str(october), oct_ref, "revenue"),
        "change_amount": numeric(metrics["change_amount"], python_ref, "change_amount", "scalar"),
        "change_percent": numeric(percent, python_ref, "change_percent", "scalar"),
    }
    return calls, sandbox, slots, check_calculation


def test_q1_renders_all_seven_captured_slots(alpha, q1_replay):
    calls, sandbox, slots, check_calculation = q1_replay

    def final_answer(messages, tools):
        check_calculation(messages, tools)
        return structured_turn(TEMPLATE, **slots)

    response = run(alpha, *calls, final_answer, question=QUESTION, sandbox=sandbox)

    expected = TEMPLATE
    for name, slot in slots.items():
        value = slot["value"]
        if slot.get("type") == "numeric":
            value = format(Decimal(value).normalize(), "f")
        expected = expected.replace("{{" + name + "}}", value)
    assert response.answer == expected
    assert [step.tool_name for step in response.trace] == [
        "get_schema", "query_sql", "query_sql", "execute_python",
    ]


def test_another_comparison_advertises_a_trusted_request_period(alpha):
    question = "درآمد اوت و سپتامبر ۲۰۲۶ را مقایسه کن."
    llm = ScriptedLLM(structured_turn("مقایسه آماده است."))
    agent(llm, tenant=alpha).run(ChatRequest(question=question))

    request_scope = manifest(llm.calls[0]["messages"])["context"].get("user_request")
    assert request_scope is not None
    assert {key: request_scope[key] for key in ("period", "period_start", "period_end")} == {
        "period": "اوت و سپتامبر ۲۰۲۶",
        "period_start": "2026-08-01",
        "period_end": "2026-09-30",
    }


def test_existing_sql_and_python_evidence_already_grounds_all_four_numeric_slots(alpha, q1_replay):
    calls, sandbox, slots, check_calculation = q1_replay
    amounts = {key: slot for key, slot in slots.items() if slot.get("type") == "numeric"}
    template = "فروش سپتامبر {{sept_revenue}}، فروش اکتبر {{oct_revenue}}، افزایش {{change_amount}} و درصد {{change_percent}} بود."

    def final_answer(messages, tools):
        check_calculation(messages, tools)
        return structured_turn(template, **amounts)

    answer = run(alpha, *calls, final_answer, question=QUESTION, sandbox=sandbox)
    expected = [format(Decimal(slot["value"]).normalize(), "f") for slot in amounts.values()]
    assert answer.answer == (
        f"فروش سپتامبر {expected[0]}، فروش اکتبر {expected[1]}، افزایش {expected[2]} و درصد {expected[3]} بود."
    )


@pytest.mark.parametrize("case", [
    "invented-period", "shifted-start", "exclusive-end", "invented-amount",
    "invented-percentage", "invented-reference", "wrong-calculation-field", "wrong-unit", "foreign-tenant",
])
def test_q1_rejects_each_invalid_claim_in_isolation(alpha, q1_replay, case):
    calls, sandbox, slots, _ = q1_replay
    bad_slots = {
        "invented-period": period_slot("اوت و اکتبر ۲۰۲۶"),
        "shifted-start": request_date("period_start", "2026-09-02"),
        "exclusive-end": request_date("period_end", "2026-11-01"),
        "invented-amount": {**slots["sept_revenue"], "value": str(int(slots["sept_revenue"]["value"]) + 1)},
        "invented-percentage": {**slots["change_percent"], "value": str(Decimal(slots["change_percent"]["value"]) + 1)},
        "invented-reference": {**slots["sept_revenue"], "evidence_ref": uuid4().hex},
        "wrong-calculation-field": {**slots["change_amount"], "evidence_field": "change_percent"},
        "wrong-unit": {**slots["change_percent"], "unit": "percent"},
        "foreign-tenant": {"source": "tenant_context", "value": "Beta"},
    }
    bad_answer = structured_turn("مقدار گزارش {{claim}} است.", claim=bad_slots[case])
    # The model repeats the same bad answer if the existing repair path is eligible.
    with pytest.raises(UngroundedAnswerError) as rejected:
        run(alpha, *calls, bad_answer, bad_answer, question=QUESTION, sandbox=sandbox)
    assert rejected.value.reason_code == "unsupported_number"


@pytest.mark.parametrize(
    ("question", "start", "cutoff"),
    [
        (QUESTION, "2026-08-01", "2026-09-01"),
        ("درآمد سپتامبر ۲۰۲۶ چقدر بود؟", "2026-10-01", "2026-11-01"),
    ],
    ids=["outside-comparison", "outside-single-month"],
)
def test_an_actual_sql_result_for_an_unrequested_month_cannot_ground_the_answer(
    alpha, question, start, cutoff,
):
    found = sum(amount for date, _, amount in source_paid_orders() if start <= date < cutoff)
    ref = uuid4().hex
    call_sql = tool_turn(call("query_sql", {"sql": paid_sql(start, cutoff)}, ref))

    def final_answer(messages, tools):
        assert tool_result(messages, ref) == {"rows": [{"revenue": found}]}
        return structured_turn("درآمد {{revenue}} تومان بود.", revenue=numeric(str(found), ref, "revenue"))

    with pytest.raises(UngroundedAnswerError) as rejected:
        run(alpha, call_sql, final_answer, question=question)
    assert rejected.value.reason_code == "invalid_date_range"
