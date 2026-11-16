"""Phase 4-C-C.5-E-A. Gregorian request scope and date-slot boundaries."""

import re
from unittest.mock import Mock

import pytest

from app.agent import AgentLoop, UngroundedAnswerError
from app.month_scope import requested_month, sql_matches_month
from app.sandbox import DockerPythonSandbox
from app.schemas import ChatRequest
from test_agent_loop import SCHEMA, ScriptedLLM, call, numeric_slot, structured_turn, tool_result, tool_turn
from test_evidence_reference_contract_red import manifest
from test_grounding_date_regression import source_paid_total
from test_structured_answer_grounding_red import alpha, september_total


SEPTEMBER = ("2026-09-01", "2026-09-30", "2026-10-01", "2026-09")
START, LAST, NEXT, PERIOD = SEPTEMBER

# Exact user message from the first Python + PNG request, kept independent of
# the live test and the private diagnostic file.
CAPTURED_QUESTION = (
    "با execute_python در Docker برنامه Python بساز و اجرا کن. فقط از CSVهای "
    "مجاز همین مستأجر در /data بخوان. از /data/paid_orders.csv مجموع درآمد "
    "سفارش‌های paid را برای سپتامبر میلادی 2026 حساب کن "
    "(از 2026-09-01 شامل آن روز تا قبل از 2026-10-01). فقط با کتابخانه استاندارد "
    "Python و بدون نصب بسته، یک نمودار PNG معتبر و کوچک در "
    "/output/paid_revenue.png بساز. در stdout دقیقاً یک شیء JSON به شکل "
    '{"metrics":{"revenue":"<computed_total>"}} '
    "چاپ کن. مقدار revenue را از مجموع محاسبه‌شده به صورت رشته‌ای از رقم‌های صحیح "
    "پر کن و مقدار مورد انتظار را در کد ثابت ننویس. هیچ متن دیگری از جمله "
    "پیام اشکال‌زدایی را در stdout ننویس. سپس عدد دقیق تومان را در پاسخ فارسی "
    "با یک slot عددی با واحد scalar گزارش کن و برای فیلد revenue به شناسهٔ "
    "واقعی شاهد execute_python ارجاع بده."
)


@pytest.mark.parametrize("question", [
    "درآمد سپتامبر میلادی 2026 چقدر بود؟",
    "Paid revenue for September 2026?",
    "فروش ماه سپتامبرِ سال ۲۰۲۶ میلادی چقدر بود؟",
    "Paid revenue from 2026-09-01 through 2026-09-30 inclusive.",
    "Paid revenue on or after 2026-09-01 and before 2026-10-01.",
    "درآمد از ۲۰۲۶-۰۹-۰۱ تا پیش از ۲۰۲۶-۱۰-۰۱ را حساب کن.",
    CAPTURED_QUESTION,
], ids=[
    "persian_gregorian_label", "english_month", "persian_digits",
    "inclusive_iso_range", "exclusive_iso_range", "persian_exclusive_iso_range",
    "captured_python_prompt",
])
def test_september_request_resolves_to_one_full_gregorian_month(question):
    scope = requested_month(question)
    assert scope is not None
    assert (scope.first, scope.last, scope.next_first, scope.gregorian_month) == SEPTEMBER


@pytest.mark.parametrize("question", [
    "Paid revenue from 2026-09-02 through 2026-09-30 inclusive.",
    "Paid revenue from 2026-09-01 through 2026-10-01 inclusive.",
    "Paid revenue for September 2026, on or after 2026-10-01 and before 2026-11-01.",
    "Compare September 2026 with October 2026.",
], ids=["partial_month", "includes_october_first", "conflicting_name_and_range", "comparison"])
def test_ranges_other_than_one_whole_month_do_not_infer_september(question):
    assert requested_month(question) is None


def test_shahrivar_retains_distinct_solar_hijri_scope():
    scope = requested_month("درآمد شهریور ۱۴۰۵ شمسی چقدر بود؟")
    assert scope is not None
    assert (scope.first, scope.last, scope.next_first, scope.gregorian_month) == (
        "2026-08-23", "2026-09-22", "2026-09-23", None,
    )
    assert requested_month("سپتامبر میلادی 2026 و شهریور ۱۴۰۵ شمسی را مقایسه کن.") is None


@pytest.mark.parametrize(("operator", "end", "accepted"), [
    ("<", NEXT, True),
    ("<=", LAST, True),
    ("<", LAST, False),
    ("<=", NEXT, False),
], ids=["half_open", "inclusive", "drops_last_day", "includes_next_month"])
def test_sql_month_checks_keep_operators_and_cutoffs_paired(operator, end, accepted):
    scope = requested_month("September 2026")
    assert scope is not None
    sql = f"SELECT SUM(amount_toman) FROM paid_orders WHERE order_date >= ? AND order_date {operator} ?"
    assert sql_matches_month(sql, [START, end], scope) is accepted


def run_agent(alpha, llm, question="September 2026 revenue", *, sandbox=None):
    return AgentLoop(llm=llm, tenant=alpha, schema_text=SCHEMA, sandbox=sandbox).run(ChatRequest(question=question))


def request_date(field, value):
    return {"source": "user_request", "type": "date", "field": field, "value": value}


def test_agent_guidance_separates_exclusive_filtering_from_inclusive_slots(alpha):
    llm = ScriptedLLM(structured_turn("بازهٔ مورد درخواست مشخص است."))
    run_agent(alpha, llm, "September 2026 revenue and chart", sandbox=Mock(spec=DockerPythonSandbox))
    messages = llm.calls[0]["messages"]
    assert "execute_python" in {tool["function"]["name"] for tool in llm.calls[0]["tools"]}
    assert manifest(messages)["context"]["user_request"] == {
        "period": PERIOD, "period_start": START, "period_end": LAST,
    }
    guidance = "\n".join(line for line in messages[0]["content"].splitlines() if not line.startswith("KAMIDSO_"))
    sentences = re.split(r"(?<=[.!?])\s+|\n", guidance)
    filters = [sentence for sentence in sentences if "SQL" in sentence and "Python" in sentence]
    assert any(
        re.search(r">=\s*2026-09-01\b", sentence) and re.search(r"(?<![<=>])<\s*2026-10-01\b", sentence)
        for sentence in filters
    ), "Both SQL and Python need an explicit inclusive start and exclusive cutoff"
    assert any(
        "user_request" in sentence and "period_end" in sentence and LAST in sentence
        and "inclusive" in sentence.lower()
        for sentence in sentences
    ), "The inclusive final day must be tied explicitly to user_request period_end"


@pytest.mark.parametrize(("field", "value"), [
    ("period_end", NEXT),
    ("period_end", "2026-09-31"),
    ("period_end", "2026-9-30"),
    ("period_end", "2026-09-22"),
    ("period_start", "2026-09-02"),
    ("next_first", NEXT),
], ids=["exclusive_cutoff", "invalid_date", "malformed_iso", "shahrivar_end", "wrong_start", "unknown_field"])
def test_date_slots_reject_exclusive_malformed_and_mismatched_values(alpha, field, value):
    llm = ScriptedLLM(structured_turn("بازه با تاریخ {{date}} مشخص می‌شود.", date=request_date(field, value)))
    with pytest.raises(UngroundedAnswerError) as failure:
        run_agent(alpha, llm)
    assert failure.value.reason_code == "unsupported_number"


def test_captured_question_grounds_inclusive_dates_with_half_open_sql(alpha, september_total):
    ref = "verified-september-sql"
    sql = "SELECT SUM(amount_toman) AS revenue FROM paid_orders WHERE order_date >= ? AND order_date < ?"
    llm = ScriptedLLM(
        tool_turn(call("query_sql", {"sql": sql, "params": [START, NEXT]}, ref)),
        structured_turn(
            "  گزارش من از {{start}} تا {{end}}، با درآمد {{revenue}} تومان.  ",
            start=request_date("period_start", START),
            end=request_date("period_end", "۲۰۲۶-۰۹-۳۰"),
            revenue=numeric_slot(september_total, ref, "revenue"),
        ),
    )
    response = run_agent(alpha, llm, CAPTURED_QUESTION)
    assert response.answer == f"  گزارش من از {START} تا {LAST}، با درآمد {september_total} تومان.  "
    assert [(item.tool_name, item.status) for item in response.trace] == [("query_sql", "success")]
    assert manifest(llm.calls[0]["messages"])["context"]["user_request"] == {
        "period": PERIOD, "period_start": START, "period_end": LAST,
    }
    followup = llm.calls[1]["messages"]
    assert tool_result(followup, ref) == {"rows": [{"revenue": september_total}]}
    assert manifest(followup)["verified_periods"][ref] == {"start": START, "end": LAST}


def test_captured_question_rejects_shahrivar_sql_even_when_total_matches(alpha, september_total):
    wrong_start, wrong_cutoff = "2026-08-23", "2026-09-23"
    assert source_paid_total(wrong_start, wrong_cutoff) == september_total
    ref = "wrong-month-sql"
    sql = "SELECT SUM(amount_toman) AS revenue FROM paid_orders WHERE order_date >= ? AND order_date < ?"
    llm = ScriptedLLM(
        tool_turn(call("query_sql", {"sql": sql, "params": [wrong_start, wrong_cutoff]}, ref)),
        structured_turn("درآمد پرداخت‌شده {{revenue}} تومان بود.",
                        revenue=numeric_slot(september_total, ref, "revenue")),
    )
    with pytest.raises(UngroundedAnswerError) as failure:
        run_agent(alpha, llm, CAPTURED_QUESTION)
    assert failure.value.reason_code == "invalid_date_range"
    assert tool_result(llm.calls[1]["messages"], ref) == {"rows": [{"revenue": september_total}]}
