"""Phase 4-C-C.2-A RED: numeric grounding with scripted LLM turns only.

SQL and tickets use the original tenant-scoped sources. The Python tool uses a
test double; this module never calls OpenRouter or runs generated Python.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from app.agent import AgentLoop, EnvChatCompletionsClient, UngroundedAnswerError, _ground_answer
from app.schemas import ChatRequest
from app.secure_data import SecureDataAccess
from test_agent_loop import ScriptedLLM, call, final_turn, tool_result, tool_turn


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TOKEN_MAP = json.loads((ROOT / "demo_tokens.json.example").read_text(encoding="utf-8"))
SCHEMA = (DATA / "schema.md").read_text(encoding="utf-8")
QUESTION = "برای مشتری A001 درآمد paid سپتامبر ۲۰۲۶ میلادی چقدر بود؟"
PAID_SQL = (
    "SELECT SUM(amount_toman) AS revenue FROM paid_orders "
    "WHERE order_date >= '2026-09-01' AND order_date < '2026-10-01'"
)


@pytest.fixture(autouse=True)
def no_live_provider(monkeypatch):
    def reject_provider_call(self, *, messages, tools):
        pytest.fail("Offline grounding tests must not call a live provider")

    monkeypatch.setattr(EnvChatCompletionsClient, "create", reject_provider_call)


def alpha_agent(llm, *, sandbox=None):
    tenant = SecureDataAccess(data_dir=DATA, token_map=TOKEN_MAP).authenticate("demo-alpha")
    return AgentLoop(llm=llm, tenant=tenant, schema_text=SCHEMA, sandbox=sandbox)


def paid_total():
    uri = f"file:{(DATA / 'alpha.db').resolve().as_posix()}?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        return connection.execute(
            "SELECT SUM(amount_toman) FROM orders WHERE status = 'paid' "
            "AND order_date >= '2026-09-01' AND order_date < '2026-10-01'"
        ).fetchone()[0]


def run_paid_answer(answer, *, sql=PAID_SQL, question=QUESTION):
    llm = ScriptedLLM(
        tool_turn(call("query_sql", {"sql": sql}, "sql-1")),
        final_turn(answer),
    )
    result = alpha_agent(llm).run(ChatRequest(question=question))
    assert tool_result(llm.calls[1]["messages"], "sql-1")["rows"]
    return result


@pytest.mark.parametrize(
    "amount_text",
    [
        "53,250,000",
        "۵۳٬۲۵۰٬۰۰۰",
        "۵۳.۲۵۰.۰۰۰",
        "53,250,000.00",
        "۵۳٬۲۵۰٬۰۰۰٫۰۰",
    ],
    ids=("english_grouped", "persian_grouped", "persian_dot_grouped",
         "english_zero_decimal", "persian_zero_decimal"),
)
def test_real_paid_sql_amount_accepts_equivalent_numeric_formats(amount_text):
    assert paid_total() == 53_250_000
    answer = f"درآمد پرداخت‌شده سپتامبر ۲۰۲۶ برابر {amount_text} تومان بود."
    assert run_paid_answer(answer).answer == answer


def test_real_paid_sql_rejects_unsupported_revenue_amount():
    with pytest.raises(UngroundedAnswerError) as caught:
        run_paid_answer("درآمد پرداخت‌شده سپتامبر ۲۰۲۶ برابر ۵۳٬۲۵۰٬۰۰۱ تومان بود.")
    assert caught.value.reason_code == "unsupported_number"


@pytest.mark.parametrize("amount_text", ["۱۲٫۵۰", "12.50"], ids=("persian", "english"))
def test_evidenced_percentage_accepts_equivalent_decimal_precision(amount_text):
    _ground_answer(
        f"نرخ پرداخت {amount_text} درصد است.", "نرخ پرداخت چقدر است؟",
        [{"rows": [{"paid_rate_pct": 12.5}]}], set(), [],
    )


def test_evidenced_percentage_accepts_matching_persian_decimal():
    _ground_answer(
        "نرخ پرداخت ۱۲٫۵ درصد است.", "نرخ پرداخت چقدر است؟",
        [{"rows": [{"paid_rate_pct": 12.5}]}], set(), [],
    )


def test_decimal_percentage_cannot_be_assembled_from_unrelated_integers():
    with pytest.raises(UngroundedAnswerError) as caught:
        _ground_answer(
            "نرخ پرداخت ۱۲٫۵ درصد است.", "نرخ پرداخت چقدر است؟",
            [{"rows": [{"paid_count": 12, "unpaid_count": 5}]}], set(), [],
        )
    assert caught.value.reason_code == "unsupported_number"


def test_ticket_count_cannot_be_relabelled_as_paid_revenue():
    with pytest.raises(UngroundedAnswerError) as caught:
        _ground_answer(
            "درآمد پرداخت‌شده ۲ تومان بود.", "درآمد پرداخت‌شده چقدر بود؟",
            [{"rows": [{"revenue": 53_250_000, "ticket_count": 2}]}], set(), [],
        )
    assert caught.value.reason_code == "unsupported_number"


def test_valid_gregorian_sql_literal_dates_can_be_reported():
    assert paid_total() == 53_250_000
    answer = (
        "درآمد پرداخت‌شده از ۲۰۲۶-۰۹-۰۱ تا ۲۰۲۶-۰۹-۳۰ "
        "برابر ۵۳٬۲۵۰٬۰۰۰ تومان بود."
    )
    assert run_paid_answer(answer).answer == answer


def test_date_cannot_be_assembled_from_unrelated_year_month_day():
    with pytest.raises(UngroundedAnswerError) as caught:
        _ground_answer(
            "تاریخ گزارش ۲۰۲۶-۰۹-۳۰ بود.", "تاریخ گزارش چه بود؟",
            [{"rows": [{"year": 2026, "month": 9, "day": 30}]}], set(), [],
        )
    assert caught.value.reason_code == "unsupported_number"


def test_wrong_gregorian_month_still_rejected_when_paid_total_matches():
    wrong_sql = (
        "SELECT SUM(amount_toman) AS revenue FROM paid_orders "
        "WHERE order_date >= '2026-08-23' AND order_date < '2026-09-23'"
    )
    assert paid_total() == 53_250_000
    with pytest.raises(UngroundedAnswerError) as caught:
        run_paid_answer("درآمد سپتامبر ۲۰۲۶ برابر ۵۳٬۲۵۰٬۰۰۰ تومان بود.", sql=wrong_sql)
    assert caught.value.reason_code == "invalid_date_range"


def run_ticket_answer(answer, *, question="تیکت‌های مشتری A001 را بررسی کن"):
    llm = ScriptedLLM(
        tool_turn(call("get_tickets", {"customer_id": "A001"}, "tickets-1")),
        final_turn(answer),
    )
    response = alpha_agent(llm).run(ChatRequest(question=question))
    tickets = tool_result(llm.calls[1]["messages"], "tickets-1")["tickets"]
    assert {ticket["ticket_id"] for ticket in tickets} == {"AT001", "AT002"}
    return response


def test_tenant_ticket_count_is_supported_by_actual_retrieval():
    answer = "برای مشتری A001 تعداد ۲ تیکت ثبت شده است."
    assert run_ticket_answer(answer).answer == answer


def test_tenant_ticket_identifier_and_date_are_supported():
    answer = "تیکت AT001 در ۲۰۲۶-۱۰-۰۳ ثبت شده است."
    assert run_ticket_answer(answer).answer == answer


@pytest.mark.parametrize("answer", [
    "تیکت AT999 برای مشتری A001 ثبت شده است.",
    "تیکت AT001 برای مشتری A999 ثبت شده است.",
], ids=("invented_ticket_id", "invented_customer_id"))
def test_ticket_and_customer_identifiers_must_be_evidenced(answer):
    with pytest.raises(UngroundedAnswerError) as caught:
        run_ticket_answer(answer)
    assert caught.value.reason_code == "unsupported_number"


def test_untrusted_ticket_text_cannot_ground_invented_revenue():
    with pytest.raises(UngroundedAnswerError) as caught:
        run_ticket_answer("طبق تیکت، درآمد پرداخت‌شده ۹۹۹ تومان بود.")
    assert caught.value.reason_code == "unsupported_number"


class PythonResultStub:
    def execute(self, code, tenant):
        assert code == "print('53,250,000')"
        assert tenant.state_key
        return type("Execution", (), {
            "stdout": "53,250,000\n", "stderr": "", "exit_code": 0,
            "artifacts": [],
        })()


@pytest.mark.parametrize(("amount_text", "expected_success"), [
    ("۵۳٬۲۵۰٬۰۰۰", True),
    ("۵۳٬۲۵۰٬۰۰۱", False),
], ids=("supported_python_output", "unsupported_python_claim"))
def test_python_output_numbers_are_grounded_but_not_invented(amount_text, expected_success):
    answer = f"خروجی محاسبه {amount_text} تومان بود."
    llm = ScriptedLLM(
        tool_turn(call("execute_python", {"code": "print('53,250,000')"}, "py-1")),
        final_turn(answer),
    )
    operation = lambda: alpha_agent(llm, sandbox=PythonResultStub()).run(
        ChatRequest(question="خروجی محاسبه را گزارش کن")
    )
    if expected_success:
        assert operation().answer == answer
    else:
        with pytest.raises(UngroundedAnswerError) as caught:
            operation()
        assert caught.value.reason_code == "unsupported_number"
    assert tool_result(llm.calls[1]["messages"], "py-1")["stdout"] == "53,250,000\n"


def test_real_sql_and_ticket_results_can_ground_revenue_and_ticket_count_together():
    assert paid_total() == 53_250_000
    answer = "درآمد پرداخت‌شده سپتامبر ۲۰۲۶ برابر ۵۳٬۲۵۰٬۰۰۰ تومان و تعداد تیکت‌های A001 برابر ۲ است."
    llm = ScriptedLLM(
        tool_turn(
            call("get_tickets", {"customer_id": "A001"}, "tickets-1"),
            call("query_sql", {"sql": PAID_SQL}, "sql-1"),
        ),
        final_turn(answer),
    )
    response = alpha_agent(llm).run(ChatRequest(question=QUESTION))
    assert len(tool_result(llm.calls[1]["messages"], "tickets-1")["tickets"]) == 2
    assert tool_result(llm.calls[1]["messages"], "sql-1") == {"rows": [{"revenue": 53_250_000}]}
    assert response.answer == answer
