"""B1 acceptance tests for the standalone grounding boundary."""

import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from decimal import Decimal, localcontext

import pytest

from app.month_scope import requested_month
from app.secure_data import SecureDataAccess
from app.structured_grounding import (
    EvidenceScope, FieldKind, GroundingContext, RecordedToolResult,
    StructuredGroundingError, validate_and_render,
)
from test_agent_loop import DATA, TOKEN_MAP


SEPTEMBER = requested_month("درآمد سپتامبر ۲۰۲۶ چقدر بود؟")


def answer(template, **slots):
    return json.dumps({"template": template, "slots": slots}, ensure_ascii=False)


def revenue(**overrides):
    return {
        "type": "numeric", "unit": "toman", "value": "53,250,000.00",
        "evidence_ref": "sql_01", "evidence_field": "paid_amount", **overrides,
    }


@pytest.fixture(scope="module")
def alpha():
    return SecureDataAccess(data_dir=DATA, token_map=TOKEN_MAP).authenticate("demo-alpha")


@pytest.fixture(scope="module")
def context(alpha):
    return GroundingContext(EvidenceScope(alpha.state_key, "conversation_alpha"), "Alpha", SEPTEMBER)


@pytest.fixture(scope="module")
def paid_query(alpha, context):
    uri = f"file:{(DATA / 'alpha.db').as_posix()}?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        amounts = connection.execute(
            "SELECT amount_toman FROM orders WHERE status = 'paid' AND order_date LIKE '2026-09-%'"
        )
        total = sum(amount for amount, in amounts)
    assert total == 53_250_000
    rows = alpha.agent_query(
        "SELECT SUM(amount_toman) AS paid_amount, COUNT(*) AS order_count FROM paid_orders "
        "WHERE order_date >= ? AND order_date < ?", ["2026-09-01", "2026-10-01"],
    )
    assert rows == [{"paid_amount": total, "order_count": 120}]
    # The fixture supplies metadata for this known projection. JSON and result
    # aliases cannot declare a column authoritative.
    return RecordedToolResult(
        context.scope, "sql_01", "query_sql", True, {"rows": rows},
        sql_fields={"paid_amount": FieldKind.TOMAN, "order_count": FieldKind.COUNT},
        sql_month=SEPTEMBER,
    )


@pytest.mark.parametrize("number", ["53,250,000.00", "۵۳٬۲۵۰٬۰۰۰٫۰۰", "٥٣٢٥٠٠٠٠", "53250000"])
def test_render_preserves_prose_and_uses_authoritative_spelling(context, paid_query, number):
    raw = answer(
        "  در {{period}}، فروش {{company}} برابر {{amount}} تومان بود.\n"
        "مبلغ تکرار می‌شود: {{amount}}.  ",
        period={"source": "user_request", "value": "۲۰۲۶-۰۹"},
        company={"source": "tenant_context", "value": "Alpha"},
        amount=revenue(value=number),
    )
    assert validate_and_render(raw, context=context, tool_results=[paid_query]) == (
        "  در 2026-09، فروش Alpha برابر 53250000 تومان بود.\n"
        "مبلغ تکرار می‌شود: 53250000.  "
    )


def test_replacement_text_is_literal_and_empty_slots_are_valid(context):
    context = replace(context, tenant_label="{{other}}")
    raw = answer("شرکت {{label}}", label={"source": "tenant_context", "value": "{{other}}"})
    assert validate_and_render(raw, context=context) == "شرکت {{other}}"
    assert validate_and_render(answer("داده‌ای یافت نشد."), context=context) == "داده‌ای یافت نشد."


@pytest.mark.parametrize(
    "raw",
    [
        "داده‌ای یافت نشد.",
        '{"template":"پاسخ","template":"پاسخ دیگر","slots":{}}',
        '{"template":"{{x}}","slots":{"x":{"type":"date","type":"numeric"}}}',
        '{"template":"پاسخ","slots":{},"tenant":"Beta"}',
        '{"template":"پاسخ\\ud800","slots":{}}',
        '{"template":"پاسخ","slots":{"x":NaN}}',
        answer("مبلغ {{missing}}"),
        answer("مبلغ {{x.__class__}}", x=revenue()),
        answer("{% if x %} پاسخ {% endif %}"),
        answer("مبلغ {{x}", x=revenue()),
        answer("مبلغ ثبت شد.", x=revenue()),
        answer("عدد ۵۳٬۲۵۰٬۰۰۰ تومان بود."),
        answer("تاریخ 2026-09-01 ثبت شد."),
        answer("عدد ² درج شد."),
    ],
)
def test_strict_json_placeholder_integrity_and_unchecked_literals(context, raw):
    with pytest.raises(StructuredGroundingError, match=r"^invalid_template$"):
        validate_and_render(raw, context=context)


@pytest.mark.parametrize(
    "patch",
    [
        {"value": "53,250,001"}, {"value": "53,250,000.001"}, {"value": "5,325,0000"},
        {"value": "5.325e7"}, {"value": "NaN"}, {"value": "Infinity"},
        {"value": 53_250_000}, {"value": True}, {"unit": "rial"}, {"unit": "percent"},
        {"type": "date"}, {"evidence_ref": "invented"}, {"evidence_field": "absent"},
        {"evidence_field": "order_count", "unit": "count"}, {"row": True}, {"row": -1},
        {"row": None}, {"row": 1}, {"tenant": "Alpha"}, {"formula": "amount + amount"},
    ],
)
def test_rejects_invalid_value_field_type_unit_or_row(context, paid_query, patch):
    with pytest.raises(StructuredGroundingError):
        validate_and_render(
            answer("مبلغ {{x}} تومان بود.", x=revenue(**patch)),
            context=context, tool_results=[paid_query],
        )


def test_same_number_cannot_supply_missing_field_or_unverified_provenance(alpha, context, paid_query):
    rows = alpha.agent_query("SELECT 53250000 AS paid_amount")
    assert rows == [{"paid_amount": 53_250_000}]
    raw = answer("مبلغ {{x}} تومان بود.", x=revenue())
    for record in (
        replace(paid_query, result={"rows": rows, "sql_fields": {"paid_amount": "toman"}},
                sql_fields={}),
        replace(paid_query, result={"rows": [{"some_other_field": 53_250_000}]}),
    ):
        with pytest.raises(StructuredGroundingError, match=r"^unsupported_number$"):
            validate_and_render(raw, context=context, tool_results=[record])


def test_reference_requires_a_unique_successful_result_in_the_owned_conversation(context, paid_query):
    beta = SecureDataAccess(data_dir=DATA, token_map=TOKEN_MAP).authenticate("demo-beta")
    raw = answer("مبلغ {{x}} تومان بود.", x=revenue())
    foreign = replace(paid_query, scope=EvidenceScope(beta.state_key, context.scope.conversation_id))
    other_conversation = replace(paid_query, scope=replace(context.scope, conversation_id="other"))
    failed = replace(paid_query, succeeded=False)
    tool_error = replace(paid_query, result={**paid_query.result, "error": "internal sentinel"})
    for records in (
        [], [foreign], [other_conversation], [failed], [tool_error],
        [paid_query, paid_query], [failed, paid_query],
    ):
        with pytest.raises(StructuredGroundingError, match=r"^unsupported_number$"):
            validate_and_render(raw, context=context, tool_results=records)
    assert validate_and_render(raw, context=context, tool_results=[foreign, paid_query]) == (
        "مبلغ 53250000 تومان بود."
    )


def test_corrected_window_does_not_make_wrong_window_evidence_eligible(context, paid_query):
    shahrivar = requested_month("درآمد شهریور ۱۴۰۵ چقدر بود؟")
    wrong_window = replace(paid_query, call_id="wrong", sql_month=shahrivar)
    raw = answer("مبلغ {{x}} تومان بود.", x=revenue(evidence_ref="wrong"))
    with pytest.raises(StructuredGroundingError, match=r"^invalid_date_range$"):
        validate_and_render(raw, context=context, tool_results=[wrong_window, paid_query])
    with pytest.raises(StructuredGroundingError, match=r"^invalid_date_range$"):
        validate_and_render(
            answer("مبلغ {{x}} تومان بود.", x=revenue()), context=context,
            tool_results=[replace(paid_query, sql_month=None)],
        )


def test_only_verified_context_values_can_fill_period_date_and_label(context):
    start = {"source": "user_request", "type": "date", "field": "period_start", "value": "۲۰۲۶-۰۹-۰۱"}
    assert validate_and_render(answer("از {{start}}", start=start), context=context) == "از 2026-09-01"
    for slot in (
        {**start, "value": "2026-09-30"}, {**start, "value": "2026-09-31"},
        {"source": "user_request", "value": "2026-10"},
        {"source": "tenant_context", "value": "Beta"},
    ):
        with pytest.raises(StructuredGroundingError):
            validate_and_render(answer("برای {{slot}}", slot=slot), context=context)


def test_monthly_sql_count_requires_a_verified_window(context, paid_query):
    raw = answer("تعداد {{x}} بود.", x=revenue(evidence_field="order_count", value="120", unit="count"))
    assert validate_and_render(raw, context=context, tool_results=[paid_query]) == "تعداد 120 بود."
    with pytest.raises(StructuredGroundingError, match=r"^invalid_date_range$"):
        validate_and_render(raw, context=context, tool_results=[replace(paid_query, sql_month=None)])


def test_ticket_counts_and_selected_metadata_use_actual_retrieval(context, alpha):
    tickets = alpha.tickets(customer_id="A001")
    assert {ticket["ticket_id"] for ticket in tickets} == {"AT001", "AT002"}
    record = RecordedToolResult(context.scope, "tickets_01", "get_tickets", True, {"tickets": tickets})
    count = revenue(evidence_ref="tickets_01", evidence_field="ticket_count", unit="count", value="۲")
    ticket = {"type": "identifier", "value": "AT001", "evidence_ref": "tickets_01",
              "evidence_field": "ticket_id", "row": 0}
    date = {**ticket, "type": "date", "value": "۲۰۲۶-۱۰-۰۳", "evidence_field": "created_at"}
    raw = answer("از {{n}} تیکت، {{ticket}} در {{date}} ثبت شد.", n=count, ticket=ticket, date=date)
    assert validate_and_render(raw, context=context, tool_results=[record]) == (
        "از 2 تیکت، AT001 در 2026-10-03 ثبت شد."
    )
    assert any("۶" in item["description"] for item in tickets)
    for slot in (
        {**count, "unit": "toman"}, {**count, "value": "3"},
        {**count, "evidence_field": "description", "value": "6", "row": 1},
        {**ticket, "row": 1}, {key: value for key, value in ticket.items() if key != "row"},
    ):
        with pytest.raises(StructuredGroundingError):
            validate_and_render(answer("مقدار {{x}} ثبت شد.", x=slot), context=context, tool_results=[record])


def test_percentage_uses_the_typed_field_and_exact_decimal_value(context):
    context = replace(context, month=None)
    record = RecordedToolResult(
        context.scope, "sql_01", "query_sql", True,
        {"rows": [{"order_share": 3.33, "other_field": 6.66}]},
        sql_fields={"order_share": FieldKind.PERCENT, "other_field": FieldKind.SCALAR},
    )
    slot = revenue(evidence_field="order_share", unit="percent", value="۳٫۳۳۰")
    assert validate_and_render(answer("سهم {{x}} درصد بود.", x=slot), context=context,
                               tool_results=[record]) == "سهم 3.33 درصد بود."
    for patch in ({"value": "6.66"}, {"evidence_field": "other_field", "value": "6.66"}):
        with pytest.raises(StructuredGroundingError):
            validate_and_render(answer("سهم {{x}} درصد بود.", x={**slot, **patch}),
                                context=context, tool_results=[record])


def test_python_only_exposes_named_scalar_metrics_from_successful_stdout(context):
    metric = "9007199254740993.125"
    result = {"stdout": json.dumps({"metrics": {"index": metric}}), "stderr": "317", "exit_code": 0}
    record = RecordedToolResult(context.scope, "py_01", "execute_python", True, result)
    slot = revenue(evidence_ref="py_01", evidence_field="index", unit="scalar",
                   value="۹٬۰۰۷٬۱۹۹٬۲۵۴٬۷۴۰٬۹۹۳٫۱۲۵۰")
    raw = answer("شاخص {{x}} بود.", x=slot)
    with localcontext() as decimal_context:
        decimal_context.prec = 4
        assert validate_and_render(raw, context=context, tool_results=[record]) == f"شاخص {metric} بود."
    for patch in ({"unit": "percent"}, {"unit": "toman"}, {"evidence_field": "stderr", "value": "317"},
                  {"value": "9007199254740993.124"}):
        with pytest.raises(StructuredGroundingError):
            validate_and_render(answer("شاخص {{x}} بود.", x={**slot, **patch}),
                                context=context, tool_results=[record])
    for patch in ({"exit_code": 1}, {"exit_code": False}, {"stdout": f"شاخص {metric}"},
                  {"stdout": '{"metrics":{"index":"0","index":"' + metric + '"}}'}):
        with pytest.raises(StructuredGroundingError):
            validate_and_render(raw, context=context, tool_results=[replace(record, result={**result, **patch})])


@pytest.mark.parametrize("value", [None, True, float("inf"), Decimal("1E+900")])
def test_invalid_authoritative_numeric_types_are_rejected(context, paid_query, value):
    record = replace(paid_query, result={"rows": [{"paid_amount": value}]})
    with pytest.raises(StructuredGroundingError, match=r"^unsupported_number$"):
        validate_and_render(answer("مبلغ {{x}}", x=revenue()), context=context, tool_results=[record])
