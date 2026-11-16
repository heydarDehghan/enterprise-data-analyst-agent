"""Phase 4-C-C.2 RED contract at the Agent and authenticated HTTP boundaries."""

import json
import sqlite3
from contextlib import closing
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.agent import AgentLoop, EnvChatCompletionsClient, UngroundedAnswerError
from app.main import create_app
from app.sandbox import SandboxResult
from app.schemas import ChatRequest
from app.secure_data import SecureDataAccess
from app.state import SQLiteConversationStore
from test_agent_loop import DATA, SCHEMA, TOKEN_MAP, ScriptedLLM, call, final_turn, tool_result, tool_turn


QUESTION = "درآمد پرداخت‌شده سپتامبر ۲۰۲۶ میلادی چقدر بود؟"
WINDOW = "WHERE order_date >= '2026-09-01' AND order_date < '2026-10-01'"
PAID_SQL = f"SELECT SUM(amount_toman) AS total_revenue FROM paid_orders {WINDOW}"


@pytest.fixture(autouse=True)
def forbid_live_provider(monkeypatch):
    def unexpected_call(*args, **kwargs):
        pytest.fail("Structured answer tests require a scripted LLM")

    monkeypatch.setattr(EnvChatCompletionsClient, "create", unexpected_call)


@pytest.fixture
def alpha():
    return SecureDataAccess(data_dir=DATA, token_map=TOKEN_MAP).authenticate("demo-alpha")


def source_paid_orders():
    uri = f"file:{(DATA / 'alpha.db').as_posix()}?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        return connection.execute(
            "SELECT order_date, customer_id, amount_toman FROM orders WHERE status = 'paid'"
        ).fetchall()


@pytest.fixture
def september_total():
    return sum(amount for date, _, amount in source_paid_orders() if date.startswith("2026-09"))


def structured(template, **slots):
    return final_turn(json.dumps({"template": template, "slots": slots}, ensure_ascii=False))


def numeric(value, ref="sql_01", field="total_revenue", unit="toman", **extra):
    return {
        "type": "numeric", "value": value, "evidence_ref": ref,
        "evidence_field": field, "unit": unit, **extra,
    }


def sql_turn(sql=PAID_SQL, ref="sql_01"):
    return tool_turn(call("query_sql", {"sql": sql}, ref))


def run(alpha, *responses, question=QUESTION, sandbox=None):
    llm = ScriptedLLM(*responses)
    return AgentLoop(llm=llm, tenant=alpha, schema_text=SCHEMA, sandbox=sandbox).run(
        ChatRequest(question=question)
    )


@pytest.mark.parametrize(
    ("template", "value"),
    [
        ("درآمد پرداخت‌شده {{revenue}} تومان بود.", "53,250,000.00"),
        ("اگر کوتاه بگویم، حاصل کار شد {{revenue}}.", "۵۳٬۲۵۰٬۰۰۰٫۰۰"),
    ],
    ids=["english_grouping", "persian_grouping"],
)
def test_agent_renders_model_wording_from_the_actual_sql_field(alpha, september_total, template, value):
    assert september_total == 53_250_000
    answer = run(
        alpha,
        sql_turn(PAID_SQL.replace("total_revenue", "paid_amount")),
        structured(template, revenue=numeric(value, field="paid_amount")),
    )
    assert answer.answer == template.replace("{{revenue}}", str(september_total))
    assert [(step.tool_name, step.status) for step in answer.trace] == [("query_sql", "success")]


def test_validated_period_and_tenant_label_render_with_revenue(alpha, september_total):
    answer = run(
        alpha, sql_turn(),
        structured(
            "در {{period}}، فروش {{company}} برابر با {{revenue}} تومان بود.",
            period={"source": "user_request", "value": "۲۰۲۶-۰۹"},
            company={"source": "tenant_context", "value": "Alpha"},
            revenue=numeric(str(september_total)),
        ),
    )
    assert answer.answer == f"در 2026-09، فروش Alpha برابر با {september_total} تومان بود."


def test_validated_gregorian_start_and_end_dates_render(alpha, september_total):
    answer = run(
        alpha, sql_turn(),
        structured(
            "از {{start}} تا {{end}}، درآمد {{revenue}} تومان بود.",
            start={"source": "user_request", "type": "date",
                   "field": "period_start", "value": "۲۰۲۶-۰۹-۰۱"},
            end={"source": "user_request", "type": "date",
                 "field": "period_end", "value": "۲۰۲۶-۰۹-۳۰"},
            revenue=numeric(str(september_total)),
        ),
    )
    assert answer.answer == f"از 2026-09-01 تا 2026-09-30، درآمد {september_total} تومان بود."


@pytest.mark.parametrize(
    "slot",
    [
        {"source": "user_request", "value": "2026-10"},
        {"source": "user_request", "type": "date", "field": "period_end", "value": "2026-09-22"},
        {"source": "tenant_context", "value": "Beta"},
    ],
    ids=["wrong_period", "shahrivar_date", "foreign_label"],
)
def test_context_slots_cannot_override_request_or_authenticated_tenant(alpha, september_total, slot):
    with pytest.raises(UngroundedAnswerError):
        run(
            alpha, sql_turn(),
            structured("گزارش {{context}} با مبلغ {{revenue}} تومان.",
                       context=slot, revenue=numeric(str(september_total))),
        )


@pytest.mark.parametrize(
    ("sql", "slot"),
    [
        (PAID_SQL, numeric("53250001")),
        (PAID_SQL, numeric("53250000", ref="invented_ref")),
        (PAID_SQL, numeric("53250000", field="nonexistent_field")),
        (PAID_SQL, numeric("53250000", unit="rial")),
        (PAID_SQL, numeric("53250000", unit="percent")),
        (PAID_SQL, {**numeric("53250000"), "type": "text"}),
        (PAID_SQL, numeric(53_250_000)),
        (PAID_SQL, numeric("53250000", tenant="Beta")),
        (PAID_SQL, numeric("53250000", formula="total_revenue + total_revenue")),
        (f"SELECT 53250000 AS total_revenue FROM paid_orders {WINDOW} LIMIT 1", numeric("53250000")),
        (f"SELECT COUNT(*) AS total_revenue FROM paid_orders {WINDOW}", numeric("120")),
    ],
    ids=[
        "wrong_value", "forged_ref", "missing_field", "wrong_currency", "unsupported_percent",
        "wrong_type", "non_string_value", "forged_owner", "model_formula", "sql_literal",
        "count_with_revenue_alias",
    ],
)
def test_numeric_slot_requires_matching_value_field_type_unit_and_provenance(alpha, sql, slot):
    if sql.startswith("SELECT COUNT"):
        count = sum(date.startswith("2026-09") for date, _, _ in source_paid_orders())
        assert slot["value"] == str(count)
    with pytest.raises(UngroundedAnswerError):
        run(alpha, sql_turn(sql), structured("مقدار {{value}} بود.", value=slot))


def test_sql_slot_cannot_borrow_a_value_from_a_different_result_field(alpha, september_total):
    sql = PAID_SQL.replace("total_revenue", "total_revenue, COUNT(*) AS order_count")
    with pytest.raises(UngroundedAnswerError):
        run(
            alpha, sql_turn(sql),
            structured("تعداد سفارش‌ها {{count}} بود.",
                       count=numeric(str(september_total), field="order_count", unit="count")),
        )


@pytest.mark.parametrize("field", ["revenue", "order_share"], ids=["uses_other_field", "wrong_calculation"])
def test_percentage_cannot_be_invented_even_if_other_numeric_evidence_exists(alpha, field):
    sql = (
        "SELECT SUM(amount_toman) AS revenue, "
        "ROUND(100.0 * COUNT(CASE WHEN customer_id = 'A001' THEN 1 END) / COUNT(*), 2) "
        "AS order_share FROM paid_orders"
    )
    orders = source_paid_orders()
    actual_percent = (Decimal(sum(customer == "A001" for _, customer, _ in orders))
                      * 100 / Decimal(len(orders))).quantize(Decimal("0.01"))
    assert actual_percent == Decimal("3.33")
    actual_revenue = sum(amount for _, _, amount in orders)
    with pytest.raises(UngroundedAnswerError):
        run(
            alpha, sql_turn(sql),
            structured("سهم مشتری {{share}} درصد بود.",
                       share=numeric(str(actual_revenue) if field == "revenue" else "6.66",
                                     field=field, unit="percent")),
            question="سهم مشتری از سفارش‌های پرداخت‌شده چقدر بود؟",
        )


def test_sql_calculation_can_support_an_exact_percentage(alpha):
    sql = (
        "SELECT ROUND(100.0 * COUNT(CASE WHEN customer_id = 'A001' THEN 1 END) / COUNT(*), 2) "
        "AS order_share FROM paid_orders"
    )
    orders = source_paid_orders()
    expected = (Decimal(sum(customer == "A001" for _, customer, _ in orders))
                * 100 / Decimal(len(orders))).quantize(Decimal("0.01"))
    answer = run(
        alpha, sql_turn(sql),
        structured("سهم مشتری {{share}} درصد بود.",
                   share=numeric("۳٫۳۳۰", field="order_share", unit="percent")),
        question="سهم مشتری از سفارش‌های پرداخت‌شده چقدر بود؟",
    )
    assert answer.answer == f"سهم مشتری {expected} درصد بود."


@pytest.mark.parametrize(
    ("template", "slots"),
    [
        ("مقدار {{unknown}} اعلام شد.", {"value": numeric("53250000")}),
        ("مقدار {{value}} اعلام شد.", {}),
        ("نتیجه آماده است.", {"value": numeric("53250000")}),
        ("مقدار {{ value.__class__ }} است.", {"value": numeric("53250000")}),
        ("مقدار {{value است.", {"value": numeric("53250000")}),
        ("{% if value %}نتیجه آماده است.{% endif %}", {}),
    ],
    ids=["unknown", "missing", "extra", "attribute", "unclosed", "control_flow"],
)
def test_invalid_placeholder_structure_is_rejected(alpha, template, slots):
    with pytest.raises(UngroundedAnswerError):
        run(alpha, sql_turn(), structured(template, **slots))


def test_duplicate_json_member_is_rejected(alpha):
    duplicate = '{"template":"نتیجه آماده است.","template":"نتیجه دیگر است.","slots":{}}'
    with pytest.raises(UngroundedAnswerError):
        run(alpha, final_turn(duplicate))


@pytest.mark.parametrize(
    "literal", ["53,250,000", "۵۳٬۲۵۰٬۰۰۰", "۲۰۲۶-۰۹-۰۱", "²"],
    ids=["ascii_amount", "persian_amount", "date", "unicode_numeric"],
)
def test_even_evidenced_numeric_literals_must_be_in_slots(alpha, literal):
    with pytest.raises(UngroundedAnswerError):
        run(alpha, sql_turn(), structured(f"در گزارش مقدار {literal} ثبت شد."))


def test_plain_persian_fallback_is_rejected(alpha):
    with pytest.raises(UngroundedAnswerError):
        run(alpha, final_turn("داده کافی در دسترس نیست."))


def test_model_can_return_its_own_insufficient_evidence_template(alpha):
    answer = run(alpha, structured("داده کافی در دسترس نیست."))
    assert answer.answer == "داده کافی در دسترس نیست."


def test_ticket_count_is_derived_from_returned_tenant_records(alpha, september_total):
    def finish(messages, tools):
        tickets = tool_result(messages, "tickets_01")["tickets"]
        assert {ticket["ticket_id"] for ticket in tickets} == {"AT001", "AT002"}
        assert all(ticket["is_untrusted"] for ticket in tickets)
        assert tool_result(messages, "sql_01")["rows"] == [{"total_revenue": september_total}]
        assert {"query_sql", "get_schema", "get_tickets"} <= {
            tool["function"]["name"] for tool in tools
        }
        return structured(
            "درآمد {{revenue}} تومان بود و {{count}} تیکت بازگردانده شد.",
            revenue=numeric(str(september_total)),
            count=numeric("۲", "tickets_01", "ticket_count", "count"),
        )

    answer = run(
        alpha, tool_turn(call("get_tickets", {"customer_id": "A001"}, "tickets_01")),
        tool_turn(call("get_schema", {}, "schema_01")), sql_turn(), finish,
    )
    assert answer.answer == f"درآمد {september_total} تومان بود و 2 تیکت بازگردانده شد."
    assert [step.tool_name for step in answer.trace] == ["get_tickets", "get_schema", "query_sql"]


@pytest.mark.parametrize(
    "slot",
    [
        numeric("3", "tickets_01", "ticket_count", "count"),
        numeric("2", "tickets_01", "ticket_count", "toman"),
        numeric("6", "tickets_01", "description", "toman", row=1),
    ],
    ids=["forged_ticket_count", "ticket_count_as_money", "ticket_text_as_money"],
)
def test_ticket_fields_do_not_authorize_arbitrary_numeric_claims(alpha, slot):
    with pytest.raises(UngroundedAnswerError):
        run(
            alpha, tool_turn(call("get_tickets", {"customer_id": "A001"}, "tickets_01")),
            structured("مقدار ثبت‌شده {{value}} بود.", value=slot),
        )


def ticket_slot(value, field, kind, row):
    return {
        "type": kind, "value": value, "evidence_ref": "tickets_01",
        "evidence_field": field, "row": row,
    }


def test_ticket_identifier_and_date_slots_render_from_the_selected_record(alpha):
    def finish(messages, tools):
        first = tool_result(messages, "tickets_01")["tickets"][0]
        assert (first["ticket_id"], first["created_at"]) == ("AT001", "2026-10-03")
        return structured(
            "تیکت {{ticket}} در {{date}} ثبت شد.",
            ticket=ticket_slot("AT001", "ticket_id", "identifier", 0),
            date=ticket_slot("۲۰۲۶-۱۰-۰۳", "created_at", "date", 0),
        )

    answer = run(alpha, tool_turn(call("get_tickets", {"customer_id": "A001"}, "tickets_01")),
                 finish, question="تاریخ ثبت تیکت مشتری چقدر است؟")
    assert answer.answer == "تیکت AT001 در 2026-10-03 ثبت شد."


@pytest.mark.parametrize(
    "slot",
    [
        ticket_slot("AT999", "ticket_id", "identifier", 0),
        ticket_slot("A999", "customer_id", "identifier", 0),
        ticket_slot("AT002", "ticket_id", "identifier", 0),
        {k: v for k, v in ticket_slot("AT001", "ticket_id", "identifier", 0).items() if k != "row"},
    ],
    ids=["invented_ticket", "invented_customer", "different_row", "ambiguous_row"],
)
def test_ticket_id_slot_requires_the_exact_record_field(alpha, slot):
    with pytest.raises(UngroundedAnswerError):
        run(
            alpha, tool_turn(call("get_tickets", {"customer_id": "A001"}, "tickets_01")),
            structured("شناسه ثبت‌شده {{value}} بود.", value=slot),
            question="شناسه ثبت‌شده در ردیف ۰ تیکت‌های مشتری چیست؟",
        )


class PythonResult:
    def execute(self, code, tenant):
        return SandboxResult(
            stdout='{"metrics":{"index":"9007199254740993.125"}}',
            stderr="diagnostic 317",
            exit_code=0,
        )


def python_turn():
    return tool_turn(call("execute_python", {"code": "print(result_json)"}, "py_01"))


def test_python_named_metric_renders_without_float_rounding(alpha):
    answer = run(
        alpha, python_turn(),
        structured("شاخص محاسبه‌شده {{index}} بود.",
                   index=numeric("۹٬۰۰۷٬۱۹۹٬۲۵۴٬۷۴۰٬۹۹۳٫۱۲۵۰", "py_01", "index", "scalar")),
        sandbox=PythonResult(), question="شاخص محاسبه‌شده را گزارش کن",
    )
    assert answer.answer == "شاخص محاسبه‌شده 9007199254740993.125 بود."
    assert [(step.tool_name, step.status) for step in answer.trace] == [
        ("execute_python", "success")
    ]


@pytest.mark.parametrize(
    "slot",
    [
        numeric("9007199254740993.124", "py_01", "index", "scalar"),
        numeric("9007199254740993.125", "py_01", "index", "percent"),
        numeric("9007199254740993.125", "py_01", "index", "toman"),
        numeric("317", "py_01", "stderr", "scalar"),
        numeric("NaN", "py_01", "index", "scalar"),
        numeric("Infinity", "py_01", "index", "scalar"),
    ],
    ids=["precision_mismatch", "untyped_percent", "untyped_currency", "stderr", "nan", "infinity"],
)
def test_python_slots_require_the_exact_typed_metric(alpha, slot):
    with pytest.raises(UngroundedAnswerError):
        run(alpha, python_turn(), structured("شاخص {{index}} بود.", index=slot),
            sandbox=PythonResult(), question="شاخص محاسبه‌شده را گزارش کن")


def test_same_total_from_wrong_month_still_fails(alpha, september_total):
    previous = """
        SELECT SUM(amount_toman) AS total_revenue FROM paid_orders
        WHERE order_date >= '2026-08-23' AND order_date < '2026-09-23'
    """
    assert sum(amount for date, _, amount in source_paid_orders()
               if "2026-08-23" <= date < "2026-09-23") == september_total
    with pytest.raises(UngroundedAnswerError) as error:
        run(alpha, sql_turn(previous),
            structured("درآمد {{revenue}} تومان بود.", revenue=numeric(str(september_total))))
    assert error.value.reason_code == "invalid_date_range"


def test_corrected_month_does_not_rehabilitate_the_earlier_evidence_reference(alpha, september_total):
    earlier = PAID_SQL.replace("2026-09-01", "2026-08-23").replace("2026-10-01", "2026-09-23")
    with pytest.raises(UngroundedAnswerError):
        run(
            alpha, sql_turn(earlier, "wrong_window"), sql_turn(),
            structured("درآمد {{revenue}} تومان بود.",
                       revenue=numeric(str(september_total), ref="wrong_window")),
        )


def test_failed_raw_orders_query_cannot_supply_evidence(alpha, september_total):
    def finish(messages, tools):
        assert "error" in tool_result(messages, "raw_orders")
        return structured("درآمد {{revenue}} تومان بود.",
                          revenue=numeric(str(september_total), ref="raw_orders"))

    with pytest.raises(UngroundedAnswerError):
        run(alpha, sql_turn(PAID_SQL.replace("paid_orders", "orders"), "raw_orders"),
            sql_turn(), finish)


def test_duplicate_tool_call_id_cannot_bind_to_an_ambiguous_result(alpha, september_total):
    restricted_sql = PAID_SQL + " AND customer_id = 'A001'"
    with pytest.raises(UngroundedAnswerError):
        run(alpha, sql_turn(), sql_turn(restricted_sql),
            structured("درآمد {{value}} تومان بود.", value=numeric(str(september_total))))


def seed_alpha_evidence(store, alpha):
    """Persist one real Alpha tool result through the verified execution path."""
    llm = ScriptedLLM(sql_turn(), structured("گزارش ثبت شد."))
    return AgentLoop(llm=llm, tenant=alpha, schema_text=SCHEMA, state_store=store).run(
        ChatRequest(question=QUESTION)
    ).conversation_id


def test_owned_conversation_can_reuse_its_tool_evidence_after_restart(tmp_path, alpha, september_total):
    path = tmp_path / "turns.db"
    artifacts = tmp_path / "artifacts"
    first = SQLiteConversationStore(db_path=path, artifact_root=artifacts)
    conversation = seed_alpha_evidence(first, alpha)
    reopened = SQLiteConversationStore(db_path=path, artifact_root=artifacts)
    llm = ScriptedLLM(structured("مبلغ قبلی {{revenue}} تومان بود.",
                                 revenue=numeric(str(september_total))))
    answer = AgentLoop(llm=llm, tenant=alpha, schema_text=SCHEMA, state_store=reopened).run(
        ChatRequest(question="مبلغ قبلی چقدر بود؟", conversation_id=conversation)
    )
    assert answer.answer == f"مبلغ قبلی {september_total} تومان بود."
    assert answer.trace == []
    snapshot = reopened.get_conversation(alpha, conversation)
    assert snapshot.turns[-1].answer == answer.answer
    assert snapshot.turns[-1].failure_code is None


def test_foreign_evidence_is_rejected_even_with_same_number_and_http_failure_is_safe(
    tmp_path, alpha, september_total, caplog,
):
    sentinel = "PRIVATE_FINAL_TEMPLATE_SENTINEL"
    rejected = structured(f"درآمد {{{{revenue}}}} تومان بود. {sentinel}",
                          revenue=numeric(str(september_total)))
    # Give Beta the same numerical value through its own paid calculation.
    # The reference in the proposed answer still points only to Alpha's turn.
    beta_sql = PAID_SQL.replace("SUM(amount_toman)", "SUM(amount_toman) / 2")
    llm = ScriptedLLM(sql_turn(beta_sql, ref="beta_only"), rejected)
    root = DATA.parent
    service = create_app(
        data_dir=DATA, token_map_path=root / "demo_tokens.json.example", schema_path=DATA / "schema.md",
        state_db_path=tmp_path / "turns.db", artifact_root=tmp_path / "artifacts",
        docker_image="python:3.13-slim", llm_client=llm,
    )
    store = service.state.conversation_store
    alpha_conversation = seed_alpha_evidence(store, alpha)
    beta = service.state.secure_data.authenticate("demo-beta")
    beta_conversation = store.create_conversation(beta)
    headers = {"X-Demo-Token": "demo-beta"}
    with TestClient(service) as client:
        foreign = client.post("/chat", headers=headers, json={
            "question": QUESTION, "conversation_id": alpha_conversation,
        })
        assert foreign.status_code == 404
        assert llm.calls == []
        response = client.post("/chat", headers=headers, json={
            "question": QUESTION, "conversation_id": beta_conversation,
        })
    assert tool_result(llm.calls[1]["messages"], "beta_only")["rows"] == [
        {"total_revenue": september_total}
    ]
    assert response.status_code == 502
    assert response.json() == {"detail": "Analysis answer rejected"}
    snapshot = store.get_conversation(beta, beta_conversation)
    turn, = snapshot.turns
    assert turn.status == "failed"
    assert turn.answer is None
    assert turn.failure_code == "unsupported_number"
    assert sentinel not in json.dumps(snapshot.messages, ensure_ascii=False)
    assert sentinel not in response.text
    assert sentinel not in caplog.text
