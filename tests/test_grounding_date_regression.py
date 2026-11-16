"""Phase 4-C-C.1-A RED: diagnose and prevent the first live answer rejection.

The LLM is scripted; SQL uses the original Alpha database read-only. No
OpenRouter request or generated Python execution occurs in this module.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agent import AgentLoop, EnvChatCompletionsClient, UngroundedAnswerError
from app.main import create_app
from app.schemas import ChatRequest
from app.secure_data import SecureDataAccess
from test_agent_loop import ScriptedLLM, call, numeric_slot, structured_turn, tool_result, tool_turn


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TOKEN_FILE = ROOT / "demo_tokens.json.example"
TOKEN_MAP = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
SCHEMA = (DATA / "schema.md").read_text(encoding="utf-8")
QUESTION = "درآمد paid سپتامبر ۲۰۲۶ میلادی چقدر بود؟"
SHAHRIVAR_QUESTION = "درآمد paid شهریور ۱۴۰۵ هجری شمسی چقدر بود؟"
PRIVATE_SENTINEL = "PRIVATE_FINAL_ANSWER_SENTINEL"


@pytest.fixture(autouse=True)
def forbid_live_provider(monkeypatch):
    def unexpected_provider_call(self, *, messages, tools):
        pytest.fail("A mocked grounding test attempted a live provider call")

    monkeypatch.setattr(EnvChatCompletionsClient, "create", unexpected_provider_call)


def alpha_tenant():
    return SecureDataAccess(data_dir=DATA, token_map=TOKEN_MAP).authenticate(
        "demo-alpha"
    )


def alpha_agent(llm, *, state_store=None):
    return AgentLoop(
        llm=llm, tenant=alpha_tenant(), schema_text=SCHEMA,
        state_store=state_store, trace_id_factory=lambda: "trace-red-grounding",
    )


def source_paid_total(start, end, *, inclusive_end=False):
    """Independent calculation against the immutable source orders."""
    uri = f"file:{(DATA / 'alpha.db').resolve().as_posix()}?mode=ro&immutable=1"
    comparison = "<=" if inclusive_end else "<"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        return connection.execute(
            "SELECT SUM(amount_toman) FROM orders WHERE status = 'paid' "
            f"AND order_date >= ? AND order_date {comparison} ?",
            (start, end),
        ).fetchone()[0]


def month_request(*, conversation_id=None):
    return ChatRequest(question=QUESTION, conversation_id=conversation_id)


def test_first_llm_turn_explains_gregorian_september_boundaries():
    llm = ScriptedLLM(structured_turn("برای پاسخ به داده نیاز است."))
    alpha_agent(llm).run(month_request())

    prompt_seen_by_model = "\n".join(
        item["content"] for item in llm.calls[0]["messages"]
        if isinstance(item.get("content"), str)
    )
    if "2026-09-01" not in prompt_seen_by_model:
        pytest.fail("Gregorian September start is missing from model context", pytrace=False)
    if "2026-09-30" not in prompt_seen_by_model:
        pytest.fail("Gregorian September end is missing from model context", pytrace=False)
    assert "میلادی" in prompt_seen_by_model or "Gregorian" in prompt_seen_by_model


def test_shahrivar_1405_has_distinct_persian_calendar_guidance():
    llm = ScriptedLLM(structured_turn("برای پاسخ به داده نیاز است."))
    alpha_agent(llm).run(ChatRequest(question=SHAHRIVAR_QUESTION))

    prompt_seen_by_model = "\n".join(
        item["content"] for item in llm.calls[0]["messages"]
        if isinstance(item.get("content"), str)
    )
    if "2026-08-23" not in prompt_seen_by_model:
        pytest.fail("Shahrivar start is missing from model context", pytrace=False)
    if "2026-09-22" not in prompt_seen_by_model:
        pytest.fail("Shahrivar end is missing from model context", pytrace=False)
    assert "شمسی" in prompt_seen_by_model or "Solar Hijri" in prompt_seen_by_model


@pytest.mark.parametrize(
    ("completion", "expected_code"),
    [
        (structured_turn(f"PRIVATE {PRIVATE_SENTINEL}"), "non_persian"),
        (structured_turn(f"درآمد {{{{revenue}}}} تومان است. {PRIVATE_SENTINEL}",
                         revenue=numeric_slot(999, "sql-1", "revenue")), "unsupported_number"),
    ],
    ids=("non_persian", "unsupported_number"),
)
def test_grounding_error_has_precise_safe_code(completion, expected_code, caplog):
    llm = ScriptedLLM(completion)
    with pytest.raises(UngroundedAnswerError) as caught:
        alpha_agent(llm).run(ChatRequest(question="درآمد چیست؟"))

    error = caught.value
    assert PRIVATE_SENTINEL not in str(error)
    assert PRIVATE_SENTINEL not in caplog.text
    assert len(llm.calls) == 1
    assert error.reason_code == expected_code
    assert error.reason_code in {"non_persian", "unsupported_number"}


def test_http_failure_records_safe_diagnosis_without_raw_answer(
    tmp_path, caplog,
):
    llm = ScriptedLLM(structured_turn(
        f"درآمد ادعایی {{{{revenue}}}} تومان است. {PRIVATE_SENTINEL}",
        revenue=numeric_slot(999, "sql-1", "revenue"),
    ))
    service = create_app(
        data_dir=DATA, token_map_path=TOKEN_FILE, schema_path=DATA / "schema.md",
        state_db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts",
        docker_image="python:3.13-slim", llm_client=llm,
    )
    tenant = service.state.secure_data.authenticate("demo-alpha")
    conversation_id = service.state.conversation_store.create_conversation(tenant)
    with TestClient(service) as client:
        response = client.post(
            "/chat", json={"question": "درآمد چیست؟",
                           "conversation_id": conversation_id},
            headers={"X-Demo-Token": "demo-alpha"},
        )

    assert response.status_code == 502
    assert response.json() == {"detail": "Analysis answer rejected"}
    snapshot = service.state.conversation_store.get_conversation(
        tenant, conversation_id
    )
    assert len(snapshot.turns) == 1
    assert snapshot.turns[0].status == "failed"
    assert snapshot.turns[0].answer is None
    assert PRIVATE_SENTINEL not in response.text
    assert PRIVATE_SENTINEL not in json.dumps(snapshot.messages, ensure_ascii=False)
    assert PRIVATE_SENTINEL not in caplog.text
    assert len(llm.calls) == 1
    assert snapshot.turns[0].failure_code == "unsupported_number"


@pytest.mark.parametrize(
    ("sql", "params"),
    [
        (
            "SELECT SUM(amount_toman) AS revenue FROM paid_orders "
            "WHERE order_date BETWEEN '2026-08-23' AND '2026-09-22'",
            [],
        ),
        (
            "SELECT SUM(amount_toman) AS revenue FROM paid_orders "
            "WHERE order_date >= ? AND order_date <= ?",
            ["2026-08-23", "2026-09-22"],
        ),
    ],
    ids=("literal_window", "parameter_window"),
)
def test_wrong_date_window_is_rejected_despite_matching_paid_total(sql, params):
    correct = source_paid_total("2026-09-01", "2026-10-01")
    wrong = source_paid_total("2026-08-23", "2026-09-22", inclusive_end=True)
    assert correct == wrong == 53_250_000  # The supplied data hide the bug.

    llm = ScriptedLLM(
        tool_turn(call("query_sql", {"sql": sql, "params": params}, "sql-1")),
        structured_turn("درآمد پرداخت‌شده سپتامبر برابر {{revenue}} تومان بود.",
                        revenue=numeric_slot(correct, "sql-1", "revenue")),
    )
    with pytest.raises(UngroundedAnswerError) as caught:
        alpha_agent(llm).run(month_request())

    assert caught.value.reason_code == "invalid_date_range"
    assert len(llm.calls) == 2


def test_gregorian_window_is_not_accepted_for_shahrivar_1405():
    correct_shahrivar = source_paid_total("2026-08-23", "2026-09-23")
    gregorian_september = source_paid_total("2026-09-01", "2026-10-01")
    assert correct_shahrivar == gregorian_september == 53_250_000

    llm = ScriptedLLM(
        tool_turn(call("query_sql", {
            "sql": "SELECT SUM(amount_toman) AS revenue FROM paid_orders "
                   "WHERE order_date >= ? AND order_date < ?",
            "params": ["2026-09-01", "2026-10-01"],
        }, "sql-1")),
        structured_turn(
            "درآمد پرداخت‌شده شهریور برابر {{revenue}} تومان بود.",
            revenue=numeric_slot(correct_shahrivar, "sql-1", "revenue"),
        ),
    )
    with pytest.raises(UngroundedAnswerError) as caught:
        alpha_agent(llm).run(ChatRequest(question=SHAHRIVAR_QUESTION))

    assert caught.value.reason_code == "invalid_date_range"
    assert len(llm.calls) == 2


def test_extra_day_filter_cannot_hide_behind_correct_month_predicate():
    sql = (
        "SELECT SUM(amount_toman) AS revenue FROM paid_orders "
        "WHERE order_date >= ? AND order_date < ? "
        "AND CAST(substr(order_date, 9, 2) AS INTEGER) <= 15"
    )
    llm = ScriptedLLM(
        tool_turn(call("query_sql", {
            "sql": sql, "params": ["2026-09-01", "2026-10-01"],
        }, "sql-1")),
        structured_turn("برای پاسخ به داده نیاز است."),
    )
    with pytest.raises(UngroundedAnswerError) as caught:
        alpha_agent(llm).run(month_request())

    assert caught.value.reason_code == "invalid_date_range"


@pytest.mark.parametrize(
    ("sql", "params"),
    [
        (
            "SELECT SUM(amount_toman) AS revenue FROM paid_orders "
            "WHERE order_date >= ? AND order_date < ?",
            ["2026-09-01", "2026-10-01"],
        ),
        (
            "SELECT SUM(amount_toman) AS revenue FROM paid_orders "
            "WHERE order_date BETWEEN '2026-09-01' AND '2026-09-30'",
            [],
        ),
    ],
    ids=("half_open_window", "inclusive_end_window"),
)
def test_correct_gregorian_window_accepts_grounded_persian_answer(sql, params):
    expected = source_paid_total("2026-09-01", "2026-10-01")
    answer = f"درآمد پرداخت‌شده سپتامبر برابر {expected} تومان بود."
    llm = ScriptedLLM(
        tool_turn(call("query_sql", {"sql": sql, "params": params}, "sql-1")),
        structured_turn("درآمد پرداخت‌شده سپتامبر برابر {{revenue}} تومان بود.",
                        revenue=numeric_slot(expected, "sql-1", "revenue")),
    )
    response = alpha_agent(llm).run(month_request())

    assert tool_result(llm.calls[1]["messages"], "sql-1") == {
        "rows": [{"revenue": expected}]
    }
    assert response.answer == answer
    assert [(item.tool_name, item.status) for item in response.trace] == [
        ("query_sql", "success")
    ]


def test_correct_shahrivar_window_accepts_grounded_persian_answer():
    expected = source_paid_total("2026-08-23", "2026-09-23")
    answer = f"درآمد پرداخت‌شده شهریور برابر {expected} تومان بود."
    llm = ScriptedLLM(
        tool_turn(call("query_sql", {
            "sql": "SELECT SUM(amount_toman) AS revenue FROM paid_orders "
                   "WHERE order_date BETWEEN ? AND ?",
            "params": ["2026-08-23", "2026-09-22"],
        }, "sql-1")),
        structured_turn("درآمد پرداخت‌شده شهریور برابر {{revenue}} تومان بود.",
                        revenue=numeric_slot(expected, "sql-1", "revenue")),
    )
    response = alpha_agent(llm).run(ChatRequest(question=SHAHRIVAR_QUESTION))

    assert tool_result(llm.calls[1]["messages"], "sql-1") == {
        "rows": [{"revenue": expected}]
    }
    assert response.answer == answer
    assert [(item.tool_name, item.status) for item in response.trace] == [
        ("query_sql", "success")
    ]


def test_unsupported_number_is_rejected_after_valid_paid_sql():
    expected = source_paid_total("2026-09-01", "2026-10-01")
    llm = ScriptedLLM(
        tool_turn(call("query_sql", {
            "sql": "SELECT SUM(amount_toman) AS revenue FROM paid_orders "
                   "WHERE order_date >= ? AND order_date < ?",
            "params": ["2026-09-01", "2026-10-01"],
        }, "sql-1")),
        structured_turn(
            "درآمد سپتامبر {{revenue}} تومان و تعداد مشتریان {{count}} بود.",
            revenue=numeric_slot(expected, "sql-1", "revenue"),
            count=numeric_slot(999, "sql-1", "customer_count", "count"),
        ),
    )
    with pytest.raises(UngroundedAnswerError) as caught:
        alpha_agent(llm).run(month_request())

    assert caught.value.reason_code == "unsupported_number"
    assert tool_result(llm.calls[1]["messages"], "sql-1") == {
        "rows": [{"revenue": expected}]
    }
