"""Phase 4-A RED contract for durable, tenant-bound conversations.

Acceptance criteria:
* New requests create SQLite-backed conversations; later requests and new store
  instances replay the same messages, tool results, and turn records.
* A server-authenticated tenant is the authority for loading a conversation.
  Unknown, malformed, and foreign IDs fail closed before invoking the LLM.
* Prior successful SQL evidence can ground a follow-up without rerunning SQL.
* Transient tool failures are recorded and can recover; three failed attempts
  (initial call plus two retries) terminate and persist a failed turn.
"""

import json
import sqlite3
from contextlib import closing

import pytest

from app.schemas import ChatRequest
from app.secure_data import SecureDataAccess

from test_agent_loop import DATA, SCHEMA, TOKEN_MAP, ScriptedLLM, call, numeric_slot, structured_turn, tool_result, tool_turn


def state_store(tmp_path):
    from app.state import SQLiteConversationStore

    return SQLiteConversationStore(
        db_path=tmp_path / "conversations.sqlite3",
        artifact_root=tmp_path / "artifacts",
    )


def tenant(token="demo-alpha"):
    return SecureDataAccess(data_dir=DATA, token_map=TOKEN_MAP).authenticate(token)


def agent(store, llm, subject, trace_id="trace-state-1", sandbox=None):
    from app.agent import AgentLoop

    return AgentLoop(
        llm=llm, tenant=subject, schema_text=SCHEMA, sandbox=sandbox,
        state_store=store, trace_id_factory=lambda: trace_id,
    )


def test_messages_and_turns_survive_new_requests_and_store_restart(tmp_path):
    first_store = state_store(tmp_path)
    alpha = tenant()
    first = agent(
        first_store,
        ScriptedLLM(tool_turn(call("get_schema", {}, "schema-1")),
                    structured_turn("ساختار سفارش‌های پرداخت‌شده ثبت شد.")),
        alpha,
    ).run(ChatRequest(question="ساختار داده را بررسی کن"))

    assert first.conversation_id
    assert first.trace_id == "trace-state-1"
    assert first.trace[0].tool_name == "get_schema"

    reopened = state_store(tmp_path)

    def inspect_history(messages, tools):
        assert messages[0]["role"] == "system"
        assert any(m.get("role") == "user" and m.get("content") == "ساختار داده را بررسی کن"
                   for m in messages)
        assert tool_result(messages, "schema-1")["schema"] == SCHEMA
        assert any(m.get("role") == "assistant" and
                   m.get("content") == first.answer for m in messages)
        assert messages[-1] == {"role": "user", "content": "حالا خلاصه کن"}
        return structured_turn("خلاصه ساختار داده آماده است.")

    second = agent(reopened, ScriptedLLM(inspect_history), alpha, "trace-state-2").run(
        ChatRequest(question="حالا خلاصه کن", conversation_id=first.conversation_id)
    )
    snapshot = reopened.get_conversation(alpha, first.conversation_id)
    assert second.conversation_id == first.conversation_id
    assert [turn.trace_id for turn in snapshot.turns] == ["trace-state-1", "trace-state-2"]
    assert [turn.status for turn in snapshot.turns] == ["success", "success"]
    assert [turn.answer for turn in snapshot.turns] == [first.answer, second.answer]
    assert any(m.get("role") == "tool" for m in snapshot.messages)


def test_tenant_isolation_and_unknown_ids_have_same_failure(tmp_path):
    from app.state import ConversationNotFoundError

    store = state_store(tmp_path)
    alpha, beta = tenant("demo-alpha"), tenant("demo-beta")
    first = agent(store, ScriptedLLM(structured_turn("گفتگوی آلفا ثبت شد.")), alpha).run(
        ChatRequest(question="گفتگوی آلفا را شروع کن")
    )
    beta_llm = ScriptedLLM(structured_turn("نباید اجرا شود."))

    with pytest.raises(ConversationNotFoundError) as foreign:
        agent(store, beta_llm, beta).run(ChatRequest(
            question="گفتگوی قبلی را ادامه بده", conversation_id=first.conversation_id,
        ))
    with pytest.raises(ConversationNotFoundError) as unknown:
        agent(store, beta_llm, beta).run(ChatRequest(
            question="گفتگوی قبلی را ادامه بده", conversation_id="missing-conversation",
        ))
    assert str(foreign.value) == str(unknown.value)
    assert beta_llm.calls == []
    with pytest.raises(ConversationNotFoundError):
        store.get_conversation(beta, first.conversation_id)

    second = agent(store, ScriptedLLM(structured_turn("گفتگوی بتا ثبت شد.")), beta).run(
        ChatRequest(question="گفتگوی بتا را شروع کن")
    )
    assert second.conversation_id != first.conversation_id
    assert "آلفا" not in json.dumps(
        [m.get("content") for m in store.get_conversation(beta, second.conversation_id).messages],
        ensure_ascii=False,
    )


def test_prior_paid_sql_result_grounds_followup_without_requery(tmp_path):
    store, alpha = state_store(tmp_path), tenant()
    uri = f"file:{(DATA / 'alpha.db').as_posix()}?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as source:
        raw_rows = source.execute("SELECT order_date, amount_toman, status FROM orders").fetchall()
    expected = sum(amount for date, amount, status in raw_rows
                   if status == "paid" and date.startswith("2026-09"))
    assert expected == 53_250_000
    sql = "SELECT SUM(amount_toman) AS revenue FROM paid_orders WHERE substr(order_date, 1, 7) = ?"
    first = agent(store, ScriptedLLM(
        tool_turn(call("query_sql", {"sql": sql, "params": ["2026-09"]}, "paid-1")),
        structured_turn("درآمد پرداخت‌شده شهریور برابر {{revenue}} تومان بود.",
                        revenue=numeric_slot(expected, "paid-1", "revenue")),
    ), alpha).run(ChatRequest(question="درآمد پرداخت‌شده شهریور ۲۰۲۶ چقدر بود؟"))

    def reuse(messages, tools):
        assert tool_result(messages, "paid-1") == {"rows": [{"revenue": expected}]}
        assert messages[-1]["content"] == "همان مبلغ پرداخت‌شده چقدر بود؟"
        return structured_turn("همان مبلغ {{revenue}} تومان است.",
                               revenue=numeric_slot(expected, "paid-1", "revenue"))

    followup_llm = ScriptedLLM(reuse)
    second = agent(state_store(tmp_path), followup_llm, alpha, "trace-followup").run(
        ChatRequest(question="همان مبلغ پرداخت‌شده چقدر بود؟",
                    conversation_id=first.conversation_id)
    )
    assert second.answer == f"همان مبلغ {expected} تومان است."
    assert second.trace == []
    assert len(followup_llm.calls) == 1


def test_transient_tool_failure_recovers_and_persists_trace(tmp_path, monkeypatch):
    store, alpha = state_store(tmp_path), tenant()
    original = alpha.agent_query
    attempts = 0

    def flaky(sql, params=()):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary backend failure; secret-sentinel")
        return original(sql, params)

    monkeypatch.setattr(alpha, "agent_query", flaky)
    sql = "SELECT COUNT(*) AS count FROM paid_orders"

    def after_error(messages, tools):
        assert tool_result(messages, "try-1")["error"]
        assert "secret-sentinel" not in json.dumps(messages)
        return tool_turn(call("query_sql", {"sql": sql}, "try-2"))

    def after_success(messages, tools):
        assert tool_result(messages, "try-2")["rows"][0]["count"] > 0
        return structured_turn("خطای موقت برطرف شد و داده بررسی شد.")

    response = agent(store, ScriptedLLM(
        tool_turn(call("query_sql", {"sql": sql}, "try-1")),
        after_error, after_success,
    ), alpha, "trace-recovered").run(ChatRequest(question="سفارش‌های پرداخت‌شده را بشمار"))

    assert attempts == 2
    assert [item.status for item in response.trace] == ["error", "success"]
    snapshot = state_store(tmp_path).get_conversation(alpha, response.conversation_id)
    assert snapshot.turns[-1].trace_id == response.trace_id
    assert snapshot.turns[-1].status == "success"
    assert [item.status for item in snapshot.turns[-1].trace] == ["error", "success"]
    assert "secret-sentinel" not in json.dumps(
        [m.get("content") for m in snapshot.messages], ensure_ascii=False,
    )


def test_terminal_tool_failure_stops_after_two_retries_and_records_failed_turn(
    tmp_path, monkeypatch,
):
    from app.agent import AgentOperationFailed

    store, alpha = state_store(tmp_path), tenant()
    started = agent(store, ScriptedLLM(structured_turn("گفتگو شروع شد.")), alpha).run(
        ChatRequest(question="شروع کن")
    )
    attempts = 0

    def broken(sql, params=()):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("database unavailable; secret-sentinel")

    monkeypatch.setattr(alpha, "agent_query", broken)
    sql = "SELECT COUNT(*) AS count FROM paid_orders"
    llm = ScriptedLLM(*[
        tool_turn(call("query_sql", {"sql": sql}, f"failed-{n}"))
        for n in range(1, 5)
    ])
    with pytest.raises(AgentOperationFailed) as caught:
        agent(store, llm, alpha, "trace-failed").run(ChatRequest(
            question="دوباره بررسی کن", conversation_id=started.conversation_id,
        ))

    assert attempts == 3
    assert len(llm.calls) == 3
    assert [item.status for item in caught.value.trace] == ["error"] * 3
    snapshot = state_store(tmp_path).get_conversation(alpha, started.conversation_id)
    assert [turn.status for turn in snapshot.turns] == ["success", "failed"]
    assert snapshot.turns[-1].trace_id == "trace-failed"
    assert snapshot.turns[-1].answer is None
    assert [item.step for item in snapshot.turns[-1].trace] == [1, 2, 3]
    assert sum(m.get("role") == "tool" for m in snapshot.messages) == 3
    assert "secret-sentinel" not in json.dumps(
        [m.get("content") for m in snapshot.messages], ensure_ascii=False,
    )

    def inspect_failed_history(messages, tools):
        for n in range(1, 4):
            assert tool_result(messages, f"failed-{n}")["error"]
        assert messages[-1]["content"] == "از ساختار داده ادامه بده"
        return tool_turn(call("get_schema", {}, "recovery-schema"))

    recovered = agent(state_store(tmp_path), ScriptedLLM(
        inspect_failed_history,
        structured_turn("ساختار داده برای ادامه بررسی آماده است."),
    ), alpha, "trace-after-failure").run(ChatRequest(
        question="از ساختار داده ادامه بده", conversation_id=started.conversation_id,
    ))
    assert [item.status for item in recovered.trace] == ["success"]
    final_state = state_store(tmp_path).get_conversation(alpha, started.conversation_id)
    assert [turn.status for turn in final_state.turns] == ["success", "failed", "success"]


@pytest.mark.parametrize("bad_id", ["missing-conversation", "../alpha", "alpha/../../beta"])
def test_invalid_or_missing_explicit_session_fails_before_llm(tmp_path, bad_id):
    from app.state import ConversationNotFoundError

    store, alpha = state_store(tmp_path), tenant()
    llm = ScriptedLLM(structured_turn("نباید اجرا شود."))
    with pytest.raises(ConversationNotFoundError):
        agent(store, llm, alpha).run(ChatRequest(
            question="گفتگوی نامعتبر را ادامه بده", conversation_id=bad_id,
        ))
    assert llm.calls == []
    with pytest.raises(ConversationNotFoundError):
        store.get_conversation(alpha, bad_id)


def test_blank_conversation_id_is_rejected_by_request_schema():
    with pytest.raises(ValueError):
        ChatRequest(question="ادامه بده", conversation_id="  ")
