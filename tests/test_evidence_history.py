"""B3 regression coverage for persisted evidence and denied reuse."""

import json
import sqlite3
from contextlib import closing

import pytest

import app.evidence_history
import app.state
from app.agent import UngroundedAnswerError
from app.schemas import ChatRequest, ToolTrace
from app.state import ConversationNotFoundError
from app.structured_grounding import FieldKind
from test_agent_loop import ScriptedLLM, call, numeric_slot, structured_turn, tool_turn
from test_state_recovery import agent, state_store, tenant


QUESTION = "درآمد پرداخت‌شده سپتامبر ۲۰۲۶ چقدر بود؟"
SQL = "SELECT SUM(amount_toman) AS revenue FROM paid_orders WHERE substr(order_date, 1, 7) = ?"
TOTAL = 53_250_000


def sql_turn(ref, *, sql=SQL, params=None):
    return tool_turn(call("query_sql", {"sql": sql, "params": ["2026-09"] if params is None else params}, ref))


def cite(ref):
    return structured_turn("درآمد {{revenue}} تومان بود.", revenue=numeric_slot(TOTAL, ref, "revenue"))


def run(store, subject, *responses, question=QUESTION, conversation_id=None):
    return agent(store, ScriptedLLM(*responses), subject).run(
        ChatRequest(question=question, conversation_id=conversation_id)
    )


def test_reopened_store_recovers_result_type_source_and_window_for_owner(tmp_path):
    alpha = tenant()
    first = run(state_store(tmp_path), alpha, sql_turn("sep"), cite("sep"))
    reopened = state_store(tmp_path)
    saved, = reopened.get_conversation(alpha, first.conversation_id).evidence
    assert saved.scope.tenant_key == alpha.state_key
    assert saved.scope.conversation_id == first.conversation_id
    assert (saved.call_id, saved.tool_name, saved.succeeded) == ("sep", "query_sql", True)
    assert saved.result == {"rows": [{"revenue": TOTAL}]}
    assert saved.sql_fields == {"revenue": FieldKind.TOMAN}
    assert (saved.sql_month.first, saved.sql_month.last) == ("2026-09-01", "2026-09-30")
    source = saved.sql_provenance["revenue"]
    assert (source.table, source.column, source.calculation) == ("paid_orders", "amount_toman", "SUM(amount_toman)")

    with pytest.raises(ConversationNotFoundError):
        reopened.get_conversation(tenant("demo-beta"), first.conversation_id)
    other = reopened.create_conversation(alpha)
    with pytest.raises(UngroundedAnswerError, match="unsupported number"):
        run(reopened, alpha, cite("sep"), conversation_id=other)


def test_restart_revalidates_period_and_never_promotes_matching_total(tmp_path):
    alpha = tenant()
    explicit_range = (
        "SELECT SUM(amount_toman) AS revenue FROM paid_orders "
        "WHERE order_date >= ? AND order_date < ?"
    )
    started = run(
        state_store(tmp_path), alpha, sql_turn("shahrivar", sql=explicit_range,
                                              params=["2026-08-23", "2026-09-23"]),
        cite("shahrivar"), question="درآمد شهریور ۱۴۰۵ چه بود؟",
    )
    assert str(TOTAL) in started.answer
    reopened = state_store(tmp_path)
    with pytest.raises(UngroundedAnswerError) as failure:
        run(reopened, alpha, cite("shahrivar"), conversation_id=started.conversation_id)
    assert failure.value.reason_code == "invalid_date_range"


def test_legacy_database_keeps_transcript_but_never_backfills_trust(tmp_path):
    alpha, before = tenant(), state_store(tmp_path)
    conversation = before.create_conversation(alpha)
    # Legacy history can appear to contain successful SQL or claimed provenance.
    # A matching result, assistant tool call and success trace still need proof.
    result = json.dumps({"rows": [{"revenue": TOTAL}], "sql_fields": {"revenue": "TOMAN"}})
    assistant = sql_turn("legacy")["choices"][0]["message"]
    before.save_turn(
        alpha, conversation, trace_id="old", status="success", answer="گزارش قبلی",
        trace=[ToolTrace(step=1, tool_name="query_sql", status="success")], artifacts=[],
        messages=[assistant, {"role": "tool", "tool_call_id": "legacy", "content": result}],
    )
    with closing(sqlite3.connect(before.db_path)) as connection, connection:
        connection.execute("DROP TABLE turn_evidence")
    reopened = state_store(tmp_path)
    snapshot = reopened.get_conversation(alpha, conversation)
    assert snapshot.turns[0].trace[0].status == "success"
    assert snapshot.messages[-1]["content"] == result
    assert snapshot.evidence == []
    with pytest.raises(UngroundedAnswerError):
        run(reopened, alpha, cite("legacy"), conversation_id=conversation)


def test_tampered_result_is_unavailable_even_when_proof_and_trace_remain(tmp_path):
    alpha, store = tenant(), state_store(tmp_path)
    first = run(store, alpha, sql_turn("sep"), cite("sep"))
    with closing(sqlite3.connect(store.db_path)) as connection, connection:
        stored, = connection.execute("SELECT messages_json FROM turns").fetchone()
        messages = json.loads(stored)
        tool = next(message for message in messages if message["role"] == "tool")
        tool["content"] = json.dumps({"rows": [{"revenue": TOTAL + 1}]})
        connection.execute("UPDATE turns SET messages_json = ?", (json.dumps(messages),))
    reopened = state_store(tmp_path)
    assert reopened.get_conversation(alpha, first.conversation_id).evidence == []
    changed = structured_turn("درآمد {{revenue}} بود.",
                              revenue=numeric_slot(TOTAL + 1, "sep", "revenue"))
    with pytest.raises(UngroundedAnswerError):
        run(reopened, alpha, changed, conversation_id=first.conversation_id)


def test_collision_with_prior_turn_taints_both_attempts_across_restart(tmp_path):
    alpha, store = tenant(), state_store(tmp_path)
    first = run(store, alpha, sql_turn("same"), cite("same"))
    reopened = state_store(tmp_path)
    with pytest.raises(UngroundedAnswerError):
        run(reopened, alpha, sql_turn("same"), cite("same"), conversation_id=first.conversation_id)
    with pytest.raises(UngroundedAnswerError):
        run(state_store(tmp_path), alpha, cite("same"), conversation_id=first.conversation_id)


def test_retention_drops_proof_without_releasing_old_reference_for_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(app.state, "MAX_EVIDENCE_TURNS", 1)
    alpha, store = tenant(), state_store(tmp_path)
    first = run(store, alpha, sql_turn("old"), cite("old"))
    conversation = first.conversation_id
    run(store, alpha, sql_turn("recent"), cite("recent"), conversation_id=conversation)
    reopened = state_store(tmp_path)
    snapshot = reopened.get_conversation(alpha, conversation)
    assert [record.call_id for record in snapshot.evidence] == ["recent"]
    assert any(message.get("tool_call_id") == "old" for message in snapshot.messages)
    assert run(reopened, alpha, cite("recent"), conversation_id=conversation).trace == []
    with pytest.raises(UngroundedAnswerError):
        run(reopened, alpha, sql_turn("old"), cite("old"), conversation_id=conversation)


def test_failed_tool_attempt_cannot_be_reused_or_replaced_after_restart(tmp_path):
    alpha, store = tenant(), state_store(tmp_path)
    first = run(store, alpha, sql_turn("failed", sql="SELECT * FROM orders", params=[]),
                structured_turn("پرس‌وجو رد شد."))
    assert first.trace[0].status == "error"
    reopened = state_store(tmp_path)
    assert reopened.get_conversation(alpha, first.conversation_id).evidence == []
    with pytest.raises(UngroundedAnswerError):
        run(reopened, alpha, sql_turn("failed"), cite("failed"), conversation_id=first.conversation_id)


def test_payload_too_large_for_replay_still_works_in_current_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(app.evidence_history, "_MAX_RESULT_BYTES", 16)
    alpha, store = tenant(), state_store(tmp_path)
    first = run(store, alpha, sql_turn("large"), cite("large"))
    assert str(TOTAL) in first.answer
    reopened = state_store(tmp_path)
    snapshot = reopened.get_conversation(alpha, first.conversation_id)
    assert snapshot.evidence == []
    assert any(message.get("tool_call_id") == "large" for message in snapshot.messages)
    with pytest.raises(UngroundedAnswerError):
        run(reopened, alpha, cite("large"), conversation_id=first.conversation_id)
