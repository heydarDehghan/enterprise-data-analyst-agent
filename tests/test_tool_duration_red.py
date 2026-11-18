"""Measure executed tool attempts and preserve the measurement in SQLite."""

from types import SimpleNamespace

import pytest

import app.agent as agent_module
from app.schemas import ChatRequest
from test_agent_loop import ScriptedLLM, call, structured_turn, tool_result, tool_turn
from test_state_recovery import agent, state_store, tenant


@pytest.mark.parametrize(
    ("sql", "status", "expected_ms"),
    [
        ("SELECT COUNT(*) AS count FROM paid_orders", "success", 7),
        ("DELETE FROM paid_orders", "error", 3),
    ],
    ids=["successful-execution", "failed-execution"],
)
def test_executed_tool_duration_is_measured_and_persisted(
    tmp_path, monkeypatch, sql, status, expected_ms,
):
    store, alpha = state_store(tmp_path), tenant()
    original_query = alpha.agent_query
    elapsed_ns = 1_000_000_000

    def timed_query(sql, params=()):
        nonlocal elapsed_ns
        try:
            return original_query(sql, params)
        finally:
            elapsed_ns += expected_ms * 1_000_000

    # Replace the Agent's clock locally so elapsed time needs no wall-clock wait.
    monkeypatch.setattr(
        agent_module, "time", SimpleNamespace(monotonic_ns=lambda: elapsed_ns), raising=False,
    )
    monkeypatch.setattr(alpha, "agent_query", timed_query)

    def report_result(messages, tools):
        result = tool_result(messages, "timed-query")
        if status == "success":
            assert result["rows"][0]["count"] > 0
        else:
            assert result == {"error": "Tool call rejected or failed"}
        return structured_turn("نتیجه اجرای ابزار بررسی شد.")

    llm = ScriptedLLM(tool_turn(call("query_sql", {"sql": sql}, "timed-query")), report_result)
    response = agent(store, llm, alpha).run(ChatRequest(question="داده‌های سفارش را بررسی کن"))

    step, = response.trace
    assert (step.tool_name, step.status) == ("query_sql", status)
    assert step.error == (None if status == "success" else "Tool call rejected or failed")
    reopened = state_store(tmp_path).get_conversation(alpha, response.conversation_id)
    turn, = reopened.turns
    assert turn.trace_id == response.trace_id
    assert turn.trace == response.trace
    assert type(step.duration_ms) is int and step.duration_ms == expected_ms
