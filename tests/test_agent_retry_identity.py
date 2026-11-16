"""Phase 4-B.1-A RED contract for semantic retry identity.

Acceptance criteria:
* Equivalent JSON representations of one validated operation share a failure
  counter, including an omitted optional ``params`` versus an empty list.
* JSON key order and insignificant whitespace cannot reset that counter.
* A failed operation runs at most three times: its first attempt and two retries.
* Distinct operations have independent counters; an unrelated successful tool
  call does not erase the first operation's failure history.
* The request still executes at most eight tool calls, including successes.
"""

from collections import Counter

import pytest

from app.agent import AgentLoop, AgentOperationFailed, AgentStepLimitExceeded
from app.schemas import ChatRequest

from test_agent_loop import ScriptedLLM, call, structured_turn, tool_turn


SQL = "SELECT COUNT(*) AS count FROM paid_orders WHERE customer_id = ?"


class RecordingTenant:
    state_key = "recording-tenant"

    def __init__(self, *, fail: bool) -> None:
        self.fail = fail
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def agent_query(self, sql, params=()):
        self.calls.append((sql, tuple(params)))
        if self.fail:
            raise RuntimeError("temporary database failure")
        return [{"count": 1}]


def raw_sql_call(raw: str, call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "query_sql", "arguments": raw},
    }


def run(llm: ScriptedLLM, tenant: RecordingTenant):
    return AgentLoop(llm=llm, tenant=tenant, schema_text="approved schema").run(
        ChatRequest(question="سفارش‌های پرداخت‌شده را بررسی کن")
    )


@pytest.mark.parametrize("raws", [
    (
        '{"sql":"' + SQL + '","params":["A001"]}',
        '{"params":["A001"],"sql":"' + SQL + '"}',
        '{ "sql" : "' + SQL + '" , "params" : [ "A001" ] }',
        '{"sql": "' + SQL + '", "params": ["A001"]}',
    ),
    (
        '{"sql":"SELECT COUNT(*) AS count FROM paid_orders"}',
        '{"params":[],"sql":"SELECT COUNT(*) AS count FROM paid_orders"}',
        '{ "sql" : "SELECT COUNT(*) AS count FROM paid_orders", "params" : [] }',
        '{"sql": "SELECT COUNT(*) AS count FROM paid_orders", "params": []}',
    ),
], ids=["reordered_keys_and_whitespace", "omitted_optional_params"])
def test_equivalent_json_uses_one_three_attempt_budget(raws):
    tenant = RecordingTenant(fail=True)
    llm = ScriptedLLM(
        *(tool_turn(raw_sql_call(raw, f"attempt-{n}"))
          for n, raw in enumerate(raws, 1)),
        structured_turn("عملیات ناموفق بود."),
    )

    with pytest.raises(AgentOperationFailed) as caught:
        run(llm, tenant)

    assert len(tenant.calls) == 3
    assert tenant.calls[0] == tenant.calls[1] == tenant.calls[2]
    assert len(llm.calls) == 3
    assert [entry.status for entry in caught.value.trace] == ["error"] * 3
    assert [entry.step for entry in caught.value.trace] == [1, 2, 3]


def test_unrelated_success_does_not_reset_failed_operation_budget():
    tenant = RecordingTenant(fail=True)
    failing = [
        '{"sql":"' + SQL + '","params":["A001"]}',
        '{"params":["A001"],"sql":"' + SQL + '"}',
        '{ "sql" : "' + SQL + '", "params" : [ "A001" ] }',
    ]
    llm = ScriptedLLM(
        tool_turn(raw_sql_call(failing[0], "failure-1")),
        tool_turn(call("get_schema", {}, "other-tool")),
        tool_turn(raw_sql_call(failing[1], "failure-2")),
        tool_turn(raw_sql_call(failing[2], "failure-3")),
        structured_turn("عملیات ناموفق بود."),
    )

    with pytest.raises(AgentOperationFailed) as caught:
        run(llm, tenant)

    assert len(tenant.calls) == 3
    assert len(llm.calls) == 4
    assert [entry.tool_name for entry in caught.value.trace] == [
        "query_sql", "get_schema", "query_sql", "query_sql",
    ]
    assert [entry.status for entry in caught.value.trace] == [
        "error", "success", "error", "error",
    ]


def test_distinct_failed_operations_keep_independent_retry_budgets():
    tenant = RecordingTenant(fail=True)
    sequence = ["A001", "A002", "A001", "A002", "A002"]
    llm = ScriptedLLM(
        *(tool_turn(call("query_sql", {"sql": SQL, "params": [customer]}, f"call-{n}"))
          for n, customer in enumerate(sequence, 1)),
        structured_turn("عملیات ناموفق بود."),
    )

    with pytest.raises(AgentOperationFailed) as caught:
        run(llm, tenant)

    assert Counter(params[0] for _, params in tenant.calls) == {"A001": 2, "A002": 3}
    assert len(llm.calls) == 5
    assert len(caught.value.trace) == 5
    assert all(entry.status == "error" for entry in caught.value.trace)


def test_eight_step_budget_still_applies_to_distinct_successful_calls():
    tenant = RecordingTenant(fail=False)
    llm = ScriptedLLM(*[
        tool_turn(call("query_sql", {"sql": SQL, "params": [f"A{n:03d}"]}, f"step-{n}"))
        for n in range(1, 10)
    ])

    with pytest.raises(AgentStepLimitExceeded) as caught:
        run(llm, tenant)

    assert len(llm.calls) == 9
    assert len(tenant.calls) == 8
    assert [params[0] for _, params in tenant.calls] == [f"A{n:03d}" for n in range(1, 9)]
    assert [entry.step for entry in caught.value.trace] == list(range(1, 9))
    assert all(entry.status == "success" for entry in caught.value.trace)
