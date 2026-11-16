"""B2 tool documentation and server-produced evidence at the Agent boundary."""

import json

import pytest

from app.agent import AgentLoop, UngroundedAnswerError, _arguments
from app.sandbox import SandboxResult
from app.schemas import ChatRequest
from app.secure_data import SecureDataAccess
from app.sql_evidence import sql_provenance
from app.structured_grounding import FieldKind
from test_agent_loop import (
    DATA, SCHEMA, TOKEN_MAP, ScriptedLLM, agent, call, numeric_slot, request, structured_turn,
    tool_result, tool_turn,
)


def test_documented_tool_examples_match_schemas_and_dispatch():
    expected = {
        "query_sql": ({"sql", "params"}, {"sql"}, ("Gregorian", "paid_orders", "rows", "AS")),
        "get_schema": (set(), set(), ("schema", "reference")),
        "get_tickets": ({"customer_id"}, set(), ("is_untrusted", "ticket_count", "Gregorian")),
        "execute_python": (
            {"code"}, {"code"},
            ("/data/customers.csv", "/data/paid_orders.csv", "/data/tickets.csv",
             "Docker", "20 seconds", "512 MiB", "64 PIDs", "exit_code", "scalar", "/output"),
        ),
    }
    examples = {}

    class Sandbox:
        def execute(self, code, tenant):
            assert code == examples["execute_python"]["code"]
            assert tenant.display_label == "Alpha"
            return SandboxResult('{"metrics":{"index":"3.25"}}', "", 0)

    def select(messages, tools):
        assert "exactly template and slots" in messages[0]["content"]
        assert "The authenticated tenant label is Alpha." in messages[0]["content"]
        assert len(tools) == len(expected)
        chosen = []
        for tool in tools:
            function = tool["function"]
            name = function["name"]
            properties, required, phrases = expected[name]
            parameters = function["parameters"]
            assert tool["type"] == "function"
            assert parameters["type"] == "object"
            assert set(parameters["properties"]) == properties
            assert set(parameters["required"]) == required
            assert parameters["additionalProperties"] is False
            description = function["description"]
            assert all(phrase in description for phrase in phrases)
            assert "backend authentication" in description and '{"error":' in description
            example, _ = json.JSONDecoder().raw_decode(description.split("Example: ", 1)[1])
            assert _arguments(name, json.dumps(example)) == example
            with pytest.raises(ValueError):
                _arguments(name, json.dumps({**example, "tenant": "Beta"}))
            for key in required:
                with pytest.raises(ValueError):
                    _arguments(name, json.dumps({k: v for k, v in example.items() if k != key}))
            examples[name] = example
            chosen.append(call(name, example, name))
        return tool_turn(*chosen)

    def finish(messages, tools):
        assert tool_result(messages, "get_schema") == {"schema": SCHEMA}
        assert tool_result(messages, "query_sql") == {"rows": [{"revenue": 53_250_000}]}
        assert [r["ticket_id"] for r in tool_result(messages, "get_tickets")["tickets"]] == [
            "AT001", "AT002",
        ]
        assert tool_result(messages, "execute_python") == {
            "stdout": '{"metrics":{"index":"3.25"}}', "stderr": "", "exit_code": 0, "artifacts": [],
        }
        return structured_turn(
            "  درآمد {{money}} تومان و شاخص {{index}} است.  ",
            money=numeric_slot(53_250_000, "query_sql", "revenue"),
            index=numeric_slot("3.25", "execute_python", "index", "scalar"),
        )

    tenant = SecureDataAccess(data_dir=DATA, token_map=TOKEN_MAP).authenticate("demo-alpha")
    response = AgentLoop(
        llm=ScriptedLLM(select, finish), tenant=tenant, schema_text=SCHEMA, sandbox=Sandbox(),
    ).run(ChatRequest(question="درآمد سپتامبر ۲۰۲۶ و شاخص را بررسی کن"))
    assert response.answer == "  درآمد 53250000 تومان و شاخص 3.25 است.  "
    assert all(item.status == "success" for item in response.trace)


def test_sql_provenance_qualifies_expression_and_window_independent_of_alias():
    sql = (
        "SELECT COUNT(*) AS total_revenue, SUM(amount_toman) AS n FROM paid_orders "
        "WHERE order_date >= ? AND order_date < ?"
    )
    columns, window = sql_provenance(sql, ["2026-09-01", "2026-10-01"], [
        {"total_revenue": 120, "n": 53_250_000},
    ])
    assert (columns["total_revenue"].kind, columns["total_revenue"].column) == (FieldKind.COUNT, "*")
    assert (columns["n"].kind, columns["n"].table, columns["n"].column) == (
        FieldKind.TOMAN, "paid_orders", "amount_toman",
    )
    assert (window.first, window.last) == ("2026-09-01", "2026-09-30")


@pytest.mark.parametrize("sql", [
    "WITH p AS (SELECT amount_toman FROM paid_orders) SELECT SUM(amount_toman) AS revenue FROM p",
    "SELECT SUM(amount_toman) AS revenue FROM paid_orders WHERE NOT (order_date >= '2026-09-01' "
    "AND order_date < '2026-10-01')",
    "SELECT SUM(amount_toman) AS revenue, COUNT(*) AS revenue FROM paid_orders",
])
def test_unverified_projections_can_execute_without_earning_numeric_evidence(sql):
    def finish(messages, tools):
        result = tool_result(messages, "sql")
        assert result["rows"]
        return structured_turn(
            "درآمد {{revenue}} تومان است.",
            revenue=numeric_slot(result["rows"][0]["revenue"], "sql", "revenue"),
        )

    llm = ScriptedLLM(tool_turn(call("query_sql", {"sql": sql}, "sql")), finish)
    with pytest.raises(UngroundedAnswerError) as rejected:
        agent(llm).run(request("درآمد پرداخت‌شده را محاسبه کن"))
    assert rejected.value.reason_code == "unsupported_number"
    assert rejected.value.trace[0].status == "success"
