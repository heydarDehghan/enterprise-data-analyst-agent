"""Phase 4-C-C.4-A. Exercise the live diagnostic gates with no provider traffic."""

import csv
import json
import sqlite3
from contextlib import closing, contextmanager
from types import SimpleNamespace

import pytest

from app.agent import EnvChatCompletionsClient
from app.main import app
from live_tests import conftest as live_fixtures
from live_tests.llm_diagnostics import capture_run
from test_agent_loop import DATA, call, numeric_slot, structured_turn, tool_turn
from test_evidence_reference_contract_red import manifest
from test_llm_io_diagnostics_red import (
    ALPHA_NODE, CAPTURE_FLAG, CAPTURE_PATH, DEMO_SECRET, QUESTION, body_bytes,
    fake_provider, isolated_provider_environment, wire,
)
from test_live_llm_integration import SQL, runtime_client


BETA_NODE = (
    "live_tests/test_agent_end_to_end_live.py::"
    "test_live_chat_calls_paid_sql_and_tenant_tickets[demo-beta-beta.db-beta_tickets.csv-B001-A]"
)
UNAPPROVED_NODE = BETA_NODE.replace("-B001-A]", "-B002-A]")
ALPHA_SQL = "call_alpha_diagnostic_sql"
ALPHA_TICKETS = "call_alpha_diagnostic_tickets"
BETA_SQL = "call_beta_diagnostic_sql"
BETA_TICKETS = "call_beta_diagnostic_tickets"


def fixture_scope(monkeypatch, nodes):
    items = [SimpleNamespace(nodeid=node) for node in nodes]
    request = SimpleNamespace(session=SimpleNamespace(items=items), node=items[0] if items else None)
    return contextmanager(live_fixtures.provider_environment.__wrapped__)(monkeypatch, request)


@pytest.mark.parametrize("node", [ALPHA_NODE, BETA_NODE], ids=["alpha", "beta"])
def test_fixture_accepts_one_exact_approved_node(monkeypatch, node):
    monkeypatch.setenv(CAPTURE_FLAG, "1")
    seen = []

    @contextmanager
    def record_selection(*, node_id, monkeypatch):
        seen.append(node_id)
        yield

    monkeypatch.setattr(live_fixtures, "capture_run", record_selection)
    with fixture_scope(monkeypatch, [node]):
        assert seen == [node]


@pytest.mark.parametrize("nodes", [
    [],
    [ALPHA_NODE, BETA_NODE],
    [BETA_NODE, BETA_NODE],
    [BETA_NODE, "live_tests/test_provider_live.py::test_live_https_provider_returns_chat_completion"],
    [BETA_NODE.split("[", 1)[0]],
    [UNAPPROVED_NODE],
], ids=["none", "alpha_and_beta", "duplicate", "mixed", "unparameterized", "lookalike"])
def test_fixture_rejects_invalid_selection_before_config_or_capture(monkeypatch, nodes):
    monkeypatch.setenv(CAPTURE_FLAG, "1")

    def unexpected(*args, **kwargs):
        raise AssertionError("Selection must be rejected before configuration or recorder setup")

    monkeypatch.setattr(live_fixtures, "load_llm_config", unexpected)
    monkeypatch.setattr(live_fixtures, "capture_run", unexpected)
    with pytest.raises(pytest.fail.Exception):
        with fixture_scope(monkeypatch, nodes):
            raise AssertionError("Invalid selection reached the test body")


def test_direct_recorder_rejects_beta_node_lookalike(tmp_path, monkeypatch):
    path = tmp_path / "lookalike.json"
    monkeypatch.setenv(CAPTURE_FLAG, "1")
    monkeypatch.setenv(CAPTURE_PATH, str(path))
    with pytest.raises(ValueError):
        with capture_run(node_id=UNAPPROVED_NODE, monkeypatch=monkeypatch):
            raise AssertionError("An unapproved node must not enter the capture context")
    assert not path.exists()


def reference_total(database):
    uri = (DATA / database).resolve().as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        return connection.execute(
            "SELECT SUM(amount_toman) FROM orders WHERE status = 'paid' "
            "AND order_date >= '2026-09-01' AND order_date < '2026-10-01'"
        ).fetchone()[0]


def reference_tickets(filename, customer_id):
    with (DATA / filename).open(encoding="utf-8-sig", newline="") as file:
        return {row["ticket_id"] for row in csv.DictReader(file) if row["customer_id"] == customer_id}


def replies_for(customer, sql_id, tickets_id, total, ticket_count, *, revenue_ref=None):
    selected = tool_turn(
        call("query_sql", {"sql": SQL, "params": ["2026-09"]}, sql_id),
        call("get_tickets", {"customer_id": customer}, tickets_id),
    )
    final = structured_turn(
        "درآمد {{revenue}} تومان بود و {{count}} تیکت بازگردانده شد.",
        revenue=numeric_slot(total, revenue_ref or sql_id, "revenue"),
        count=numeric_slot(ticket_count, tickets_id, "ticket_count", "count"),
    )
    replies = [(200, wire(selected, cost=0.00017)), (200, wire(final, cost=0.00019))]
    return replies, final["choices"][0]["message"]["content"]


@pytest.fixture
def beta_replies():
    total = reference_total("beta.db")
    assert total == 106_500_000
    tickets = reference_tickets("beta_tickets.csv", "B001")
    assert tickets and all(ticket.startswith("BT") for ticket in tickets)
    replies, final = replies_for("B001", BETA_SQL, BETA_TICKETS, total, len(tickets))
    return total, tickets, replies, final


def run_chat(client, monkeypatch, *, token, node, path, replies, enabled, foreign=None):
    with monkeypatch.context() as patch:
        if enabled:
            patch.setenv(CAPTURE_FLAG, "1")
        else:
            patch.delenv(CAPTURE_FLAG, raising=False)
        patch.setenv(CAPTURE_PATH, str(path))
        seen = fake_provider(patch, replies)
        headers = {"X-Demo-Token": token}
        with capture_run(node_id=node, monkeypatch=patch):
            if foreign is not None:
                denied = client.post("/chat", headers=headers, json={
                    "question": QUESTION, "conversation_id": foreign,
                })
                assert denied.status_code == 404
                assert seen == []
            response = client.post("/chat", headers=headers, json={"question": QUESTION})
    return response, seen


def load_capture(path, node):
    content = path.read_bytes()
    assert DEMO_SECRET.encode("ascii") not in content
    capture = json.loads(content)
    assert capture["test_node_id"] == node and capture["capture_errors"] == []
    return capture


def test_beta_capture_disabled_by_default(tmp_path, monkeypatch, beta_replies):
    total, tickets, replies, _ = beta_replies
    path = tmp_path / "disabled.json"
    with runtime_client(tmp_path, monkeypatch, EnvChatCompletionsClient()) as client:
        response, seen = run_chat(client, monkeypatch, token="demo-beta", node=BETA_NODE,
                                  path=path, replies=replies, enabled=False)
    assert response.status_code == 200 and len(seen) == 2
    assert response.json()["answer"] == f"درآمد {total} تومان بود و {len(tickets)} تیکت بازگردانده شد."
    assert not path.exists()


def test_beta_capture_preserves_exact_provider_and_grounding_observations(tmp_path, monkeypatch, beta_replies):
    total, tickets, replies, final = beta_replies
    path = tmp_path / "beta.json"
    with runtime_client(tmp_path, monkeypatch, EnvChatCompletionsClient()) as client:
        baseline, baseline_seen = run_chat(client, monkeypatch, token="demo-beta", node=BETA_NODE,
                                           path=tmp_path / "disabled.json", replies=replies, enabled=False)
        result, seen = run_chat(client, monkeypatch, token="demo-beta", node=BETA_NODE,
                                path=path, replies=replies, enabled=True)
    assert baseline.status_code == result.status_code == 200
    assert result.json()["answer"] == baseline.json()["answer"]
    assert result.json()["trace"] == baseline.json()["trace"]
    assert seen == baseline_seen and len(seen) == 2
    data = load_capture(path, BETA_NODE)
    first, second = data["provider_calls"]
    for entry, request, (status, response) in zip((first, second), seen, replies):
        assert body_bytes(entry["request"]) == request
        assert body_bytes(entry["response"]) == response
        assert entry["response"]["status_code"] == status
        assert entry["request"]["headers"]["authorization"] == "[REDACTED]"
        assert entry["trace_id"] == result.json()["trace_id"]
        assert entry["conversation_id"] == result.json()["conversation_id"]

    messages = json.loads(seen[1])["messages"]
    evidence = manifest(messages)
    assert evidence["context"]["tenant_context"] == {"value": "Beta"}
    assert {item["evidence_ref"] for item in evidence["tools"]} == {BETA_SQL, BETA_TICKETS}
    returned = {item["tool_call_id"]: item["content"] for item in messages if item["role"] == "tool"}
    assert json.loads(returned[BETA_SQL]) == {"rows": [{"revenue": total}]}
    rows = json.loads(returned[BETA_TICKETS])["tickets"]
    assert {row["ticket_id"] for row in rows} == tickets
    assert all(row["customer_id"] == "B001" and row["is_untrusted"] is True for row in rows)
    assert [(item["tool_call_id"], item["status"]) for item in data["tool_executions"]] == [
        (BETA_SQL, "success"), (BETA_TICKETS, "success"),
    ]
    assert {item["tool_call_id"]: item["result_text"] for item in data["tool_executions"]} == returned
    assert all(item["provider_call_id"] == first["call_id"] for item in data["tool_executions"])
    decision, = data["grounding"]
    assert decision["provider_call_id"] == second["call_id"]
    assert decision["raw_final_content"] == final
    assert decision["rendered_answer"] == result.json()["answer"]
    assert decision["rejection_reason"] is None
    assert {slot["slot"]: (slot["decision"], slot["evidence_ref"], slot["authoritative_value"])
            for slot in decision["slot_decisions"]} == {
        "revenue": ("accepted", BETA_SQL, total), "count": ("accepted", BETA_TICKETS, len(tickets)),
    }


def test_alpha_and_beta_captures_stay_separate_and_beta_cannot_use_alpha_evidence(
    tmp_path, monkeypatch, beta_replies,
):
    total, tickets, _, _ = beta_replies
    alpha_path, beta_path = tmp_path / "alpha.json", tmp_path / "beta.json"
    alpha_replies, _ = replies_for(
        "A001", ALPHA_SQL, ALPHA_TICKETS, reference_total("alpha.db"),
        len(reference_tickets("alpha_tickets.csv", "A001")),
    )
    foreign_replies, _ = replies_for("B001", BETA_SQL, BETA_TICKETS, total, len(tickets), revenue_ref=ALPHA_SQL)
    with runtime_client(tmp_path, monkeypatch, EnvChatCompletionsClient()) as client:
        alpha, _ = run_chat(client, monkeypatch, token="demo-alpha", node=ALPHA_NODE,
                            path=alpha_path, replies=alpha_replies, enabled=True)
        assert alpha.status_code == 200
        previous = alpha_path.read_bytes()
        denied, sent = run_chat(client, monkeypatch, token="demo-beta", node=BETA_NODE,
                                path=beta_path, replies=foreign_replies, enabled=True,
                                foreign=alpha.json()["conversation_id"])
        assert denied.status_code == 502 and denied.json() == {"detail": "Analysis answer rejected"}
        assert len(sent) == 2
        store = app.state.conversation_store
        beta = app.state.secure_data.authenticate("demo-beta")
        alpha_capture = load_capture(alpha_path, ALPHA_NODE)
        beta_capture = load_capture(beta_path, BETA_NODE)
        assert alpha_capture["run_id"] != beta_capture["run_id"]
        assert alpha_path.read_bytes() == previous
        conversation = beta_capture["grounding"][0]["conversation_id"]
        assert conversation != alpha.json()["conversation_id"]
        turn, = store.get_conversation(beta, conversation).turns
        assert turn.status == "failed" and turn.answer is None and turn.failure_code == "unsupported_number"

        evidence = manifest(json.loads(sent[1])["messages"])
        assert {record["evidence_ref"] for record in evidence["tools"]} == {BETA_SQL, BETA_TICKETS}
        decision, = beta_capture["grounding"]
        rejected = decision["slot_decisions"][0]
        assert (rejected["decision"], rejected["evidence_ref"], rejected["authoritative_value"]) == (
            "rejected", ALPHA_SQL, None,
        )
