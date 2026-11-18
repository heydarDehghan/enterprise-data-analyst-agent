"""Q2-F contract through the configured app, with a scripted LLM and real tenant data."""

import csv
import sqlite3
from collections import defaultdict
from contextlib import closing

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.secure_data import TenantContext
from test_agent_loop import DATA, ROOT, ScriptedLLM, call, structured_turn, tool_result, tool_turn
from test_evidence_reference_contract_red import manifest
from test_q2_ranking_sql_provenance_red import QUESTION, advertised_addresses, identifier
from test_structured_answer_grounding_red import forbid_live_provider, numeric


TOOL = "get_customer_purchase_drops"
RANK_REF = "q2-f-ranking"
BASE = {"baseline_month": "2026-09", "comparison_month": "2026-10", "limit": 5}
AMOUNTS = ("baseline_revenue_toman", "comparison_revenue_toman", "decrease_toman")
EMPTY_ANSWER = structured_turn("نتیجهٔ ابزار دریافت شد.")
REJECTED_TOOL = {"error": "Tool call rejected or failed"}


def ranked_from_source(database, args, *, paid_only=True):
    """Use the immutable source and compute the ranking independently in Python."""
    uri = (DATA / f"{database}.db").resolve().as_uri() + "?mode=ro&immutable=1"
    totals = defaultdict(lambda: [0, 0])
    baseline, comparison = args["baseline_month"], args["comparison_month"]
    with closing(sqlite3.connect(uri, uri=True)) as db:
        for customer, date, amount, status in db.execute(
            "SELECT customer_id, order_date, amount_toman, status FROM orders"
        ):
            month = date[:7]
            if month in (baseline, comparison) and (not paid_only or status == "paid"):
                totals[customer][int(month == comparison)] += amount
    rows = [
        {"customer_id": customer, "baseline_revenue_toman": first,
         "comparison_revenue_toman": second, "decrease_toman": first - second}
        for customer, (first, second) in totals.items() if first > second
    ]
    return sorted(rows, key=lambda row: (-row["decrease_toman"], row["customer_id"]))[:args["limit"]]


def tool_turn_for(args=BASE):
    return tool_turn(call(TOOL, args, RANK_REF))


def comparison_question(args):
    return f"افت خرید مشتریان را در {args['comparison_month']} نسبت به {args['baseline_month']} گزارش کن."


@pytest.fixture
def client_for(tmp_path):
    def client(llm):
        return TestClient(create_app(
            data_dir=DATA, token_map_path=ROOT / "demo_tokens.json.example", schema_path=DATA / "schema.md",
            state_db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts",
            docker_image="python:3.13-slim", llm_client=llm,
        ))

    return client


def chat(client, *, database="alpha", question=QUESTION, conversation_id=None):
    body = {"question": question}
    if conversation_id is not None:
        body["conversation_id"] = conversation_id
    return client.post("/chat", headers={"X-Demo-Token": f"demo-{database}"}, json=body)


def test_configured_agent_offers_typed_ranking_alongside_dynamic_sql_and_tickets(client_for):
    llm = ScriptedLLM(EMPTY_ANSWER)
    with client_for(llm) as client:
        assert chat(client).status_code == 200
    tools = {entry["function"]["name"]: entry["function"]["parameters"] for entry in llm.calls[0]["tools"]}
    assert {TOOL, "query_sql", "get_tickets", "get_schema", "execute_python"} <= tools.keys()
    schema = tools[TOOL]
    assert schema["type"] == "object" and schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"]) == set(BASE)
    for parameter in ("baseline_month", "comparison_month"):
        assert schema["properties"][parameter]["type"] == "string"
    limit = schema["properties"]["limit"]
    assert (limit["type"], limit["minimum"], limit["maximum"]) == ("integer", 1, 5)


@pytest.mark.parametrize(("database", "baseline", "comparison", "limit", "count"), [
    ("alpha", "2026-09", "2026-10", 5, 4),
    ("beta", "2026-09", "2026-10", 5, 0),
    ("alpha", "2026-08", "2026-10", 2, 2),
    ("beta", "2026-10", "2026-11", 1, 1),
    ("alpha", "2026-07", "2026-08", 3, 0),
], ids=["fewer_than_limit", "beta_same_period_no_declines", "other_months_and_limit", "zero_comparison_sales",
        "missing_baseline_sales"])
def test_purchase_drop_rows_match_independent_paid_data(client_for, monkeypatch, database, baseline,
                                                     comparison, limit, count):
    args = {"baseline_month": baseline, "comparison_month": comparison, "limit": limit}
    expected = ranked_from_source(database, args)
    assert len(expected) == count
    if database == "alpha" and args == BASE:
        assert expected != ranked_from_source(database, args, paid_only=False)

    def forbid_raw_query(*args, **kwargs):
        pytest.fail("The ranking tool must use the bounded paid-only SQL path")

    monkeypatch.setattr(TenantContext, "query", forbid_raw_query)
    llm = ScriptedLLM(tool_turn_for(args), EMPTY_ANSWER)
    with client_for(llm) as client:
        response = chat(client, database=database, question=comparison_question(args))
    assert response.status_code == 200
    assert tool_result(llm.calls[1]["messages"], RANK_REF) == {
        "baseline_month": baseline, "comparison_month": comparison, "rows": expected,
    }
    step, = response.json()["trace"]
    assert (step["tool_name"], step["status"]) == (TOOL, "success")


@pytest.mark.parametrize("args", [
    {"baseline_month": "2026-09", "limit": 5},
    {**BASE, "comparison_month": "2026-13"},
    {**BASE, "baseline_month": "0000-09"},
    {**BASE, "baseline_month": "2026-10"},
    {**BASE, "baseline_month": "2026-11"},
    {**BASE, "limit": 0},
    {**BASE, "limit": 6},
    {**BASE, "limit": True},
    {**BASE, "limit": 2.0},
    {**BASE, "tenant": "Beta"},
], ids=["missing_month", "bad_month", "bad_year", "same_month", "reversed_months", "zero_limit",
        "above_cap", "bool_limit", "float_limit", "forged_tenant"])
def test_invalid_ranking_args_fail_before_query_execution(client_for, monkeypatch, args):
    def unexpected_query(*args, **kwargs):
        pytest.fail("Invalid arguments reached tenant SQL")

    monkeypatch.setattr(TenantContext, "agent_query", unexpected_query)
    monkeypatch.setattr(TenantContext, "query", unexpected_query)
    llm = ScriptedLLM(tool_turn_for(args), EMPTY_ANSWER)
    with client_for(llm) as client:
        response = chat(client)
    assert response.status_code == 200
    step, = response.json()["trace"]
    assert (step["tool_name"], step["status"]) == (TOOL, "error")
    assert tool_result(llm.calls[1]["messages"], RANK_REF) == REJECTED_TOOL
    assert not any(tool["evidence_ref"] == RANK_REF for tool in manifest(llm.calls[1]["messages"])["tools"])


def test_ranked_rows_can_ground_persian_answer_with_separate_tickets(client_for):
    expected = ranked_from_source("alpha", BASE)
    ticket_ids = defaultdict(list)
    with (DATA / "alpha_tickets.csv").open(encoding="utf-8-sig", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            ticket_ids[row["customer_id"]].append(row["ticket_id"])

    def choose_ticket_lookups(messages, tools):
        rows = tool_result(messages, RANK_REF).get("rows", [])
        if not rows:
            return EMPTY_ANSWER
        return tool_turn(*(call("get_tickets", {"customer_id": row["customer_id"]}, f"tickets-{index}")
                           for index, row in enumerate(rows)))

    def draft(messages, tools):
        slots, sentences = {}, []
        for index, row in enumerate(expected):
            customer = row["customer_id"]
            slots[f"customer_{index}"] = identifier(customer, RANK_REF, "customer_id", index)
            for field in AMOUNTS:
                slots[f"{field}_{index}"] = numeric(str(row[field]), RANK_REF, field, row=index)
            line = (f"{{{{customer_{index}}}}} با فروش پایه {{{{baseline_revenue_toman_{index}}}}} "
                    f"و فروش مقایسه {{{{comparison_revenue_toman_{index}}}}}، "
                    f"{{{{decrease_toman_{index}}}}} تومان افت داشت.")
            ticket_ref = f"tickets-{index}"
            matches = sorted(ticket_ids[customer])
            records = tool_result(messages, ticket_ref)["tickets"]
            assert all(record["customer_id"] == customer for record in records)
            assert sorted(record["ticket_id"] for record in records) == matches
            if not records:
                slots[f"count_{index}"] = numeric("0", ticket_ref, "ticket_count", unit="count")
                line += f" تعداد تیکت‌ها {{{{count_{index}}}}} بود."
            for ticket_index, record in enumerate(records):
                name = f"ticket_{index}_{ticket_index}"
                slots[name] = identifier(record["ticket_id"], ticket_ref, "ticket_id", ticket_index)
                line += " تیکت {{" + name + "}} ثبت شده بود."
            sentences.append(line)
        return structured_turn("\n".join(sentences) + " این تیکت‌ها علت افت را ثابت نمی‌کنند.", **slots)

    llm = ScriptedLLM(tool_turn_for(), choose_ticket_lookups, draft)
    with client_for(llm) as client:
        response = chat(client)
    assert response.status_code == 200
    payload = response.json()
    assert [(item["tool_name"], item["status"]) for item in payload["trace"]] == (
        [(TOOL, "success")] + [("get_tickets", "success")] * len(expected)
    )
    messages = llm.calls[-1]["messages"]
    addresses = advertised_addresses(messages)
    for index, row in enumerate(expected):
        assert (RANK_REF, "customer_id", index, "identifier", None) in addresses
        assert all((RANK_REF, field, index, "numeric", "toman") in addresses for field in AMOUNTS)
        assert row["customer_id"] in payload["answer"]
        assert str(row["decrease_toman"]) in payload["answer"]
        assert all(ticket in payload["answer"] for ticket in ticket_ids[row["customer_id"]])
        ref = f"tickets-{index}"
        retrieved = tool_result(messages, ref)["tickets"]
        if not retrieved:
            assert (ref, "ticket_count", None, "numeric", "count") in addresses
        for ticket_index in range(len(retrieved)):
            row_index = ticket_index if len(retrieved) > 1 else None
            assert (ref, "ticket_id", row_index, "identifier", None) in addresses
    assert "تعداد تیکت‌ها 0 بود" in payload["answer"]
    assert payload["answer"].endswith("این تیکت‌ها علت افت را ثابت نمی‌کنند.")


@pytest.mark.parametrize("patch", [
    {"evidence_ref": "never-issued"}, {"row": 1}, {"evidence_field": "baseline_revenue_toman"},
    {"evidence_field": "customer_name"}, {"value": "999999999"}, {"unit": "count"},
], ids=["forged_id", "wrong_row", "wrong_field", "unknown_field", "wrong_value", "wrong_unit"])
def test_ranked_evidence_rejects_invalid_slot_addresses(client_for, patch):
    expected = ranked_from_source("alpha", BASE)
    claim = numeric(str(expected[0]["decrease_toman"]), RANK_REF, "decrease_toman", row=0)
    draft = structured_turn("مبلغ افت {{drop}} تومان است.", drop={**claim, **patch})
    llm = ScriptedLLM(tool_turn_for(), draft, draft)
    with client_for(llm) as client:
        response = chat(client)
    assert response.status_code == 502 and response.json() == {"detail": "Analysis answer rejected"}
    messages = llm.calls[1]["messages"]
    assert tool_result(messages, RANK_REF).get("rows") == expected
    assert (RANK_REF, "decrease_toman", 0, "numeric", "toman") in advertised_addresses(messages)


def test_ranking_for_wrong_months_cannot_supply_evidence_to_the_requested_period(client_for):
    args = {**BASE, "baseline_month": "2026-08"}
    rows = ranked_from_source("alpha", args)
    claim = numeric(str(rows[0]["decrease_toman"]), RANK_REF, "decrease_toman", row=0)
    llm = ScriptedLLM(tool_turn_for(args), structured_turn("مبلغ افت {{drop}} تومان است.", drop=claim))
    with client_for(llm) as client:
        assert chat(client).status_code == 502
    entries = manifest(llm.calls[1]["messages"])["tools"]
    assert not any(entry["evidence_ref"] == RANK_REF for entry in entries)


def test_ranking_evidence_cannot_cross_tenant_or_conversation(client_for):
    expected = ranked_from_source("alpha", BASE)
    claim = numeric(str(expected[0]["decrease_toman"]), RANK_REF, "decrease_toman", row=0)
    cited = structured_turn("مبلغ افت {{drop}} تومان است.", drop=claim)
    llm = ScriptedLLM(tool_turn_for(), cited, cited, cited)
    with client_for(llm) as client:
        issued = chat(client)
        assert issued.status_code == 200
        conversation_id = issued.json()["conversation_id"]
        assert chat(client, database="beta", conversation_id=conversation_id).status_code == 404
        assert len(llm.calls) == 2
        assert chat(client).status_code == 502
        assert chat(client, database="beta").status_code == 502
    assert len(llm.calls) == 4
    for exchange in llm.calls[2:]:
        assert not any(entry["evidence_ref"] == RANK_REF for entry in manifest(exchange["messages"])["tools"])
