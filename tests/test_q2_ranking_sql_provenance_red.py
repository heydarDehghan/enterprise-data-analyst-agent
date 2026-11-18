"""Q2-B: qualify the captured ranking through real tenant SQL and scripted replies."""

import json
from collections import defaultdict
from uuid import uuid4

import pytest

from app.agent import UngroundedAnswerError
from app.schemas import ChatRequest
from app.secure_data import SecureDataAccess
from app.structured_grounding import (
    EvidenceScope,
    FieldKind,
    GroundingContext,
    RecordedToolResult,
    UnknownEvidenceReference,
    validate_and_render,
)
from test_agent_loop import (
    DATA, TOKEN_MAP, ScriptedLLM, agent, call, structured_turn, tool_result, tool_turn,
)
from test_evidence_reference_contract_red import manifest
from test_structured_answer_grounding_red import alpha, forbid_live_provider, numeric, source_paid_orders


QUESTION = (
    "۵ مشتری با بیشترین افت مبلغ خرید\n"
    "در اکتبر نسبت به سپتامبر ۲۰۲۶\n"
    "را مشخص کن. تیکت‌های مرتبط را\n"
    "بررسی کن."
)
# The exact query from the Q2 capture. Keep it independent of local runtime artifacts.
RANKING_SQL = """SELECT customer_id,
  SUM(CASE WHEN substr(order_date,1,7)='2026-09' THEN amount_toman ELSE 0 END) AS sep_revenue,
  SUM(CASE WHEN substr(order_date,1,7)='2026-10' THEN amount_toman ELSE 0 END) AS oct_revenue,
  SUM(CASE WHEN substr(order_date,1,7)='2026-09' THEN amount_toman ELSE 0 END) - SUM(CASE WHEN substr(order_date,1,7)='2026-10' THEN amount_toman ELSE 0 END) AS drop_toman
FROM paid_orders
WHERE order_date >= '2026-09-01' AND order_date < '2026-11-01'
GROUP BY customer_id
HAVING drop_toman > 0
ORDER BY drop_toman DESC
LIMIT 5"""

TICKET_IDS = {"A004": ["AT004"], "A001": ["AT001", "AT002"], "A007": [], "A002": ["AT003"]}
AMOUNT_FIELDS = ("sep_revenue", "oct_revenue", "drop_toman")
RANKING_FIELDS = {"customer_id": FieldKind.IDENTIFIER, **dict.fromkeys(AMOUNT_FIELDS, FieldKind.TOMAN)}
SHORT_WINDOW_SQL = RANKING_SQL.replace("2026-09-01", "2026-09-02")


@pytest.fixture
def expected_ranking():
    # Read paid raw orders from immutable SQLite and compute independently in Python.
    totals = defaultdict(lambda: [0, 0])
    for order_date, customer, amount in source_paid_orders():
        if "2026-09-01" <= order_date < "2026-10-01":
            totals[customer][0] += amount
        elif "2026-10-01" <= order_date < "2026-11-01":
            totals[customer][1] += amount
    ranked = sorted(totals.items(), key=lambda entry: entry[1][0] - entry[1][1], reverse=True)
    rows = [
        {"customer_id": customer, "sep_revenue": sep, "oct_revenue": oct_, "drop_toman": sep - oct_}
        for customer, (sep, oct_) in ranked if sep > oct_
    ][:5]
    assert [row["customer_id"] for row in rows] == list(TICKET_IDS)
    return rows


def ranking_turn(ref, sql=RANKING_SQL):
    return tool_turn(call("query_sql", {"sql": sql, "params": []}, ref))


def identifier(value, ref, field, row):
    return {"type": "identifier", "value": value, "evidence_ref": ref,
            "evidence_field": field, "row": row}


def advertised_addresses(messages):
    return {
        (tool["evidence_ref"], field["evidence_field"], field.get("row"), field["type"], field.get("unit"))
        for tool in manifest(messages)["tools"] for field in tool["fields"]
    }


def ranking_addresses(ref, rows):
    return {
        (ref, field, index, *kind.value)
        for index in range(len(rows))
        for field, kind in RANKING_FIELDS.items()
    }


def test_q2_advertises_each_verified_customer_and_amount(alpha, expected_ranking):
    ref = uuid4().hex
    llm = ScriptedLLM(ranking_turn(ref), structured_turn("رتبه‌بندی آماده است."))
    response = agent(llm, tenant=alpha).run(ChatRequest(question=QUESTION))

    assert response.trace[0].status == "success"
    messages = llm.calls[-1]["messages"]
    assert tool_result(messages, ref) == {"rows": expected_ranking}
    assert {address for address in advertised_addresses(messages) if address[0] == ref} == (
        ranking_addresses(ref, expected_ranking)
    )


def test_q2_renders_all_four_rows_with_separate_ticket_citations(alpha, expected_ranking):
    rank_ref = uuid4().hex
    ticket_refs = {customer: uuid4().hex for customer in TICKET_IDS}
    slots = {}
    lines = []
    expected_ticket_addresses = set()
    for index, row in enumerate(expected_ranking):
        customer = row["customer_id"]
        ticket_ref = ticket_refs[customer]
        slots[f"customer_{index}"] = identifier(customer, rank_ref, "customer_id", index)
        for field in AMOUNT_FIELDS:
            slots[f"{field}_{index}"] = numeric(str(row[field]), rank_ref, field, row=index)
        line = (
            f"{{{{customer_{index}}}}}، سپتامبر {{{{sep_revenue_{index}}}}} تومان، "
            f"اکتبر {{{{oct_revenue_{index}}}}} تومان، افت {{{{drop_toman_{index}}}}} تومان."
        )
        if TICKET_IDS[customer]:
            names = []
            for ticket_row, ticket_id in enumerate(TICKET_IDS[customer]):
                name = f"ticket_{index}_{ticket_row}"
                slots[name] = identifier(ticket_id, ticket_ref, "ticket_id", ticket_row)
                names.append("{{" + name + "}}")
                # A one-row ticket result is advertised with the default row address.
                advertised_row = ticket_row if len(TICKET_IDS[customer]) > 1 else None
                expected_ticket_addresses.add((ticket_ref, "ticket_id", advertised_row, "identifier", None))
            line += " تیکت‌ها " + " و ".join(names) + "."
        else:
            slots["no_tickets"] = numeric("0", ticket_ref, "ticket_count", "count")
            expected_ticket_addresses.add((ticket_ref, "ticket_count", None, "numeric", "count"))
            line += " تعداد تیکت‌ها {{no_tickets}} است."
        lines.append(line)
    template = "\n".join(lines)

    def finish(messages, tools):
        assert tool_result(messages, rank_ref) == {"rows": expected_ranking}
        for customer, ref in ticket_refs.items():
            tickets = tool_result(messages, ref)["tickets"]
            assert [(t["customer_id"], t["ticket_id"]) for t in tickets] == [
                (customer, ticket) for ticket in TICKET_IDS[customer]
            ]
        return structured_turn(template, **slots)

    llm = ScriptedLLM(
        ranking_turn(rank_ref),
        tool_turn(*(call("get_tickets", {"customer_id": customer}, ref)
                    for customer, ref in ticket_refs.items())),
        finish,
    )
    response = agent(llm, tenant=alpha).run(ChatRequest(question=QUESTION))

    expected = template
    for name, slot in slots.items():
        expected = expected.replace("{{" + name + "}}", slot["value"])
    assert response.answer == expected
    expected_trace = [("query_sql", "success")] + [("get_tickets", "success")] * len(ticket_refs)
    assert [(step.tool_name, step.status) for step in response.trace] == expected_trace
    expected_addresses = ranking_addresses(rank_ref, expected_ranking) | expected_ticket_addresses
    assert expected_addresses <= advertised_addresses(llm.calls[-1]["messages"])


@pytest.mark.parametrize("sql", [
    RANKING_SQL.replace("2026-11", "2026-12").replace("2026-10", "2026-11").replace("2026-09", "2026-10"),
    SHORT_WINDOW_SQL,
    RANKING_SQL.replace(" END) - SUM", " END) + SUM"),
], ids=["other_complete_months", "one_day_short_even_if_total_matches", "addition_under_drop_alias"])
def test_q2_does_not_advertise_or_cite_successful_altered_sql(alpha, expected_ranking, sql):
    ref = uuid4().hex

    def finish(messages, tools):
        rows = tool_result(messages, ref)["rows"]
        assert set(rows[0]) == RANKING_FIELDS.keys()
        if sql == SHORT_WINDOW_SQL:
            assert rows == expected_ranking
        assert not any(address[0] == ref for address in advertised_addresses(messages))
        return structured_turn("مبلغ افت {{drop}} تومان است.",
                               drop=numeric(str(rows[0]["drop_toman"]), ref, "drop_toman", row=0))

    with pytest.raises(UngroundedAnswerError) as failure:
        agent(ScriptedLLM(ranking_turn(ref, sql), finish), tenant=alpha).run(ChatRequest(question=QUESTION))
    assert failure.value.trace[0].status == "success"


@pytest.mark.parametrize("case", ["swapped_row", "fabricated_amount"])
def test_q2_cannot_cite_an_amount_outside_the_selected_row(alpha, expected_ranking, case):
    ref = uuid4().hex
    drop = expected_ranking[0]["drop_toman"]
    slot = (numeric(str(drop), ref, "drop_toman", row=1) if case == "swapped_row"
            else numeric(str(drop + 1), ref, "drop_toman", row=0))
    assert slot["value"] != str(expected_ranking[slot["row"]]["drop_toman"])
    llm = ScriptedLLM(ranking_turn(ref), structured_turn("مبلغ افت {{drop}} تومان است.", drop=slot))

    with pytest.raises(UngroundedAnswerError) as failure:
        agent(llm, tenant=alpha).run(ChatRequest(question=QUESTION))
    assert failure.value.reason_code == "unsupported_number"
    assert tool_result(llm.calls[-1]["messages"], ref)["rows"] == expected_ranking


@pytest.mark.parametrize("foreign_scope", ["tenant", "conversation"])
def test_q2_cannot_cite_ranking_evidence_from_a_foreign_scope(alpha, foreign_scope):
    beta = SecureDataAccess(data_dir=DATA, token_map=TOKEN_MAP).authenticate("demo-beta")
    owner = EvidenceScope(alpha.state_key, "alpha-ranking")
    other = (EvidenceScope(beta.state_key, owner.conversation_id) if foreign_scope == "tenant"
             else EvidenceScope(owner.tenant_key, "other-alpha-conversation"))
    rows = alpha.agent_query(RANKING_SQL)
    ref = uuid4().hex
    # Even an otherwise authoritative ranking record cannot cross a scope boundary.
    record = RecordedToolResult(owner, ref, "query_sql", True, {"rows": rows},
                                {"drop_toman": FieldKind.TOMAN})
    raw = json.dumps({"template": "مبلغ افت {{drop}} تومان است.",
                      "slots": {"drop": numeric(str(rows[0]["drop_toman"]), ref, "drop_toman", row=0)}})

    with pytest.raises(UnknownEvidenceReference) as failure:
        validate_and_render(raw, context=GroundingContext(other), tool_results=[record])
    assert failure.value.reference == ref
