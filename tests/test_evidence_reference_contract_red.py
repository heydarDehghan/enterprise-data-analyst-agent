"""Phase 4-C-C.3-E-A. Define the model-facing contract without a live provider."""

import json

import pytest

from app.agent import EnvChatCompletionsClient, UngroundedAnswerError
from app.sandbox import SandboxResult
from app.schemas import ChatRequest
from app.state import ConversationNotFoundError
from test_agent_loop import ScriptedLLM, call, numeric_slot, structured_turn, tool_result, tool_turn
from test_state_recovery import agent, state_store, tenant
from test_structured_answer_grounding_red import PAID_SQL, QUESTION, september_total


EVIDENCE_PREFIX = "KAMIDSO_EVIDENCE_V1 "
EXAMPLE_PREFIX = "KAMIDSO_SLOT_EXAMPLES_V1 "
SQL_REF = "call_paid_161783"
TICKET_REF = "call_tickets_f7a5"


@pytest.fixture(autouse=True)
def prohibit_live_provider(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Evidence contract tests require a mocked provider")

    monkeypatch.setattr(EnvChatCompletionsClient, "create", no_network)


def system_block(messages, prefix):
    assert messages[0]["role"] == "system"
    assert all(message["role"] != "system" for message in messages[1:])
    entries = [line.removeprefix(prefix) for line in messages[0]["content"].splitlines()
               if line.startswith(prefix)]
    if len(entries) != 1:
        pytest.fail(f"Expected one trusted system block beginning with {prefix}", pytrace=False)
    return json.loads(entries[0])


def manifest(messages):
    return system_block(messages, EVIDENCE_PREFIX)


def sql_call(ref=SQL_REF, sql=PAID_SQL):
    return call("query_sql", {"sql": sql}, ref)


def requested_slots(total, sql_ref=SQL_REF, ticket_ref=TICKET_REF):
    return {
        "tenant": {"source": "tenant_context", "value": "Alpha"},
        "period": {"source": "user_request", "value": "2026-09"},
        "start": {"source": "user_request", "type": "date", "field": "period_start", "value": "2026-09-01"},
        "end": {"source": "user_request", "type": "date", "field": "period_end", "value": "2026-09-30"},
        "revenue": numeric_slot(total, sql_ref, "total_revenue"),
        "count": numeric_slot(2, ticket_ref, "ticket_count", "count"),
        "ticket": {"type": "identifier", "value": "AT001", "evidence_ref": ticket_ref,
                   "evidence_field": "ticket_id", "row": 0},
        "date": {"type": "date", "value": "2026-10-03", "evidence_ref": ticket_ref,
                 "evidence_field": "created_at", "row": 0},
    }


def run(llm, *, store=None, subject=None, question=QUESTION, conversation_id=None, sandbox=None):
    return agent(store, llm, subject or tenant(), sandbox=sandbox).run(
        ChatRequest(question=question, conversation_id=conversation_id)
    )


def test_system_examples_distinguish_context_from_tool_slots():
    llm = ScriptedLLM(structured_turn("برای پاسخ به شواهد نیاز است."))
    run(llm)
    examples = system_block(llm.calls[0]["messages"], EXAMPLE_PREFIX)
    assert examples["tenant_label"]["source"] == "tenant_context"
    assert examples["user_period"]["source"] == "user_request"
    assert set(examples["tenant_label"]) == set(examples["user_period"]) == {"source", "value"}
    for label in ("period_start", "period_end"):
        slot = examples[label]
        assert set(slot) == {"source", "type", "field", "value"}
        assert (slot["source"], slot["type"], slot["field"]) == ("user_request", "date", label)
    for unit in ("toman", "count", "percent", "scalar"):
        slot = examples[unit]
        assert set(slot) == {"type", "unit", "value", "evidence_ref", "evidence_field"}
        assert (slot["type"], slot["unit"], slot["evidence_ref"]) == (
            "numeric", unit, "<actual_tool_call_id>",
        )
        assert isinstance(slot["value"], str)
    for kind in ("identifier", "date"):
        slot = examples[kind]
        assert set(slot) == {"type", "value", "evidence_ref", "evidence_field", "row"}
        assert (slot["type"], slot["evidence_ref"], slot["row"]) == (
            kind, "<actual_tool_call_id>", 0,
        )


def test_explicit_context_and_exact_tool_refs_are_discoverable_without_changing_tool_messages(september_total):
    template = (
        "  در شرکت {{tenant}}، بازهٔ {{period}} از {{start}} تا {{end}} بود.\n"
        "گزارش همان‌طور که خواستم: درآمد {{revenue}} تومان، {{count}} تیکت؛ "
        "تیکت {{ticket}} در {{date}} ثبت شد.  "
    )
    expected_answer = (
        f"  در شرکت Alpha، بازهٔ 2026-09 از 2026-09-01 تا 2026-09-30 بود.\n"
        f"گزارش همان‌طور که خواستم: درآمد {september_total} تومان، 2 تیکت؛ "
        "تیکت AT001 در 2026-10-03 ثبت شد.  "
    )
    llm = ScriptedLLM(
        tool_turn(call("get_tickets", {"customer_id": "A001"}, TICKET_REF), sql_call()),
        structured_turn(template, **requested_slots(september_total)),
    )
    response = run(llm)
    assert response.answer == expected_answer
    assert [entry.tool_name for entry in response.trace] == ["get_tickets", "query_sql"]
    messages = llm.calls[1]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "tool"]
    assert {t["id"] for t in messages[2]["tool_calls"]} == {TICKET_REF, SQL_REF}
    assert tool_result(messages, SQL_REF) == {"rows": [{"total_revenue": september_total}]}
    assert [row["ticket_id"] for row in tool_result(messages, TICKET_REF)["tickets"]] == ["AT001", "AT002"]
    initial = manifest(llm.calls[0]["messages"])
    assert initial["tools"] == []
    assert initial["context"] == {
        "tenant_context": {"value": "Alpha"},
        "user_request": {"period": "2026-09", "period_start": "2026-09-01", "period_end": "2026-09-30"},
    }
    current = manifest(messages)
    assert current["context"] == initial["context"]
    entries = {item["evidence_ref"]: item for item in current["tools"]}
    assert len(current["tools"]) == len(entries) == 2
    assert entries[SQL_REF] == {
        "evidence_ref": SQL_REF, "tool_name": "query_sql",
        "fields": [{"evidence_field": "total_revenue", "type": "numeric", "unit": "toman"}],
    }
    ticket = entries[TICKET_REF]
    assert ticket["tool_name"] == "get_tickets"
    fields = {(item["evidence_field"], item.get("row")): item for item in ticket["fields"]}
    assert len(fields) == len(ticket["fields"]) == 7
    assert fields[("ticket_count", None)] == {
        "evidence_field": "ticket_count", "type": "numeric", "unit": "count", "value": 2,
    }
    for row in (0, 1):
        for name, kind in (("ticket_id", "identifier"), ("customer_id", "identifier"), ("created_at", "date")):
            assert fields[(name, row)] == {"evidence_field": name, "type": kind, "row": row}


def test_manifest_advertises_verified_type_and_hides_unverified_failed_and_null_results():
    count_sql = PAID_SQL.replace("SUM(amount_toman)", "COUNT(*)")
    unverified = "WITH visible AS (SELECT amount_toman FROM paid_orders) SELECT SUM(amount_toman) AS total_revenue FROM visible"
    # Both of these queries successfully execute. The CTE has no verified
    # projection, and SQL NULL cannot fill a numeric slot.
    null_result = PAID_SQL + " AND customer_id = 'UNKNOWN'"
    llm = ScriptedLLM(
        tool_turn(
            sql_call("unverified", unverified), sql_call("denied", "SELECT * FROM orders"),
            call("get_schema", {}, "schema"), sql_call("null_sum", null_result),
            sql_call("valid_count", count_sql),
        ),
        structured_turn("اطلاعات بررسی شد."),
    )
    run(llm, question="اطلاعات پرداخت را بررسی کن")
    messages = llm.calls[1]["messages"]
    assert tool_result(messages, "unverified")["rows"]
    assert "error" in tool_result(messages, "denied")
    assert "schema" in tool_result(messages, "schema")
    assert tool_result(messages, "null_sum") == {"rows": [{"total_revenue": None}]}
    entries = manifest(messages)["tools"]
    assert entries == [{
        "evidence_ref": "valid_count", "tool_name": "query_sql",
        "fields": [{"evidence_field": "total_revenue", "type": "numeric", "unit": "count"}],
    }]


def test_python_manifest_uses_only_valid_named_scalar_metrics():
    class Sandbox:
        def execute(self, code, subject):
            return SandboxResult(
                '{"metrics":{"index":"3.25","not_finite":"NaN","wrong_type":42}}',
                "not evidence", 0,
            )

    ref = "call_sandbox_f28"
    llm = ScriptedLLM(
        tool_turn(call("execute_python", {"code": "print('fixture')"}, ref)),
        structured_turn("نتیجه {{index}} است.", index=numeric_slot("3.25", ref, "index", "scalar")),
    )
    assert run(llm, question="شاخص را محاسبه کن", sandbox=Sandbox()).answer == "نتیجه 3.25 است."
    assert manifest(llm.calls[1]["messages"])["tools"] == [{
        "evidence_ref": ref, "tool_name": "execute_python",
        "fields": [{"evidence_field": "index", "type": "numeric", "unit": "scalar"}],
    }]


def test_manifest_omits_same_total_from_wrong_month_and_ambiguous_refs(september_total):
    wrong_window = PAID_SQL.replace("2026-09-01", "2026-08-23").replace("2026-10-01", "2026-09-23")
    llm = ScriptedLLM(
        tool_turn(sql_call("wrong_window", wrong_window), sql_call("duplicate"), sql_call("duplicate"), sql_call()),
        structured_turn("درآمد {{amount}} تومان بود.", amount=numeric_slot(september_total, SQL_REF, "total_revenue")),
    )
    assert run(llm).answer == f"درآمد {september_total} تومان بود."
    messages = llm.calls[1]["messages"]
    assert tool_result(messages, "wrong_window") == {"rows": [{"total_revenue": september_total}]}
    assert [entry["evidence_ref"] for entry in manifest(messages)["tools"]] == [SQL_REF]


def test_reopened_owned_evidence_is_discoverable_without_requery(tmp_path, september_total):
    alpha = tenant()
    first = run(
        ScriptedLLM(tool_turn(sql_call()), structured_turn("گزارش ثبت شد.")),
        store=state_store(tmp_path), subject=alpha,
    )
    llm = ScriptedLLM(structured_turn("همان مبلغ {{amount}} تومان است.",
                                      amount=numeric_slot(september_total, SQL_REF, "total_revenue")))
    response = run(llm, store=state_store(tmp_path), subject=alpha,
                   question="مبلغ گزارش قبل چقدر بود؟", conversation_id=first.conversation_id)
    assert response.answer == f"همان مبلغ {september_total} تومان است."
    assert response.trace == [] and len(llm.calls) == 1
    assert [entry["evidence_ref"] for entry in manifest(llm.calls[0]["messages"])["tools"]] == [SQL_REF]


def test_fresh_beta_context_does_not_advertise_alpha_evidence(tmp_path):
    store = state_store(tmp_path)
    alpha = run(ScriptedLLM(tool_turn(sql_call()), structured_turn("گزارش ثبت شد.")), store=store)
    beta = tenant("demo-beta")
    forbidden = ScriptedLLM()
    with pytest.raises(ConversationNotFoundError):
        run(forbidden, store=store, subject=beta, conversation_id=alpha.conversation_id)
    assert forbidden.calls == []
    llm = ScriptedLLM(structured_turn("گزارش تازه آماده است."))
    run(llm, store=store, subject=beta)
    current = manifest(llm.calls[0]["messages"])
    assert current["context"]["tenant_context"] == {"value": "Beta"}
    assert current["tools"] == []


@pytest.mark.parametrize("invented_id", ["call_1", "call_2", "tenant_context", "user_request"])
def test_unissued_tool_ref_is_never_remapped(invented_id, september_total):
    llm = ScriptedLLM(
        tool_turn(sql_call()),
        structured_turn("درآمد {{amount}} تومان بود.",
                        amount=numeric_slot(september_total, invented_id, "total_revenue")),
    )
    with pytest.raises(UngroundedAnswerError) as failure:
        run(llm)
    assert failure.value.reason_code == "unsupported_number"


def test_generic_looking_ref_is_valid_when_it_is_the_real_tool_call_id(september_total):
    ref = "call_1"
    llm = ScriptedLLM(
        tool_turn(sql_call(ref)),
        structured_turn("درآمد {{amount}} تومان بود.", amount=numeric_slot(september_total, ref, "total_revenue")),
    )
    assert run(llm).answer == f"درآمد {september_total} تومان بود."


@pytest.mark.parametrize("bad_slot", [
    {"type": "text", "value": "Alpha", "evidence_ref": "tenant_context", "evidence_field": "tenant_label"},
    {"type": "date", "value": "2026-09", "evidence_ref": "user_request", "evidence_field": "period"},
    {"source": "tenant_context", "value": "Alpha", "evidence_ref": "tenant_context"},
])
def test_context_slots_do_not_accept_evidence_ref_form(bad_slot):
    llm = ScriptedLLM(structured_turn("اطلاعات {{slot}} ثبت شد.", slot=bad_slot))
    with pytest.raises(UngroundedAnswerError) as failure:
        run(llm)
    assert failure.value.reason_code == "unsupported_number"


def test_wrong_month_is_still_rejected_even_when_the_id_and_total_are_real(september_total):
    wrong = PAID_SQL.replace("2026-09-01", "2026-08-23").replace("2026-10-01", "2026-09-23")
    llm = ScriptedLLM(
        tool_turn(sql_call("wrong_window", wrong)),
        structured_turn("درآمد {{amount}} تومان بود.",
                        amount=numeric_slot(september_total, "wrong_window", "total_revenue")),
    )
    with pytest.raises(UngroundedAnswerError) as failure:
        run(llm)
    assert failure.value.reason_code == "invalid_date_range"
    assert tool_result(llm.calls[1]["messages"], "wrong_window") == {"rows": [{"total_revenue": september_total}]}
