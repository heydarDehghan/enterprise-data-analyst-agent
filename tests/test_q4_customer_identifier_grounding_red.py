"""Q4-A RED contracts for request-owned customer identifiers and ticket evidence."""

import hashlib
import json

import pytest

from app.agent import UngroundedAnswerError
from app.schemas import ChatRequest
from app.structured_grounding import (
    EvidenceScope, GroundingContext, RecordedToolResult, StructuredGroundingError, validate_and_render,
)
from test_agent_loop import ScriptedLLM, agent, call, final_turn, numeric_slot, structured_turn, tool_result, tool_turn
from test_evidence_reference_contract_red import manifest
from test_followup_period_context_red import request_date
from test_structured_answer_grounding_red import alpha, forbid_live_provider


QUESTION = "برای مشتری A007 علت افت خرید\nرا با قطعیت اعلام کن."
TICKET_REF = "call_11dc2012bce741b995cf3f22"
MONTHLY_SQL = (
    "SELECT substr(order_date,1,7) AS month, COUNT(*) AS paid_count, SUM(amount_toman) AS revenue "
    "FROM paid_orders WHERE customer_id = ? GROUP BY substr(order_date,1,7) ORDER BY month"
)
CATEGORY_SQL = (
    "SELECT category, substr(order_date,1,7) AS month, COUNT(*) AS paid_count, SUM(amount_toman) AS revenue "
    "FROM paid_orders WHERE customer_id = ? GROUP BY category, substr(order_date,1,7) ORDER BY category, month"
)
# Original Q4-Capture assistant content, including both leading newlines.
CAPTURED_FINAL = """

{
  "template": "با داده‌های موجود نمی‌توان علت افت خرید برای مشتری {{customer_id}} را با قطعیت اعلام کرد. بررسی سفارش‌های پرداختی نشان می‌دهد تعداد سفارش‌ها در ماه‌های اخیر تغییر نکرده و افت درآمد صرفاً به کاهش یکنواخت مبلغ هر سفارش در همه دسته‌ها بازمی‌گردد. از سوی دیگر هیچ تیکت پشتیبانی برای این مشتری ثبت نشده است (تعداد تیکت: {{ticket_count}})؛ بنابراین هیچ علت ریشه‌ای تأییدشده‌ای در اطلاعات سیستم وجود ندارد. برای اعلام یک علت قطعی، به اطلاعات تکمیلی مانند تیکت‌ها یا گزارش‌های مالی نیاز است.",
  "slots": {
    "customer_id": {
      "type": "identifier",
      "value": "A007",
      "source": "user_request"
    },
    "ticket_count": {
      "type": "numeric",
      "value": "0",
      "evidence_ref": "call_11dc2012bce741b995cf3f22",
      "evidence_field": "ticket_count",
      "unit": "count"
    }
  }
}"""
CAPTURED_SHA256 = "ebf982be7cb7f7b4ff964626f16a2e44b0b6669806ed217b130503496fca2d33"


def customer_slot(value):
    return {"type": "identifier", "value": value, "source": "user_request"}


def ticket_lookup(customer):
    return tool_turn(call("get_tickets", {"customer_id": customer}, TICKET_REF))


def test_exact_q4_reply_renders_requested_customer_and_verified_zero_tickets(alpha):
    assert hashlib.sha256(CAPTURED_FINAL.encode("utf-8")).hexdigest() == CAPTURED_SHA256
    expected = json.loads(CAPTURED_FINAL)["template"].replace("{{customer_id}}", "A007").replace("{{ticket_count}}", "0")
    llm = ScriptedLLM(
        tool_turn(call("get_schema", {}, "call_51bc61fdddc54cdfac70ced1")),
        tool_turn(
            call("query_sql", {"sql": MONTHLY_SQL, "params": ["A007"]}, "call_ceb85bdead884f1bb8fc186c"),
            call("get_tickets", {"customer_id": "A007"}, TICKET_REF),
        ),
        tool_turn(call("query_sql", {"sql": CATEGORY_SQL, "params": ["A007"]}, "call_7788995965a543fe881350c6")),
        final_turn(CAPTURED_FINAL),
    )
    response = agent(llm, tenant=alpha).run(ChatRequest(question=QUESTION))

    assert response.answer == expected
    assert [(entry.tool_name, entry.status) for entry in response.trace] == [
        ("get_schema", "success"), ("query_sql", "success"), ("get_tickets", "success"), ("query_sql", "success"),
    ]
    assert tool_result(llm.calls[-1]["messages"], TICKET_REF) == {"tickets": []}
    context = manifest(llm.calls[-1]["messages"])["context"].get("user_request", {})
    assert not context.keys() & {"period", "period_start", "period_end"}


@pytest.mark.parametrize(("token", "customer"), [("demo-alpha", "A007"), ("demo-beta", "B001")])
def test_request_owned_customer_id_renders_without_any_tools(token, customer):
    llm = ScriptedLLM(structured_turn("مشتری درخواستی {{customer}} است.", customer=customer_slot(customer)))
    response = agent(llm, token=token).run(ChatRequest(question=f"تیکت‌های مشتری {customer} را بررسی کن."))
    assert response.answer == f"مشتری درخواستی {customer} است."
    assert response.trace == []


@pytest.mark.parametrize(("token", "question", "claimed"), [
    ("demo-alpha", QUESTION, "A001"),
    ("demo-alpha", QUESTION, "A0070"),
    ("demo-alpha", QUESTION.replace("A007", "A999"), "A999"),
    ("demo-beta", QUESTION, "A007"),
    ("demo-alpha", QUESTION.replace("A007", "XA007"), "A007"),
], ids=["unrequested_owned_id", "altered_id", "unknown_id_in_question", "wrong_tenant", "substring"])
def test_request_identifier_cannot_ground_unowned_or_unrequested_customer(token, question, claimed):
    llm = ScriptedLLM(structured_turn("مشتری درخواستی {{customer}} است.", customer=customer_slot(claimed)))
    with pytest.raises(UngroundedAnswerError) as failure:
        agent(llm, token=token).run(ChatRequest(question=question))
    assert failure.value.reason_code == "unsupported_number"


def test_q4_does_not_infer_a_period_from_query_result(alpha):
    llm = ScriptedLLM(
        tool_turn(call("query_sql", {"sql": MONTHLY_SQL, "params": ["A007"]}, "q4-monthly")),
        structured_turn("شروع گزارش {{start}} است.", start=request_date("period_start", "2026-08-01")),
    )
    with pytest.raises(UngroundedAnswerError) as failure:
        agent(llm, tenant=alpha).run(ChatRequest(question=QUESTION))
    assert failure.value.reason_code == "unsupported_number"
    assert len(tool_result(llm.calls[1]["messages"], "q4-monthly")["rows"]) == 3


def test_successful_a007_lookup_grounds_zero_for_that_customer(alpha):
    llm = ScriptedLLM(
        ticket_lookup("A007"),
        structured_turn("تعداد تیکت‌های بازیابی‌شده {{count}} است.", count=numeric_slot(0, TICKET_REF, "ticket_count", "count")),
    )
    response = agent(llm, tenant=alpha).run(ChatRequest(question=QUESTION))
    assert response.answer == "تعداد تیکت‌های بازیابی‌شده 0 است."
    assert tool_result(llm.calls[1]["messages"], TICKET_REF) == {"tickets": []}


def test_zero_ticket_result_for_another_customer_cannot_ground_requested_customer(alpha):
    assert len(alpha.tickets(customer_id="A001")) > 0
    llm = ScriptedLLM(
        ticket_lookup("A007"),
        structured_turn("برای مشتری درخواستی {{count}} تیکت ثبت شده است.",
                        count=numeric_slot(0, TICKET_REF, "ticket_count", "count")),
    )
    with pytest.raises(UngroundedAnswerError) as failure:
        agent(llm, tenant=alpha).run(ChatRequest(question=QUESTION.replace("A007", "A001")))
    assert failure.value.reason_code == "unsupported_number"
    assert tool_result(llm.calls[1]["messages"], TICKET_REF) == {"tickets": []}


@pytest.mark.parametrize("foreign_scope", [
    EvidenceScope("foreign-tenant", "q4"), EvidenceScope("alpha", "foreign-conversation"),
], ids=["tenant", "conversation"])
def test_same_zero_count_from_foreign_evidence_is_rejected(alpha, foreign_scope):
    scope = EvidenceScope("alpha", "q4")
    context = GroundingContext(scope, "Alpha")
    record = RecordedToolResult(foreign_scope, TICKET_REF, "get_tickets", True, {"tickets": alpha.tickets(customer_id="A007")})
    raw = structured_turn("تعداد تیکت‌ها {{count}} است.",
                          count=numeric_slot(0, TICKET_REF, "ticket_count", "count"))["choices"][0]["message"]["content"]
    with pytest.raises(StructuredGroundingError) as failure:
        validate_and_render(raw, context=context, tool_results=[record])
    assert failure.value.reason_code == "unsupported_number"


def test_real_ticket_description_cannot_be_used_as_verified_cause(alpha):
    def cite_description(messages, tools):
        ticket = tool_result(messages, TICKET_REF)["tickets"][0]
        assert ticket["is_untrusted"] is True
        return structured_turn("علت قطعی افت خرید {{cause}} است.", cause={
            "type": "identifier", "value": ticket["description"], "evidence_ref": TICKET_REF,
            "evidence_field": "description", "row": 0,
        })

    llm = ScriptedLLM(ticket_lookup("A001"), cite_description)
    with pytest.raises(UngroundedAnswerError) as failure:
        agent(llm, tenant=alpha).run(ChatRequest(question=QUESTION.replace("A007", "A001")))
    assert failure.value.reason_code == "unsupported_number"
