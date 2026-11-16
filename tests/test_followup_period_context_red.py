"""Phase 4-C-C.5-H-A. Authorize report dates from tenant-owned successful turns."""

import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app import config
from app.agent import UngroundedAnswerError
from app.sandbox import DockerPythonSandbox, SandboxResult
from app.schemas import ChatRequest
from live_tests.test_agent_end_to_end_live import configured_app
from test_agent_loop import ScriptedLLM, call, numeric_slot, structured_turn, tool_result, tool_turn
from test_evidence_reference_contract_red import manifest
from test_gregorian_month_boundary_red import CAPTURED_QUESTION, LAST, NEXT, PERIOD, START, request_date
from test_python_png_diagnostic_contract_red import PYTHON_CODE, assert_decoded_chart
from test_python_sandbox import docker_ready
from test_state_recovery import agent, state_store, tenant
from test_structured_answer_grounding_red import PAID_SQL, september_total


FOLLOWUP = "عدد دقیق گزارش قبلی را دوباره بگو."
LIVE_FOLLOWUP = (
    "عدد دقیق گزارش قبلی را با رقم دوباره بگو و نمودار قبلی را نگه دار. "
    "برای slot عددی از همان شاهد execute_python و فیلد revenue استفاده کن."
)
OCTOBER_QUESTION = "عدد گزارش قبلی را برای اکتبر میلادی 2026 بگو."
PYTHON_REF = "september-python-report"
SEPTEMBER_CONTEXT = {"period": PERIOD, "period_start": START, "period_end": LAST}
OCTOBER_CONTEXT = {"period": "2026-10", "period_start": "2026-10-01", "period_end": "2026-10-31"}
REPORT_TEMPLATE = "  در دورهٔ {{period}}، از {{start}} تا {{end}}، مبلغ {{revenue}} تومان بود.  "


@pytest.fixture(autouse=True)
def isolated_http_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_ENV_FILE", tmp_path / "unused.env")
    monkeypatch.delenv("LLM_MAX_TOTAL_TOKENS_PER_REQUEST", raising=False)
    monkeypatch.delenv("LLM_MAX_CALLS_PER_REQUEST", raising=False)


def period_slot(value):
    return {"source": "user_request", "value": value}


def report_answer(total):
    return structured_turn(
        REPORT_TEMPLATE,
        period=period_slot(PERIOD),
        start=request_date("period_start", START),
        end=request_date("period_end", LAST),
        revenue=numeric_slot(total, PYTHON_REF, "revenue", "scalar"),
    )


def sandbox_result(**metrics):
    sandbox = Mock(spec=DockerPythonSandbox)
    sandbox.execute.return_value = SandboxResult(json.dumps({"metrics": metrics}), "", 0)
    return sandbox


def python_turn():
    return tool_turn(call("execute_python", {"code": PYTHON_CODE}, PYTHON_REF))


def run(store, owner, llm, question, *, conversation=None, sandbox=None):
    return agent(store, llm, owner, sandbox=sandbox).run(
        ChatRequest(question=question, conversation_id=conversation)
    )


@pytest.fixture
def previous_report(tmp_path, september_total):
    store, owner = state_store(tmp_path), tenant()
    sandbox = sandbox_result(revenue=str(september_total))
    llm = ScriptedLLM(python_turn(), report_answer(september_total))
    first = run(store, owner, llm, CAPTURED_QUESTION, sandbox=sandbox)
    assert manifest(llm.calls[-1]["messages"])["context"]["user_request"] == SEPTEMBER_CONTEXT
    return store, owner, first.conversation_id, sandbox


@pytest.mark.parametrize(("restart", "question"), [
    (False, FOLLOWUP),
    (True, FOLLOWUP),
    (True, LIVE_FOLLOWUP),
], ids=["same_app", "restarted_app", "captured_live_followup"])
def test_explicit_previous_report_recovers_authorized_period_and_original_python(
    tmp_path, docker_ready, september_total, restart, question,
):
    llm = ScriptedLLM(python_turn(), report_answer(september_total), report_answer(september_total))
    service = configured_app(tmp_path, llm)
    alpha = {"X-Demo-Token": "demo-alpha"}
    with TestClient(service) as client:
        first_response = client.post("/chat", headers=alpha, json={"question": CAPTURED_QUESTION})
        assert first_response.status_code == 200, first_response.json()
        first = first_response.json()
        artifact, = first["artifacts"]
        artifact_url = f"/artifacts/{artifact['artifact_id']}"
        image = client.get(artifact_url, headers=alpha)
        assert image.status_code == 200 and image.headers["content-type"] == "image/png"
        assert_decoded_chart(image.content, september_total)

    second_service = configured_app(tmp_path, llm) if restart else service
    with TestClient(second_service) as client:
        second_response = client.post("/chat", headers=alpha, json={
            "question": question, "conversation_id": first["conversation_id"],
        })
        same_image = client.get(artifact_url, headers=alpha)
        assert same_image.status_code == 200 and same_image.content == image.content

    assert len(llm.calls) == 3
    messages = llm.calls[-1]["messages"]
    assert messages[-1] == {"role": "user", "content": question}
    assert manifest(messages)["tools"] == [{
        "evidence_ref": PYTHON_REF,
        "tool_name": "execute_python",
        "fields": [{"evidence_field": "revenue", "type": "numeric", "unit": "scalar"}],
    }]
    original = tool_result(messages, PYTHON_REF)
    assert json.loads(original["stdout"]) == {"metrics": {"revenue": str(september_total)}}
    assert original["artifacts"] == [artifact]
    owner = second_service.state.secure_data.authenticate("demo-alpha")
    saved = second_service.state.conversation_store.get_conversation(owner, first["conversation_id"])
    evidence, = saved.evidence
    assert evidence.call_id == PYTHON_REF and evidence.result == original
    assert evidence.scope.tenant_key == owner.state_key and evidence.scope.conversation_id == first["conversation_id"]
    assert [len(turn.trace) for turn in saved.turns] == [1, 0]

    assert second_response.status_code == 200, (second_response.json(), saved.turns[-1].failure_code)
    second = second_response.json()
    assert manifest(messages)["context"]["user_request"] == SEPTEMBER_CONTEXT
    assert second["answer"] == f"  در دورهٔ {PERIOD}، از {START} تا {LAST}، مبلغ {september_total} تومان بود.  "
    assert second["conversation_id"] == first["conversation_id"]
    assert second["trace_id"] != first["trace_id"]
    assert second["artifacts"] == [artifact] and second["trace"] == []
    assert [turn.status for turn in saved.turns] == ["success", "success"]


@pytest.mark.parametrize("period", ["2026-10", PERIOD], ids=["october_accepted", "september_rejected"])
def test_explicit_new_period_takes_precedence(previous_report, period):
    store, owner, conversation, _ = previous_report
    llm = ScriptedLLM(structured_turn("دورهٔ درخواست {{period}} است.", period=period_slot(period)))
    if period == "2026-10":
        result = run(store, owner, llm, OCTOBER_QUESTION, conversation=conversation)
        assert result.answer == "دورهٔ درخواست 2026-10 است."
    else:
        with pytest.raises(UngroundedAnswerError) as failure:
            run(store, owner, llm, OCTOBER_QUESTION, conversation=conversation)
        assert failure.value.reason_code == "unsupported_number"
    assert manifest(llm.calls[0]["messages"])["context"]["user_request"] == OCTOBER_CONTEXT


def assert_period_denied(store, owner, conversation, question):
    llm = ScriptedLLM(structured_turn("دورهٔ گزارش {{period}} است.", period=period_slot(PERIOD)))
    with pytest.raises(UngroundedAnswerError) as failure:
        run(store, owner, llm, question, conversation=conversation)
    assert failure.value.reason_code == "unsupported_number"
    assert "user_request" not in manifest(llm.calls[0]["messages"])["context"]


@pytest.mark.parametrize("question", [
    "ساختار مشتریان را توضیح بده.",
    "عدد دقیق گزارش قبلی را بگو و سپتامبر 2026 را با اکتبر 2026 مقایسه کن.",
], ids=["unrelated", "ambiguous_current_period"])
def test_current_request_does_not_authorize_implicit_or_conflicting_period(previous_report, question):
    store, owner, conversation, _ = previous_report
    assert_period_denied(store, owner, conversation, question)


def test_reference_to_one_of_multiple_reports_is_ambiguous(previous_report):
    store, owner, conversation, _ = previous_report
    run(store, owner,
        ScriptedLLM(structured_turn("دورهٔ گزارش {{period}} است.", period=period_slot("2026-10"))),
        "گزارش اکتبر 2026 را بده.", conversation=conversation)
    assert_period_denied(store, owner, conversation, "عدد دقیق یکی از گزارش‌های قبلی را بگو.")


def test_failed_previous_turn_never_authorizes_a_period(tmp_path, september_total):
    store, owner = state_store(tmp_path), tenant()
    conversation = store.create_conversation(owner)
    unsupported = structured_turn(
        "مبلغ {{revenue}} تومان بود.", revenue=numeric_slot(september_total + 1, PYTHON_REF, "revenue", "scalar"),
    )
    sandbox = sandbox_result(revenue=str(september_total))
    with pytest.raises(UngroundedAnswerError):
        run(store, owner, ScriptedLLM(python_turn(), unsupported), CAPTURED_QUESTION,
            conversation=conversation, sandbox=sandbox)
    reopened = state_store(tmp_path)
    snapshot = reopened.get_conversation(owner, conversation)
    assert snapshot.turns[0].status == "failed" and snapshot.turns[0].answer is None
    assert snapshot.evidence == []
    assert_period_denied(reopened, owner, conversation, FOLLOWUP)


def test_persisted_assistant_prose_does_not_authorize_month(tmp_path):
    store, owner = state_store(tmp_path), tenant()
    llm = ScriptedLLM(
        tool_turn(call("execute_python", {"code": "print('fixture')"}, "year-source")),
        structured_turn("عنوان پیشنهادی من سپتامبر {{year}} است.",
                        year=numeric_slot(2026, "year-source", "year", "scalar")),
    )
    prior = run(store, owner, llm, "عنوان گزارش را پیشنهاد کن.", sandbox=sandbox_result(year="2026"))
    assert "سپتامبر 2026" in prior.answer
    assert_period_denied(state_store(tmp_path), owner, prior.conversation_id, FOLLOWUP)


def test_verified_sql_window_is_not_request_period_authorization(tmp_path, september_total):
    store, owner = state_store(tmp_path), tenant()
    ref = "september-sql-without-user-period"
    llm = ScriptedLLM(
        tool_turn(call("query_sql", {"sql": PAID_SQL}, ref)),
        structured_turn("مبلغ محاسبه‌شده {{revenue}} تومان بود.",
                        revenue=numeric_slot(september_total, ref, "total_revenue")),
    )
    prior = run(store, owner, llm, "یک گزارش درآمد آماده کن.")
    reopened = state_store(tmp_path)
    evidence, = reopened.get_conversation(owner, prior.conversation_id).evidence
    assert evidence.call_id == ref
    assert (evidence.sql_month.first, evidence.sql_month.last) == (START, LAST)
    assert_period_denied(reopened, owner, prior.conversation_id, FOLLOWUP)


def test_reuse_keeps_verified_python_reference_exact(previous_report, september_total):
    store, owner, conversation, sandbox = previous_report
    forged_ref = "unissued-python-report"
    llm = ScriptedLLM(structured_turn("عدد قبلی {{revenue}} بود.",
                                      revenue=numeric_slot(september_total, forged_ref, "revenue", "scalar")))
    with pytest.raises(UngroundedAnswerError) as failure:
        run(store, owner, llm, FOLLOWUP, conversation=conversation, sandbox=sandbox)
    assert failure.value.reason_code == "unsupported_number"
    assert {item["evidence_ref"] for item in manifest(llm.calls[0]["messages"])["tools"]} == {PYTHON_REF}
    assert sandbox.execute.call_count == 1


def test_followup_cannot_cite_exclusive_cutoff_as_inclusive_period_end(previous_report):
    store, owner, conversation, _ = previous_report
    llm = ScriptedLLM(structured_turn("پایان دوره {{end}} بود.", end=request_date("period_end", NEXT)))
    with pytest.raises(UngroundedAnswerError) as failure:
        run(store, owner, llm, FOLLOWUP, conversation=conversation)
    assert failure.value.reason_code == "unsupported_number"


def test_foreign_followup_cannot_load_alpha_context_or_evidence(previous_report):
    store, _, conversation, _ = previous_report
    llm = ScriptedLLM(structured_turn("گزارش قبلی در دسترس نیست."))
    service = configured_app(store.db_path.parent, llm)
    service.state.conversation_store = state_store(store.db_path.parent)
    with TestClient(service) as client:
        beta = {"X-Demo-Token": "demo-beta"}
        denied = client.post("/chat", headers=beta, json={"question": FOLLOWUP, "conversation_id": conversation})
        assert denied.status_code == 404
        assert llm.calls == []
        fresh = client.post("/chat", headers=beta, json={"question": FOLLOWUP})
    assert fresh.status_code == 200, fresh.json()
    beta_context = manifest(llm.calls[0]["messages"])
    assert beta_context["context"] == {"tenant_context": {"value": "Beta"}}
    assert beta_context["tools"] == []
