"""Phase 4-C-C.5-J-A. One answer-only regeneration for an unknown evidence reference."""

import json
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app import config
from app.agent import EnvChatCompletionsClient
from app.main import create_app
from app.observations import observer
from live_tests.test_agent_end_to_end_live import paid_september_total
from test_agent_loop import DATA, ScriptedLLM, call, numeric_slot, structured_turn, tool_turn
from test_evidence_reference_contract_red import manifest
from test_gregorian_month_boundary_red import request_date
from test_python_png_diagnostic_contract_red import PYTHON_CODE
from test_python_sandbox import docker_ready
from test_structured_answer_grounding_red import PAID_SQL, QUESTION, september_total


SQL_REF = "call_verified_september_sql"
PYTHON_REF = "call_d24e7013bf5d42358a2cc664"
UNKNOWN_REF = "call_7d1f4c2a9e8b3f6d0a5c7e91"
OTHER_REF = "call_other_conversation"
REJECTED_MARKER = "REJECTED_DRAFT_ONLY"
REJECTED_TEMPLATE = "پیش‌نویس {{revenue}} تومان بود. " + REJECTED_MARKER
REPAIRED_TEMPLATE = "  مجموع پرداخت‌های ثبت‌شده {{revenue}} تومان است.  "


@pytest.fixture(autouse=True)
def offline_configuration(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_ENV_FILE", tmp_path / "unused.env")
    monkeypatch.setenv("LLM_MAX_CALLS_PER_REQUEST", "8")
    monkeypatch.setenv("LLM_MAX_TOTAL_TOKENS_PER_REQUEST", "40000")

    def forbid_provider(*args, **kwargs):
        pytest.fail("Evidence reference repair tests must use a scripted provider")

    monkeypatch.setattr(EnvChatCompletionsClient, "create", forbid_provider)


def service_at(path, llm):
    return create_app(
        data_dir=DATA, token_map_path=DATA.parent / "demo_tokens.json.example",
        schema_path=DATA / "schema.md", state_db_path=path / "turns.db",
        artifact_root=path / "artifacts", docker_image="python:3.13-slim", llm_client=llm,
    )


def answer(total, ref, field="total_revenue", unit="toman", template=REPAIRED_TEMPLATE, **extra):
    return structured_turn(template, revenue=numeric_slot(total, ref, field, unit), **extra)


def sql_turn(ref=SQL_REF, sql=PAID_SQL):
    return tool_turn(call("query_sql", {"sql": sql}, ref))


def usage(completion, total):
    return {**completion, "usage": {"total_tokens": total}}


def script(*completions):
    return ScriptedLLM(*(usage(item, 100) if isinstance(item, dict) and "usage" not in item else item
                         for item in completions))


def chat(client, tenant="demo-alpha", conversation_id=None):
    return client.post("/chat", headers={"X-Demo-Token": tenant}, json={
        "question": QUESTION, "conversation_id": conversation_id,
    })


def snapshot(service, response=None, *, tenant="demo-alpha", conversation_id=None):
    store = service.state.conversation_store
    subject = service.state.secure_data.authenticate(tenant)
    if response is not None:
        conversation_id = response.json()["conversation_id"]
    return store.get_conversation(subject, conversation_id)


def assert_rejected(response):
    assert response.status_code == 502
    assert response.json() == {"detail": "Analysis answer rejected"}


def assert_repair_request(llm, ref, field, unit, tool_name):
    draft, repair = llm.calls[1:]
    assert [entry["evidence_ref"] for entry in manifest(draft["messages"])["tools"]] == [ref]
    assert manifest(repair["messages"])["tools"] == [{
        "evidence_ref": ref, "tool_name": tool_name,
        "fields": [{"evidence_field": field, "type": "numeric", "unit": unit}],
    }]
    assert repair["tools"] == []
    assert repair["messages"][0]["content"] != draft["messages"][0]["content"]
    assert repair["messages"][1:] == draft["messages"][1:]
    assert REJECTED_MARKER not in json.dumps(repair["messages"])
    assert UNKNOWN_REF not in json.dumps(repair["messages"])


@pytest.mark.parametrize(("tool_name", "ref", "field", "unit"), [
    pytest.param("query_sql", SQL_REF, "total_revenue", "toman", id="sql"),
    pytest.param("execute_python", PYTHON_REF, "revenue", "scalar", id="docker_python"),
])
def test_one_regeneration_uses_exact_evidence_and_persists_only_new_answer(
    tmp_path, monkeypatch, request, september_total, tool_name, ref, field, unit, caplog,
):
    if tool_name == "execute_python":
        request.getfixturevalue("docker_ready")
        first_completion = tool_turn(call(tool_name, {"code": PYTHON_CODE}, ref))
    else:
        first_completion = sql_turn(ref)
    rejected = answer(september_total, UNKNOWN_REF, field, unit, REJECTED_TEMPLATE)
    new = answer(september_total, ref, field, unit)
    llm = script(first_completion, rejected, new)
    monkeypatch.setenv("LLM_MAX_CALLS_PER_REQUEST", "3")
    service = service_at(tmp_path, llm)
    observe = Mock()
    token = observer.set(observe)
    try:
        with TestClient(service) as client:
            response = chat(client)
    finally:
        observer.reset(token)

    assert manifest(llm.calls[1]["messages"])["tools"][0]["evidence_ref"] == ref
    assert response.status_code == 200, response.text
    assert len(llm.calls) == 3
    assert_repair_request(llm, ref, field, unit, tool_name)
    expected_answer = REPAIRED_TEMPLATE.replace("{{revenue}}", str(september_total))
    body = response.json()
    assert body["answer"] == expected_answer
    assert [(step["tool_name"], step["status"]) for step in body["trace"]] == [(tool_name, "success")]
    events = [(event.args[0], event.kwargs) for event in observe.call_args_list]
    assert [values["tool_call_id"] for event, values in events if event == "tool"] == [ref]
    assert [values["raw_final_content"] for event, values in events if event == "grounding_start"] == [
        rejected["choices"][0]["message"]["content"], new["choices"][0]["message"]["content"],
    ]
    assert [values.get("rejection_reason") for event, values in events if event == "grounding_end"] == [
        "unsupported_number", None,
    ]
    reopened = service_at(tmp_path, ScriptedLLM())
    stored = snapshot(reopened, response)
    saved, = stored.turns
    assert (saved.status, saved.failure_code, saved.answer, saved.trace_id) == (
        "success", None, expected_answer, body["trace_id"],
    )
    evidence, = stored.evidence
    assert (evidence.call_id, evidence.tool_name, evidence.succeeded) == (ref, tool_name, True)
    assert evidence.scope.conversation_id == body["conversation_id"]
    transcript = json.dumps(stored.messages, ensure_ascii=False)
    assert UNKNOWN_REF not in transcript and REJECTED_MARKER not in transcript + caplog.text


@pytest.mark.parametrize("second_outcome", [
    "unknown", "wrong_value", "wrong_unit", "wrong_date", "tool_call", "provider_error",
])
def test_unsuccessful_regeneration_keeps_generic_failure_and_never_runs_another_tool(
    tmp_path, september_total, second_outcome, caplog,
):
    replacements = {
        "unknown": answer(september_total, UNKNOWN_REF),
        "wrong_value": answer(september_total + 1, SQL_REF),
        "wrong_unit": answer(september_total, SQL_REF, unit="percent"),
        "wrong_date": answer(september_total, SQL_REF, template="{{end}}، درآمد {{revenue}} بود.",
                             end=request_date("period_end", "2026-10-01")),
        "tool_call": sql_turn("call_forbidden_reexecution"),
        "provider_error": RuntimeError("provider transport failed"),
    }
    rejected = answer(september_total, UNKNOWN_REF, template=REJECTED_TEMPLATE)
    # A third final draft would succeed if the application granted a second repair.
    llm = script(sql_turn(), rejected, replacements[second_outcome], answer(september_total, SQL_REF))
    service = service_at(tmp_path, llm)
    subject = service.state.secure_data.authenticate("demo-alpha")
    store = service.state.conversation_store
    conversation_id = store.create_conversation(subject)
    with TestClient(service) as client:
        response = chat(client, conversation_id=conversation_id)

    assert_rejected(response)
    assert len(llm.calls) == 3
    assert_repair_request(llm, SQL_REF, "total_revenue", "toman", "query_sql")
    stored = snapshot(service, conversation_id=conversation_id)
    saved, = stored.turns
    assert (saved.status, saved.failure_code, saved.answer) == ("failed", "unsupported_number", None)
    assert [(step.tool_name, step.status) for step in saved.trace] == [("query_sql", "success")]
    assert stored.evidence == []
    assert REJECTED_MARKER not in json.dumps(stored.messages) + response.text + caplog.text


@pytest.mark.parametrize("invalid_slot", ["value", "unit", "context_date", "wrong_sql_month", "no_local_field"])
def test_other_grounding_failures_do_not_trigger_regeneration(tmp_path, september_total, invalid_slot):
    first = sql_turn()
    rejected = {
        "value": answer(september_total + 1, SQL_REF),
        "unit": answer(september_total, SQL_REF, unit="scalar"),
        "context_date": answer(september_total, SQL_REF, template="{{end}}، درآمد {{revenue}} بود.",
                               end=request_date("period_end", "2026-10-01")),
        "wrong_sql_month": answer(september_total, SQL_REF),
        "no_local_field": answer(september_total, UNKNOWN_REF, field="not_returned"),
    }[invalid_slot]
    if invalid_slot == "wrong_sql_month":
        first = sql_turn(sql=PAID_SQL.replace("2026-09-01", "2026-08-23").replace("2026-10-01", "2026-09-23"))
    llm = script(first, rejected, answer(september_total, SQL_REF))
    with TestClient(service_at(tmp_path, llm)) as client:
        response = chat(client)

    assert_rejected(response)
    assert len(llm.calls) == 2


def test_final_rejection_does_not_attempt_repair_without_a_call_remaining(tmp_path, monkeypatch, september_total):
    monkeypatch.setenv("LLM_MAX_CALLS_PER_REQUEST", "2")
    llm = script(sql_turn(), answer(september_total, UNKNOWN_REF), answer(september_total, SQL_REF))
    with TestClient(service_at(tmp_path, llm)) as client:
        response = chat(client)

    assert_rejected(response)
    assert len(llm.calls) == 2


def test_repair_response_usage_counts_toward_same_chat_token_budget(tmp_path, september_total):
    llm = script(
        usage(sql_turn(), 14_000),
        usage(answer(september_total, UNKNOWN_REF), 14_000),
        usage(answer(september_total, SQL_REF), 12_001),
    )
    with TestClient(service_at(tmp_path, llm)) as client:
        response = chat(client)

    assert response.status_code == 429
    assert response.json() == {"detail": "Request budget exceeded"}
    assert len(llm.calls) == 3
    assert_repair_request(llm, SQL_REF, "total_revenue", "toman", "query_sql")


@pytest.mark.parametrize("tenant", ["demo-alpha", "demo-beta"], ids=["another_conversation", "another_tenant"])
def test_repair_cannot_cite_verified_evidence_from_another_conversation(tmp_path, september_total, tenant):
    seed = script(sql_turn(OTHER_REF), answer(september_total, OTHER_REF))
    original = service_at(tmp_path, seed)
    with TestClient(original) as client:
        prior = chat(client)
    assert prior.status_code == 200
    assert [item.call_id for item in snapshot(original, prior).evidence] == [OTHER_REF]

    local_total = september_total if tenant == "demo-alpha" else paid_september_total("beta.db")
    llm = script(sql_turn(), answer(local_total, UNKNOWN_REF),
                 answer(september_total, OTHER_REF), answer(local_total, SQL_REF))
    with TestClient(service_at(tmp_path, llm)) as client:
        if tenant == "demo-beta":
            foreign = chat(client, tenant, prior.json()["conversation_id"])
            assert foreign.status_code == 404 and llm.calls == []
        response = chat(client, tenant)

    assert_rejected(response)
    assert [entry["evidence_ref"] for entry in manifest(llm.calls[1]["messages"])["tools"]] == [SQL_REF]
    assert len(llm.calls) == 3
    assert_repair_request(llm, SQL_REF, "total_revenue", "toman", "query_sql")
    assert OTHER_REF not in json.dumps(llm.calls[2]["messages"])
