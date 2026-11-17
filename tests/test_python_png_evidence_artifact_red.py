"""Phase 4-C-C.5-Q-A. Exercise live-harness artifact selection with scripted replies."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from live_tests import test_agent_end_to_end_live as live_cases
from test_agent_loop import call, numeric_slot, structured_turn, tool_turn
from test_final_json_format_recovery_red import content_of
from test_length_completion_recovery_red import isolated_settings, mocked_wire
from test_python_png_accepted_response_red import (
    FIRST_TEMPLATE, FOREIGN, ORIGINAL_REF, OTHER_REF, OWNER, SECOND_TEMPLATE, final_answer, harness,
)
from test_python_png_diagnostic_contract_red import PYTHON_CODE, assert_decoded_chart
from test_python_sandbox import docker_ready
from test_structured_answer_grounding_red import september_total


WITHOUT_CHART = "python-revenue-without-chart"
WITHOUT_METRICS = "python-without-metrics"
UNISSUED = "python-no-such-issued-reference"
WRITE_CHART = "Path('/output/paid_revenue.png').write_bytes(image)"


def python_turn(ref, code=PYTHON_CODE):
    return tool_turn(call("execute_python", {"code": code}, ref))


def result_for(state, ref):
    record, = [record for record in state.evidence if record.call_id == ref]
    assert record.succeeded and record.tool_name == "execute_python"
    return record.result


@pytest.fixture
def observed_http(harness, monkeypatch):
    """Observe actual ASGI responses and optionally change only first-chat artifact order."""
    traffic = SimpleNamespace(chats=[], downloads=[], reverse_artifacts=False, omit_first_artifact=False)

    class ObservedClient(TestClient):
        def request(self, method, url, *args, **kwargs):
            response = super().request(method, url, *args, **kwargs)
            if url == "/chat":
                payload = response.json()
                if not traffic.chats and response.status_code == 200:
                    if traffic.reverse_artifacts:
                        payload["artifacts"].reverse()
                    if traffic.omit_first_artifact:
                        payload["artifacts"].pop(0)
                    response.json = lambda: payload
                traffic.chats.append((response.status_code, payload))
            if url.startswith("/artifacts/"):
                traffic.downloads.append((url, kwargs.get("headers"), response.status_code))
            return response

    monkeypatch.setattr(live_cases, "TestClient", ObservedClient)
    return traffic


@pytest.mark.parametrize(
    ("cited_ref", "reversed_metadata", "extra_executions"),
    [
        pytest.param(OTHER_REF, False, False, id="cites-second"),
        pytest.param(ORIGINAL_REF, False, False, id="cites-first"),
        pytest.param(OTHER_REF, True, False, id="cites-second-reversed"),
        pytest.param(ORIGINAL_REF, True, False, id="cites-first-reversed"),
        pytest.param(ORIGINAL_REF, False, True, id="interleaved-no-chart-and-no-metrics"),
    ],
)
@pytest.mark.parametrize("capture_enabled", [False, True], ids=["capture-off", "capture-on"])
def test_cited_execution_selects_its_own_png_and_reuses_it_after_restart(
    harness, observed_http, september_total, cited_ref, reversed_metadata, extra_executions, capture_enabled,
):
    execute, events, services, checked, path = harness
    observed_http.reverse_artifacts = reversed_metadata
    first = final_answer(september_total, FIRST_TEMPLATE, cited_ref)
    second = final_answer(september_total, SECOND_TEMPLATE, cited_ref)
    tools = [python_turn(ORIGINAL_REF), python_turn(OTHER_REF)]
    if extra_executions:
        tools = [
            python_turn(WITHOUT_CHART, PYTHON_CODE.replace(WRITE_CHART, "")),
            *tools,
            python_turn(WITHOUT_METRICS, "..."),
        ]
    sent, _ = execute([*tools, first, second], capture_enabled=capture_enabled)

    (first_status, first_payload), (second_status, second_payload) = observed_http.chats
    assert (first_status, second_status) == (200, 200)
    conversation_id = first_payload["conversation_id"]
    traces = [first_payload["trace_id"], second_payload["trace_id"]]
    assert second_payload["conversation_id"] == conversation_id and traces[0] != traces[1]
    assert checked == [(content_of(first), {cited_ref}), (content_of(second), {cited_ref})]
    assert len(services) == 2 and services[0] is not services[1]
    assert len(sent) == len(tools) + 2

    restarted = services[1]
    state = live_cases.snapshot(restarted, "demo-alpha", conversation_id)
    assert [(turn.trace_id, turn.status) for turn in state.turns] == [(trace, "success") for trace in traces]
    assert state.turns[1].trace == []
    assert len(state.turns[0].trace) == len(tools)
    original = result_for(state, ORIGINAL_REF)
    other = result_for(state, OTHER_REF)
    for result in (original, other):
        assert result["exit_code"] == 0 and result["stderr"] == ""
        assert json.loads(result["stdout"]) == {"metrics": {"revenue": str(september_total)}}
    if extra_executions:
        no_chart = result_for(state, WITHOUT_CHART)
        assert no_chart["artifacts"] == []
        assert json.loads(no_chart["stdout"]) == {"metrics": {"revenue": str(september_total)}}
        assert result_for(state, WITHOUT_METRICS)["stdout"] == ""
    original_png, = original["artifacts"]
    other_png, = other["artifacts"]
    artifact_order = [original_png, other_png]
    assert original_png["artifact_id"] != other_png["artifact_id"]
    assert first_payload["artifacts"] == (list(reversed(artifact_order)) if reversed_metadata else artifact_order)
    selected_png = original_png if cited_ref == ORIGINAL_REF else other_png
    assert selected_png in second_payload["artifacts"]
    path_url = f"/artifacts/{selected_png['artifact_id']}"
    assert observed_http.downloads[:2] == [(path_url, OWNER, 200), (path_url, FOREIGN, 404)]
    assert all(url == path_url for url, _, _ in observed_http.downloads)

    with TestClient(restarted) as client:
        owned = client.get(path_url, headers=OWNER)
        assert owned.status_code == 200 and owned.headers["content-type"] == "image/png"
        assert_decoded_chart(owned.content, september_total)
        assert client.get(path_url, headers=FOREIGN).status_code == 404
        denied = client.post("/chat", headers=FOREIGN, json={
            "question": "گزارش قبلی را دوباره بگو.", "conversation_id": conversation_id,
        })
        assert denied.status_code == 404
    assert len(sent) == len(tools) + 2
    assert [data["trace_id"] for event, data in events if event == "turn"] == traces
    if capture_enabled:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["capture_errors"] == []
        accepted = [record for record in data["grounding"] if record["rejection_reason"] is None]
        assert [record["trace_id"] for record in accepted] == traces
        assert [record["raw_final_content"] for record in accepted] == [content_of(first), content_of(second)]
    else:
        assert not path.exists()


def test_harness_rejects_a_different_verified_first_turn_id_on_followup(harness, observed_http, september_total):
    execute, _, services, checked, _ = harness
    first = final_answer(september_total, FIRST_TEMPLATE, ORIGINAL_REF)
    followup = final_answer(september_total, SECOND_TEMPLATE, OTHER_REF)

    with pytest.raises(AssertionError, match="The final revenue slot must cite the verified Python result"):
        execute([python_turn(ORIGINAL_REF), python_turn(OTHER_REF), first, followup])

    (first_status, initial), (second_status, _) = observed_http.chats
    assert (first_status, second_status) == (200, 200)
    assert checked == [(content_of(first), {ORIGINAL_REF}), (content_of(followup), {ORIGINAL_REF})]
    state = live_cases.snapshot(services[1], "demo-alpha", initial["conversation_id"])
    assert {item.call_id for item in state.evidence} == {ORIGINAL_REF, OTHER_REF}
    assert [len(turn.trace) for turn in state.turns] == [2, 0]


def test_harness_rejects_python_reexecution_even_when_followup_cites_original(harness, observed_http, september_total):
    execute, events, _, _, _ = harness
    first = final_answer(september_total, FIRST_TEMPLATE, ORIGINAL_REF)
    followup = final_answer(september_total, SECOND_TEMPLATE, ORIGINAL_REF)

    with pytest.raises(AssertionError):
        execute([python_turn(ORIGINAL_REF), first, python_turn(WITHOUT_METRICS, "..."), followup])

    assert [status for status, _ in observed_http.chats] == [200, 200]
    assert [data["tool_call_id"] for event, data in events if event == "tool"] == [ORIGINAL_REF, WITHOUT_METRICS]


@pytest.mark.parametrize("bad_result", ["no-chart", "wrong-revenue", "artifact-not-returned"])
def test_harness_does_not_pair_citation_with_an_unrelated_png(harness, observed_http, september_total, bad_result):
    execute, events, _, _, _ = harness
    observed_http.omit_first_artifact = bad_result == "artifact-not-returned"
    code = PYTHON_CODE
    first = final_answer(september_total, FIRST_TEMPLATE, ORIGINAL_REF)
    if bad_result == "no-chart":
        code = code.replace(WRITE_CHART, "")
    elif bad_result == "wrong-revenue":
        code = code.replace("str(total)", "str(total - 1), 'benchmark': str(total)")
        # Keep the independent expected total in the answer via another field so
        # it cannot mask reliance on the wrong revenue slot in the chart result.
        first = structured_turn(
            "درآمد {{revenue}} و عدد کنترل {{benchmark}} ثبت شد.",
            revenue=numeric_slot(september_total - 1, ORIGINAL_REF, "revenue", "scalar"),
            benchmark=numeric_slot(september_total, ORIGINAL_REF, "benchmark", "scalar"),
        )

    with pytest.raises(AssertionError):
        execute([python_turn(ORIGINAL_REF, code), python_turn(OTHER_REF), first])

    (http_status, payload), = observed_http.chats
    assert http_status == 200
    original, unrelated = [json.loads(data["result_text"]) for event, data in events if event == "tool"]
    assert original["exit_code"] == unrelated["exit_code"] == 0 and unrelated["artifacts"]
    assert unrelated["artifacts"][0] in payload["artifacts"]
    if bad_result == "no-chart":
        assert original["artifacts"] == []
    elif bad_result == "wrong-revenue":
        assert original["artifacts"]
        assert json.loads(original["stdout"])["metrics"]["revenue"] == str(september_total - 1)
    else:
        selected, = original["artifacts"]
        assert selected not in payload["artifacts"]
    assert not observed_http.downloads


@pytest.mark.parametrize("issue", ["missing", "failed", "duplicate", "foreign-tenant", "other-conversation"])
def test_harness_requires_unique_successful_owned_persisted_evidence(
    harness, observed_http, monkeypatch, september_total, issue,
):
    execute, _, _, _, _ = harness
    snapshot = live_cases.snapshot

    def altered_harness_read(service, token, conversation_id):
        state = snapshot(service, token, conversation_id)
        record, = state.evidence
        if issue == "missing":
            evidence = []
        elif issue == "failed":
            evidence = [replace(record, succeeded=False)]
        elif issue == "duplicate":
            evidence = [record, record]
        elif issue == "foreign-tenant":
            foreign = service.state.secure_data.authenticate("demo-beta")
            evidence = [replace(record, scope=replace(record.scope, tenant_key=foreign.state_key))]
        else:
            evidence = [replace(record, scope=replace(record.scope, conversation_id="unrelated-conversation"))]
        return replace(state, evidence=evidence)

    monkeypatch.setattr(live_cases, "snapshot", altered_harness_read)
    first = final_answer(september_total, FIRST_TEMPLATE, ORIGINAL_REF)
    followup = final_answer(september_total, SECOND_TEMPLATE, ORIGINAL_REF)

    with pytest.raises(AssertionError):
        execute([python_turn(ORIGINAL_REF), first, followup])

    assert [status for status, _ in observed_http.chats] == [200]
    assert observed_http.downloads == []


@pytest.mark.parametrize("ambiguity", ["two-citations", "two-pngs"])
def test_harness_rejects_ambiguous_revenue_or_png(harness, observed_http, september_total, ambiguity):
    execute, _, _, _, _ = harness
    if ambiguity == "two-citations":
        turns = [python_turn(ORIGINAL_REF), python_turn(OTHER_REF)]
        first = structured_turn(
            "درآمد {{revenue}} و درآمد {{repeat}} گزارش شد.",
            revenue=numeric_slot(september_total, ORIGINAL_REF, "revenue", "scalar"),
            repeat=numeric_slot(september_total, OTHER_REF, "revenue", "scalar"),
        )
    else:
        turns = [python_turn(ORIGINAL_REF, PYTHON_CODE + "\nPath('/output/also_valid.png').write_bytes(image)\n")]
        first = final_answer(september_total, FIRST_TEMPLATE, ORIGINAL_REF)
    followup = final_answer(september_total, SECOND_TEMPLATE, ORIGINAL_REF)

    with pytest.raises(AssertionError):
        execute([*turns, first, followup])

    (status, payload), = observed_http.chats
    assert status == 200 and len(payload["artifacts"]) == 2
    assert observed_http.downloads == []


def test_unissued_reference_still_fails_grounding(harness, observed_http, september_total):
    execute, events, _, checked, _ = harness
    unissued = final_answer(september_total, FIRST_TEMPLATE, UNISSUED)

    with pytest.raises(AssertionError, match="502"):
        execute([python_turn(ORIGINAL_REF), unissued, unissued])

    assert [status for status, _ in observed_http.chats] == [502]
    reasons = [data["rejection_reason"] for event, data in events if event == "grounding_end"]
    assert reasons and set(reasons) == {"unsupported_number"}
    assert checked == []
