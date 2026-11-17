"""Phase 4-C-C.5-O-A. Run the Python/PNG harness against an offline provider."""

import json

import pytest
from fastapi.testclient import TestClient

from app import agent
from live_tests import test_agent_end_to_end_live as live_cases
from live_tests.llm_diagnostics import capture_run
from test_agent_loop import call, final_turn, numeric_slot, structured_turn, tool_turn
from test_evidence_reference_repair_red import UNKNOWN_REF, usage
from test_final_json_format_recovery_red import PROSE, content_of
from test_length_completion_recovery_red import encoded, isolated_settings, mocked_wire
from test_llm_io_diagnostics_red import CAPTURE_FLAG, CAPTURE_PATH, DEMO_SECRET, body_bytes, fake_provider
from test_python_png_diagnostic_contract_red import PYTHON_CODE, PYTHON_NODE, assert_decoded_chart
from test_python_sandbox import docker_ready
from test_structured_answer_grounding_red import september_total


ORIGINAL_REF = "python-revenue-original"
OTHER_REF = "python-revenue-other"
FIRST_TEMPLATE = "درآمد پرداخت‌شده {{revenue}} تومان است و نمودار آماده شد."
SECOND_TEMPLATE = "درآمد گزارش قبلی {{revenue}} تومان بود."
OWNER = {"X-Demo-Token": "demo-alpha"}
FOREIGN = {"X-Demo-Token": "demo-beta"}


def python_turn(ref=ORIGINAL_REF):
    return tool_turn(call("execute_python", {"code": PYTHON_CODE}, ref))


def final_answer(total, template, ref=ORIGINAL_REF):
    return structured_turn(template, revenue=numeric_slot(total, ref, "revenue", "scalar"))


def format_draft(accepted):
    return final_turn(PROSE + "\n" + content_of(accepted))


@pytest.fixture
def harness(tmp_path, monkeypatch, docker_ready, mocked_wire):
    """Exercise the unchanged live-test body with an HTTPX mock transport."""
    events = []
    services = []
    checked = []
    path = tmp_path / "python-png.json"
    observe_actual = agent.observe
    configure_actual = live_cases.configured_app
    assert_slot_actual = live_cases.assert_python_revenue_slot

    def configured_app(*args, **kwargs):
        service = configure_actual(*args, **kwargs)
        services.append(service)
        return service

    def check_slot(raw, evidence_refs):
        checked.append((raw, set(evidence_refs)))
        assert_slot_actual(raw, evidence_refs)

    monkeypatch.setattr(live_cases, "configured_app", configured_app)
    monkeypatch.setattr(live_cases, "assert_python_revenue_slot", check_slot)
    monkeypatch.setenv(CAPTURE_PATH, str(path))

    def execute(completions, *, capture_enabled=False, drop_accepted_event=False):
        def observe(event, **values):
            if drop_accepted_event and event == "grounding_end" and values["rejection_reason"] is None:
                return
            events.append((event, values))
            observe_actual(event, **values)

        monkeypatch.setattr(agent, "observe", observe)
        if capture_enabled:
            monkeypatch.setenv(CAPTURE_FLAG, "1")
        replies = [(200, encoded(usage(completion, 118))) for completion in completions]
        sent = fake_provider(monkeypatch, replies)
        with capture_run(node_id=PYTHON_NODE, monkeypatch=monkeypatch):
            live_cases.test_live_python_png_and_conversation_reuse(tmp_path)
        return sent, replies

    return execute, events, services, checked, path


def assert_accepted_observations(events, trace_ids, drafts_by_turn, expected):
    starts = [index for index, (event, _) in enumerate(events) if event == "turn"]
    assert [events[index][1]["trace_id"] for index in starts] == trace_ids
    for begin, end, drafts, raw in zip(starts, [*starts[1:], len(events)], drafts_by_turn, expected, strict=True):
        span = events[begin:end]
        raw_completions = [data["raw_final_content"] for event, data in span if event == "grounding_start"]
        outcomes = [data for event, data in span if event == "grounding_end"]
        assert raw_completions == [*drafts, raw]
        assert len(outcomes) == len(raw_completions)
        assert all(item["rejection_reason"] is not None and item["rendered_answer"] is None for item in outcomes[:-1])
        assert outcomes[-1]["rejection_reason"] is None and outcomes[-1]["rendered_answer"]


def assert_private_capture(path, sent, replies, trace_ids, expected):
    data = json.loads(path.read_text(encoding="utf-8"))
    calls = data["provider_calls"]
    assert data["test_node_id"] == PYTHON_NODE and data["capture_errors"] == []
    assert data["runtime_limits"] == {"max_calls": 8, "max_total_tokens": 40_000}
    assert len(calls) == len(sent) == len(replies)
    for record, outgoing, (_, incoming) in zip(calls, sent, replies, strict=True):
        assert body_bytes(record["request"]) == outgoing
        assert body_bytes(record["response"]) == incoming
        assert record["request"]["headers"]["authorization"] == "[REDACTED]"
    by_id = {record["call_id"]: record for record in calls}
    accepted = [entry for entry in data["grounding"] if entry["rejection_reason"] is None]
    assert [entry["trace_id"] for entry in accepted] == trace_ids
    for entry, raw in zip(accepted, expected, strict=True):
        call_record = by_id[entry["provider_call_id"]]
        assert call_record["trace_id"] == entry["trace_id"]
        assert entry["raw_final_content"] == raw
        assert content_of(json.loads(body_bytes(call_record["response"]))) == raw
    execution, = data["tool_executions"]
    assert execution["tool_call_id"] == ORIGINAL_REF and execution["status"] == "success"
    assert DEMO_SECRET not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("capture_enabled", [False, True], ids=["capture-off", "capture-on"])
@pytest.mark.parametrize("scenario", ["first_format", "followup_format", "first_evidence", "no_repair"])
def test_harness_checks_only_accepted_final_for_each_chat(harness, september_total, capture_enabled, scenario):
    execute, events, services, checked, path = harness
    first = final_answer(september_total, FIRST_TEMPLATE)
    second = final_answer(september_total, SECOND_TEMPLATE)
    first_drafts = []
    second_drafts = []
    if scenario == "first_format":
        first_drafts.append(format_draft(first))
    if scenario == "followup_format":
        second_drafts.append(format_draft(second))
    if scenario == "first_evidence":
        first_drafts.append(final_answer(september_total, FIRST_TEMPLATE, UNKNOWN_REF))
    completions = [python_turn(), *first_drafts, first, *second_drafts, second]

    sent, replies = execute(completions, capture_enabled=capture_enabled)

    expected = [content_of(first), content_of(second)]
    assert checked == [(raw, {ORIGINAL_REF}) for raw in expected]
    assert len(services) == 2 and services[0] is not services[1]
    trace_turns = [data for event, data in events if event == "turn"]
    first_turn, second_turn = trace_turns
    conversation_id = first_turn["conversation_id"]
    assert second_turn["conversation_id"] == conversation_id
    trace_ids = [first_turn["trace_id"], second_turn["trace_id"]]
    assert trace_ids[0] != trace_ids[1]
    assert_accepted_observations(events, trace_ids, [
        [content_of(draft) for draft in first_drafts],
        [content_of(draft) for draft in second_drafts],
    ], expected)

    restarted = services[1]
    state = live_cases.snapshot(restarted, "demo-alpha", conversation_id)
    assert [(turn.trace_id, turn.status, turn.answer) for turn in state.turns] == [
        (trace_id, "success", template.replace("{{revenue}}", str(september_total)))
        for trace_id, template in zip(trace_ids, (FIRST_TEMPLATE, SECOND_TEMPLATE), strict=True)
    ]
    assert [(step.tool_name, step.status) for step in state.turns[0].trace] == [("execute_python", "success")]
    assert state.turns[1].trace == []
    record, = state.evidence
    assert record.call_id == ORIGINAL_REF and record.succeeded
    assert record.scope.conversation_id == conversation_id
    assert json.loads(record.result["stdout"]) == {"metrics": {"revenue": str(september_total)}}
    artifact, = state.artifacts
    assert record.result["artifacts"] == [artifact.model_dump()]
    with TestClient(restarted) as client:
        path_url = f"/artifacts/{artifact.artifact_id}"
        image = client.get(path_url, headers=OWNER)
        assert image.status_code == 200 and image.headers["content-type"] == "image/png"
        assert_decoded_chart(image.content, september_total)
        assert client.get(path_url, headers=FOREIGN).status_code == 404
        denied = client.post("/chat", headers=FOREIGN, json={
            "question": "گزارش قبلی را دوباره بگو.", "conversation_id": conversation_id,
        })
        assert denied.status_code == 404
    assert len(sent) == len(completions)
    if capture_enabled:
        assert_private_capture(path, sent, replies, trace_ids, expected)
    else:
        assert not path.exists()


def test_harness_rejects_an_accepted_followup_that_uses_a_different_python_id(harness, september_total):
    execute, events, _, checked, _ = harness
    first = final_answer(september_total, FIRST_TEMPLATE)
    rejected = format_draft(final_answer(september_total, SECOND_TEMPLATE))
    # The second execution makes OTHER_REF a real, scoped result. Grounding can
    # accept it, but the harness must still require the original Python ID.
    accepted_with_other_id = final_answer(september_total, SECOND_TEMPLATE, OTHER_REF)
    with pytest.raises(AssertionError, match="The final revenue slot must cite the verified Python result"):
        execute([python_turn(), first, python_turn(OTHER_REF), rejected, accepted_with_other_id])

    assert checked == [(content_of(first), {ORIGINAL_REF}), (content_of(accepted_with_other_id), {ORIGINAL_REF})]
    trace_ids = [data["trace_id"] for event, data in events if event == "turn"]
    assert_accepted_observations(events, trace_ids, [[], [content_of(rejected)]],
                                 [content_of(first), content_of(accepted_with_other_id)])


def test_harness_requires_an_accepted_event_for_the_successful_chat(harness, september_total):
    execute, events, services, checked, _ = harness
    first = final_answer(september_total, FIRST_TEMPLATE)
    second = final_answer(september_total, SECOND_TEMPLATE)
    with pytest.raises(AssertionError, match=r"(?i)accepted.*grounding"):
        execute([python_turn(), first, second], drop_accepted_event=True)

    assert checked == []
    turn, = [data for event, data in events if event == "turn"]
    persisted, = live_cases.snapshot(services[0], "demo-alpha", turn["conversation_id"]).turns
    assert persisted.status == "success" and persisted.trace_id == turn["trace_id"]
    assert not any(event == "grounding_end" for event, _ in events)


def test_existing_python_slot_assertion_requires_the_exact_issued_reference(september_total):
    live_cases.assert_python_revenue_slot(content_of(final_answer(september_total, SECOND_TEMPLATE)), {ORIGINAL_REF})
    with pytest.raises(AssertionError, match="The final revenue slot must cite the verified Python result"):
        live_cases.assert_python_revenue_slot(
            content_of(final_answer(september_total, SECOND_TEMPLATE, OTHER_REF)), {ORIGINAL_REF},
        )
