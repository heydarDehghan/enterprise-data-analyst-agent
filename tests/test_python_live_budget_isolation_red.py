"""Phase 4-C-C.5-F-A. Exercise the live fixture through offline HTTP requests."""

import inspect
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import config
from app.agent import EnvChatCompletionsClient
from live_tests import test_agent_end_to_end_live as live_cases
from test_agent_loop import call, structured_turn, tool_turn
from test_beta_diagnostic_contract_red import BETA_NODE, fixture_scope, load_capture
from test_llm_io_diagnostics_red import (
    ALPHA_NODE, CAPTURE_FLAG, CAPTURE_PATH, MODEL, fake_provider, isolated_provider_environment,
)
from test_python_png_diagnostic_contract_red import PYTHON_NODE


NODE_BUDGETS = [
    pytest.param(PYTHON_NODE, "demo-alpha", 40_000, id="python_png"),
    pytest.param(ALPHA_NODE, "demo-alpha", 24_000, id="alpha_sql_tickets"),
    pytest.param(BETA_NODE, "demo-beta", 24_000, id="beta_sql_tickets"),
]
QUESTION = "گزارش پرداخت‌ها را بررسی کن."


@pytest.fixture
def sandboxed_settings(tmp_path, monkeypatch, isolated_provider_environment):
    # Use the actual loader with an isolated dotenv file and the fake HTTP credential.
    env = tmp_path / "budget-fixture.env"
    env.write_text(
        "LLM_MAX_TOTAL_TOKENS_PER_REQUEST=40000\n"
        "LLM_MAX_CALLS_PER_REQUEST=12\n"
        "LLM_MAX_COMPLETION_TOKENS=8192\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "_ENV_FILE", env)
    for name in ("LLM_MAX_TOTAL_TOKENS_PER_REQUEST", "LLM_MAX_CALLS_PER_REQUEST", "LLM_MAX_COMPLETION_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    assert config.load_llm_budget() == (12, 40_000)
    assert config.load_llm_config().max_completion_tokens == 8192


@contextmanager
def selected_live_configuration(monkeypatch, node, path):
    monkeypatch.setenv(CAPTURE_FLAG, "1")
    monkeypatch.setenv(CAPTURE_PATH, str(path))
    item = SimpleNamespace(nodeid=node)
    request = SimpleNamespace(node=item, session=SimpleNamespace(items=[item]))
    fixture = live_cases.live_configuration.__wrapped__
    # Reflect the fixture signature so pytest's request argument can be added in GREEN.
    arguments = {"monkeypatch": monkeypatch, "request": request}
    with fixture_scope(monkeypatch, [node]):
        fixture(**{name: arguments[name] for name in inspect.signature(fixture).parameters})
        yield


def provider_reply(completion, total):
    completion["usage"] = {
        "prompt_tokens": total - 128, "completion_tokens": 128, "total_tokens": total,
        "completion_tokens_details": {"reasoning_tokens": 64},
    }
    return 200, json.dumps(completion, ensure_ascii=False).encode("utf-8")


def assert_wire_configuration(capture, sent):
    assert len(capture["provider_calls"]) == len(sent)
    for entry, outbound in zip(capture["provider_calls"], sent, strict=True):
        payload = json.loads(outbound)
        assert payload["model"] == MODEL
        assert payload["max_completion_tokens"] == 8192
        # The current adapter leaves reasoning at the model/provider defaults.
        assert not {"reasoning", "reasoning_effort", "include_reasoning"} & payload.keys()
        assert entry["request"]["headers"]["authorization"] == "[REDACTED]"


@pytest.mark.parametrize(("node", "tenant", "expected_budget"), NODE_BUDGETS)
def test_agent_uses_node_scoped_budget_after_live_fixture(tmp_path, monkeypatch, sandboxed_settings,
                                                      node, tenant, expected_budget):
    path = tmp_path / "policy.json"
    sent = fake_provider(monkeypatch, [provider_reply(structured_turn("گزارش آماده است."), 1_200)])
    with selected_live_configuration(monkeypatch, node, path):
        resolved = config.load_llm_budget()
        with TestClient(live_cases.configured_app(tmp_path, EnvChatCompletionsClient())) as client:
            response = client.post("/chat", headers={"X-Demo-Token": tenant}, json={"question": QUESTION})
    assert response.status_code == 200
    capture = load_capture(path, node)
    assert_wire_configuration(capture, sent)
    assert capture["runtime_limits"] == {"max_calls": 8, "max_total_tokens": expected_budget}
    assert resolved == (8, expected_budget)


@pytest.mark.parametrize("cost_per_chat", [22_000, 28_636], ids=["existing_limit", "requires_python_limit"])
def test_python_chats_have_independent_budgets_across_restart(tmp_path, monkeypatch,
                                                            sandboxed_settings, cost_per_chat):
    path = tmp_path / "two-chats.json"
    replies = [
        provider_reply(structured_turn("گزارش آماده است."), cost_per_chat),
        provider_reply(structured_turn("گزارش قبلی در دسترس است."), cost_per_chat),
    ]
    sent = fake_provider(monkeypatch, replies)
    alpha = {"X-Demo-Token": "demo-alpha"}
    with selected_live_configuration(monkeypatch, PYTHON_NODE, path):
        with TestClient(live_cases.configured_app(tmp_path, EnvChatCompletionsClient())) as client:
            first = client.post("/chat", headers=alpha, json={"question": QUESTION})
        assert first.status_code == 200, first.json()
        conversation = first.json()["conversation_id"]
        followup = {"question": "گزارش قبلی را دوباره نشان بده.", "conversation_id": conversation}
        with TestClient(live_cases.configured_app(tmp_path, EnvChatCompletionsClient())) as restarted:
            forbidden = restarted.post("/chat", headers={"X-Demo-Token": "demo-beta"}, json=followup)
            assert forbidden.status_code == 404
            assert len(sent) == 1
            second = restarted.post("/chat", headers=alpha, json=followup)
        assert second.status_code == 200, second.json()
    assert second.json()["conversation_id"] == conversation
    assert first.json()["trace_id"] != second.json()["trace_id"]
    assert first.json()["answer"] == "گزارش آماده است."
    assert second.json()["answer"] == "گزارش قبلی در دسترس است."
    capture = load_capture(path, PYTHON_NODE)
    assert_wire_configuration(capture, sent)
    calls = capture["provider_calls"]
    assert len(calls) == 2
    assert [item["index"] for item in calls] == [1, 1]
    assert [item["trace_id"] for item in calls] == [first.json()["trace_id"], second.json()["trace_id"]]
    assert all(item["conversation_id"] == conversation for item in calls)


@pytest.mark.parametrize(("node", "tenant", "expected_budget"), NODE_BUDGETS)
def test_agent_rejects_cumulative_usage_above_node_budget(tmp_path, monkeypatch, sandboxed_settings,
                                                        node, tenant, expected_budget):
    path = tmp_path / "budget-exceeded.json"
    half = expected_budget // 2
    replies = [
        provider_reply(tool_turn(call("get_schema", {}, "before-limit")), half),
        provider_reply(tool_turn(call("get_tickets", {}, "after-limit")), half + 1),
    ]
    sent = fake_provider(monkeypatch, replies)
    with selected_live_configuration(monkeypatch, node, path):
        with TestClient(live_cases.configured_app(tmp_path, EnvChatCompletionsClient())) as client:
            response = client.post("/chat", headers={"X-Demo-Token": tenant}, json={"question": QUESTION})
    assert response.status_code == 429
    assert response.json() == {"detail": "Request budget exceeded"}
    capture = load_capture(path, node)
    assert_wire_configuration(capture, sent)
    assert len(sent) == 2
    assert [item["tool_call_id"] for item in capture["tool_executions"]] == ["before-limit"]
    assert capture["grounding"] == []
