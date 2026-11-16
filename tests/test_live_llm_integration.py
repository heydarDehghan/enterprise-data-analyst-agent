"""Phase 4-C-A RED contract for the authenticated chat service and provider.

Provider calls in this module use scripted completions or httpx.MockTransport.
Only the Docker case executes outside the API process, using the real image.
"""

import base64
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.secure_data import SecureDataAccess
from app.state import SQLiteConversationStore
from test_agent_loop import ScriptedLLM, call, final_turn, numeric_slot, structured_turn, tool_result, tool_turn
from test_python_sandbox import IMAGE, PNG, docker_ready


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TOKEN_FILE = ROOT / "demo_tokens.json.example"
TOKEN_MAP = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
SCHEMA = (DATA / "schema.md").read_text(encoding="utf-8")
SQL = ("SELECT SUM(amount_toman) AS revenue FROM paid_orders "
       "WHERE substr(order_date, 1, 7) = ?")


def runtime_client(tmp_path, monkeypatch, llm, *, sandbox=None):
    """Inject deterministic collaborators into the existing FastAPI service."""
    from app.main import app

    monkeypatch.setattr(app.state, "secure_data", SecureDataAccess(
        data_dir=DATA, token_map=TOKEN_MAP), raising=False)
    monkeypatch.setattr(app.state, "conversation_store", SQLiteConversationStore(
        db_path=tmp_path / "conversations.db", artifact_root=tmp_path / "artifacts"
    ), raising=False)
    monkeypatch.setattr(app.state, "llm_client", llm, raising=False)
    monkeypatch.setattr(app.state, "schema_text", SCHEMA, raising=False)
    monkeypatch.setattr(app.state, "sandbox", sandbox, raising=False)
    return TestClient(app)


def provider_transport(monkeypatch, handler):
    """Intercept all provider HTTP calls; no test in this file uses a network."""
    from app import agent

    real_client = httpx.Client
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(agent.httpx, "Client", lambda **kwargs: real_client(
        transport=transport, **kwargs
    ))


def provider_env(monkeypatch, *, url="https://provider.example/v1/chat/completions"):
    monkeypatch.setenv("LLM_CHAT_COMPLETIONS_URL", url)
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-provider-secret-sentinel")


def test_chat_requires_demo_token_and_returns_typed_response(tmp_path, monkeypatch):
    llm = ScriptedLLM(structured_turn("برای پاسخ به داده نیاز است."))
    client = runtime_client(tmp_path, monkeypatch, llm)
    body = {"question": "وضعیت فروش چیست؟"}

    assert client.post("/chat", json=body).status_code == 401
    assert client.post("/chat", json=body, headers={"X-Demo-Token": "invalid"}).status_code == 401
    assert llm.calls == []
    response = client.post("/chat", json=body, headers={"X-Demo-Token": "demo-alpha"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "برای پاسخ به داده نیاز است."
    assert payload["conversation_id"] and payload["trace_id"]
    assert payload["artifacts"] == [] and payload["trace"] == []
    assert len(llm.calls) == 1


def test_application_factory_configures_dependencies_at_startup(tmp_path):
    from app.main import create_app

    llm = ScriptedLLM(structured_turn("داده کافی موجود نیست."))
    service = create_app(
        data_dir=DATA, token_map_path=TOKEN_FILE, schema_path=DATA / "schema.md",
        state_db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts",
        docker_image=IMAGE, llm_client=llm,
    )
    with TestClient(service) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post("/chat", json={"question": "چه می‌دانیم؟"},
                               headers={"X-Demo-Token": "demo-alpha"})
        assert response.status_code == 200
        assert response.json()["answer"] == "داده کافی موجود نیست."
        assert isinstance(service.state.secure_data, SecureDataAccess)
        assert isinstance(service.state.conversation_store, SQLiteConversationStore)
    assert (tmp_path / "state.db").is_file()


def test_application_startup_fails_closed_for_missing_token_mapping(tmp_path):
    from app.main import create_app

    with pytest.raises((OSError, ValueError, RuntimeError)):
        service = create_app(
            data_dir=DATA, token_map_path=tmp_path / "missing-tokens.json",
            schema_path=DATA / "schema.md", state_db_path=tmp_path / "state.db",
            artifact_root=tmp_path / "artifacts", docker_image=IMAGE,
            llm_client=ScriptedLLM(structured_turn("داده کافی موجود نیست.")),
        )
        with TestClient(service):
            pass


@pytest.mark.parametrize(("token", "expected", "prefix"), [
    ("demo-alpha", 53_250_000, "A"),
    ("demo-beta", 106_500_000, "B"),
])
def test_chat_uses_paid_sql_and_tenant_tickets(tmp_path, monkeypatch, token, expected, prefix):
    def after_sql(messages, tools):
        assert tool_result(messages, "sql-1") == {"rows": [{"revenue": expected}]}
        return tool_turn(call("get_tickets", {}, "tickets-2"))

    def after_tickets(messages, tools):
        tickets = tool_result(messages, "tickets-2")["tickets"]
        assert tickets and all(ticket["ticket_id"].startswith(prefix) for ticket in tickets)
        assert all(ticket["is_untrusted"] is True for ticket in tickets)
        return structured_turn(
            "درآمد پرداخت‌شده شهریور {{revenue}} تومان است؛ تیکت‌ها شواهد احتمالی‌اند.",
            revenue=numeric_slot(expected, "sql-1", "revenue"),
        )

    llm = ScriptedLLM(
        tool_turn(call("query_sql", {"sql": SQL, "params": ["2026-09"]}, "sql-1")),
        after_sql, after_tickets,
    )
    client = runtime_client(tmp_path, monkeypatch, llm)
    response = client.post("/chat", json={"question": "درآمد شهریور و تیکت‌ها را بررسی کن"},
                           headers={"X-Demo-Token": token})
    assert response.status_code == 200
    payload = response.json()
    assert str(expected) in payload["answer"]
    assert [entry["tool_name"] for entry in payload["trace"]] == ["query_sql", "get_tickets"]
    assert all(entry["status"] == "success" for entry in payload["trace"])
    assert len(llm.calls) == 3


def test_chat_rejects_foreign_conversation_before_llm(tmp_path, monkeypatch):
    llm = ScriptedLLM(structured_turn("داده کافی موجود نیست."))
    client = runtime_client(tmp_path, monkeypatch, llm)
    first = client.post("/chat", json={"question": "گزارش بساز"},
                        headers={"X-Demo-Token": "demo-alpha"})
    assert first.status_code == 200
    foreign = client.post("/chat", json={"question": "ادامه بده",
                                         "conversation_id": first.json()["conversation_id"]},
                          headers={"X-Demo-Token": "demo-beta"})
    assert foreign.status_code == 404
    assert len(llm.calls) == 1
    assert "demo-alpha" not in foreign.text


def test_chat_sql_tickets_python_artifact_and_restarted_state(
    docker_ready, tmp_path, monkeypatch,
):
    from app.main import app
    from app.sandbox import DockerPythonSandbox

    python_code = ("import base64\nfrom pathlib import Path\n"
                   f"Path('/output/chart.png').write_bytes(base64.b64decode('{PNG}'))")

    def after_sql(messages, tools):
        assert tool_result(messages, "sql-live") == {"rows": [{"revenue": 53_250_000}]}
        return tool_turn(call("get_tickets", {"customer_id": "A001"}, "tickets-live"))

    def after_tickets(messages, tools):
        tickets = tool_result(messages, "tickets-live")["tickets"]
        assert {item["ticket_id"] for item in tickets} == {"AT001", "AT002"}
        return tool_turn(call("execute_python", {"code": python_code}, "python-live"))

    def after_python(messages, tools):
        result = tool_result(messages, "python-live")
        assert result["exit_code"] == 0 and len(result["artifacts"]) == 1
        return structured_turn(
            "درآمد پرداخت‌شده شهریور {{revenue}} تومان است و نمودار آماده است.",
            revenue=numeric_slot(53_250_000, "sql-live", "revenue"),
        )

    llm = ScriptedLLM(
        tool_turn(call("query_sql", {"sql": SQL, "params": ["2026-09"]}, "sql-live")),
        after_sql, after_tickets, after_python,
    )
    sandbox = DockerPythonSandbox(image=IMAGE, artifact_root=tmp_path / "artifacts")
    client = runtime_client(tmp_path, monkeypatch, llm, sandbox=sandbox)
    response = client.post("/chat", json={"question": "درآمد شهریور و تیکت‌های A001 را تحلیل و نمودار بساز"},
                           headers={"X-Demo-Token": "demo-alpha"})
    assert response.status_code == 200
    payload = response.json()
    assert [item["tool_name"] for item in payload["trace"]] == [
        "query_sql", "get_tickets", "execute_python"
    ]
    artifact_id = payload["artifacts"][0]["artifact_id"]
    assert client.get(f"/artifacts/{artifact_id}",
                      headers={"X-Demo-Token": "demo-alpha"}).content == base64.b64decode(PNG)
    assert client.get(f"/artifacts/{artifact_id}",
                      headers={"X-Demo-Token": "demo-beta"}).status_code == 404

    def inspect_history(messages, tools):
        assert any(item.get("role") == "tool" and item.get("tool_call_id") == "sql-live"
                   for item in messages)
        return structured_turn("گزارش و نمودار قبلی در دسترس است.")

    monkeypatch.setattr(app.state, "conversation_store", SQLiteConversationStore(
        db_path=tmp_path / "conversations.db", artifact_root=tmp_path / "artifacts"
    ))
    monkeypatch.setattr(app.state, "llm_client", ScriptedLLM(inspect_history))
    followup = TestClient(app).post(
        "/chat", json={"question": "گزارش قبلی را نشان بده",
                       "conversation_id": payload["conversation_id"]},
        headers={"X-Demo-Token": "demo-alpha"},
    )
    assert followup.status_code == 200
    assert followup.json()["conversation_id"] == payload["conversation_id"]
    assert followup.json()["artifacts"][0]["artifact_id"] == artifact_id


def test_https_chat_completions_adapter_sends_tool_protocol(monkeypatch):
    from app.agent import EnvChatCompletionsClient

    provider_env(monkeypatch)
    seen = []

    def answer(request):
        seen.append(request)
        return httpx.Response(200, json=tool_turn(call("get_schema", {}, "schema-1")))

    provider_transport(monkeypatch, answer)
    messages = [{"role": "user", "content": "سلام"}]
    tools = [{"type": "function", "function": {"name": "get_schema",
              "parameters": {"type": "object", "properties": {}}}}]
    result = EnvChatCompletionsClient().create(messages=messages, tools=tools)
    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_schema"
    assert len(seen) == 1
    assert seen[0].url.scheme == "https"
    assert seen[0].headers["authorization"] == "Bearer test-provider-secret-sentinel"
    assert json.loads(seen[0].content) == {
        "model": "test-model", "messages": messages, "tools": tools,
    }


def test_provider_rejects_plain_http_before_sending_secret(monkeypatch):
    from app.agent import EnvChatCompletionsClient

    provider_env(monkeypatch, url="http://provider.example/v1/chat/completions")
    sent = []
    provider_transport(monkeypatch, lambda request: (
        sent.append(request) or httpx.Response(200, json=final_turn("پاسخ آماده است."))
    ))
    with pytest.raises((ValueError, RuntimeError), match="HTTPS"):
        EnvChatCompletionsClient().create(messages=[{"role": "user", "content": "سلام"}],
                                          tools=[])
    assert sent == []


def test_provider_limits_completion_tokens_per_call(monkeypatch):
    from app.agent import EnvChatCompletionsClient

    provider_env(monkeypatch)
    monkeypatch.setenv("LLM_MAX_COMPLETION_TOKENS", "128")
    sent = []
    provider_transport(monkeypatch, lambda request: (
        sent.append(json.loads(request.content)) or
        httpx.Response(200, json=final_turn("پاسخ آماده است."))
    ))
    EnvChatCompletionsClient().create(messages=[{"role": "user", "content": "سلام"}], tools=[])
    assert len(sent) == 1
    assert 1 <= sent[0]["max_completion_tokens"] <= 128


def test_provider_error_does_not_expose_key(monkeypatch, caplog):
    from app.agent import EnvChatCompletionsClient

    provider_env(monkeypatch)
    seen = []
    provider_transport(monkeypatch, lambda request: (
        seen.append(request) or httpx.Response(401)
    ))
    with pytest.raises(Exception) as caught:
        EnvChatCompletionsClient().create(messages=[{"role": "user", "content": "سلام"}],
                                          tools=[])
    assert len(seen) == 1
    assert "test-provider-secret-sentinel" not in str(caught.value)
    assert "test-provider-secret-sentinel" not in caplog.text


def test_provider_requires_credentials_before_network(monkeypatch):
    from app.agent import EnvChatCompletionsClient

    provider_env(monkeypatch)
    monkeypatch.delenv("LLM_API_KEY")
    sent = []
    provider_transport(monkeypatch, lambda request: (
        sent.append(request) or httpx.Response(200, json=final_turn("پاسخ آماده است."))
    ))
    with pytest.raises(RuntimeError):
        EnvChatCompletionsClient().create(messages=[{"role": "user", "content": "سلام"}],
                                          tools=[])
    assert sent == []


def test_provider_timeout_is_bounded_and_sanitized_at_chat_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-provider-secret-sentinel")
    llm = ScriptedLLM(*[httpx.ReadTimeout("provider timed out") for _ in range(4)])
    client = runtime_client(tmp_path, monkeypatch, llm)
    response = client.post("/chat", json={"question": "درآمد چیست؟"},
                           headers={"X-Demo-Token": "demo-alpha"})
    assert response.status_code in (503, 504)
    assert len(llm.calls) == 3  # initial call and two retries
    assert "test-provider-secret-sentinel" not in response.text
    assert "provider timed out" not in response.text


def test_provider_failure_is_bounded_and_sanitized_at_chat_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-provider-secret-sentinel")
    llm = ScriptedLLM(*[RuntimeError("test-provider-secret-sentinel rejected")
                        for _ in range(4)])
    client = runtime_client(tmp_path, monkeypatch, llm)
    response = client.post("/chat", json={"question": "درآمد چیست؟"},
                           headers={"X-Demo-Token": "demo-alpha"})
    assert response.status_code == 503
    assert len(llm.calls) == 3
    assert "test-provider-secret-sentinel" not in response.text


def test_configured_provider_call_budget_stops_further_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MAX_CALLS_PER_REQUEST", "2")
    monkeypatch.setenv("LLM_API_KEY", "test-provider-secret-sentinel")
    llm = ScriptedLLM(
        tool_turn(call("get_schema", {}, "schema-1")),
        tool_turn(call("get_tickets", {}, "tickets-2")),
        structured_turn("پاسخ آماده است."),
    )
    client = runtime_client(tmp_path, monkeypatch, llm)
    response = client.post("/chat", json={"question": "داده و تیکت را بررسی کن"},
                           headers={"X-Demo-Token": "demo-alpha"})
    assert response.status_code == 429
    assert len(llm.calls) == 2
    assert "test-provider-secret-sentinel" not in response.text


def test_configured_total_token_budget_stops_further_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOTAL_TOKENS_PER_REQUEST", "80")
    first = tool_turn(call("get_schema", {}, "schema-1"))
    first["usage"] = {"prompt_tokens": 70, "completion_tokens": 10, "total_tokens": 80}
    llm = ScriptedLLM(first, structured_turn("پاسخ آماده است."))
    client = runtime_client(tmp_path, monkeypatch, llm)
    response = client.post("/chat", json={"question": "قرارداد داده را بررسی کن"},
                           headers={"X-Demo-Token": "demo-alpha"})
    assert response.status_code == 429
    assert len(llm.calls) == 1
