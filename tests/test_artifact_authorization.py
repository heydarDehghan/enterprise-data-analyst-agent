"""Phase 4-A RED contract for durable artifacts and tenant-authorized reads.

The Agent may reuse a previous sandbox result after a store restart. The HTTP
artifact boundary must authenticate a demo token, resolve the tenant on the
server, check stored ownership, and serve only a registered PNG file.
"""

import base64
from uuid import uuid4

from fastapi.testclient import TestClient

from app.schemas import ArtifactMetadata, ChatRequest
from app.secure_data import SecureDataAccess

from test_agent_loop import DATA, TOKEN_MAP, ScriptedLLM, call, structured_turn, tool_result, tool_turn
from test_python_sandbox import PNG
from test_state_recovery import agent, state_store, tenant


def test_artifact_download_route_is_registered():
    from app.main import app

    assert any(route.path == "/artifacts/{artifact_id}" and "GET" in route.methods
               for route in app.routes)


def create_artifact_conversation(tmp_path):
    from app.sandbox import SandboxResult

    store = state_store(tmp_path)
    alpha = tenant()
    artifact_id = uuid4().hex
    payload = base64.b64decode(PNG)
    artifact_path = tmp_path / "artifacts" / f"{artifact_id}.png"

    class SandboxStub:
        calls = 0

        def execute(self, code, subject):
            self.calls += 1
            assert subject is alpha
            assert code == "print('chart ready')"
            artifact_path.write_bytes(payload)
            return SandboxResult(
                stdout="chart ready\n", stderr="", exit_code=0,
                artifacts=[ArtifactMetadata(
                    artifact_id=artifact_id, kind="image", title="paid-chart",
                    media_type="image/png",
                )],
            )

    sandbox = SandboxStub()

    def check_execution(messages, tools):
        result = tool_result(messages, "python-1")
        assert result["stdout"] == "chart ready\n"
        assert result["artifacts"][0]["artifact_id"] == artifact_id
        return structured_turn("نمودار سفارش‌های پرداخت‌شده آماده است.")

    first = agent(store, ScriptedLLM(
        tool_turn(call("execute_python", {"code": "print('chart ready')"}, "python-1")),
        check_execution,
    ), alpha, sandbox=sandbox).run(ChatRequest(question="یک نمودار از سفارش‌های پرداخت‌شده بساز"))
    assert sandbox.calls == 1
    assert [item.artifact_id for item in first.artifacts] == [artifact_id]
    return first, artifact_id, payload


def test_prior_artifact_and_tool_result_reused_after_store_restart(tmp_path):
    first, artifact_id, payload = create_artifact_conversation(tmp_path)
    reopened, alpha = state_store(tmp_path), tenant()

    def inspect_replay(messages, tools):
        prior = tool_result(messages, "python-1")
        assert prior["artifacts"][0]["artifact_id"] == artifact_id
        assert prior["stdout"] == "chart ready\n"
        assert messages[-1]["content"] == "نمودار قبلی را دوباره نشان بده"
        return structured_turn("نمودار قبلی آماده است.")

    followup = agent(reopened, ScriptedLLM(inspect_replay), alpha, "trace-artifact-2").run(
        ChatRequest(question="نمودار قبلی را دوباره نشان بده",
                    conversation_id=first.conversation_id)
    )
    assert followup.conversation_id == first.conversation_id
    assert [item.artifact_id for item in followup.artifacts] == [artifact_id]
    snapshot = reopened.get_conversation(alpha, first.conversation_id)
    assert [item.artifact_id for item in snapshot.artifacts] == [artifact_id]
    assert (tmp_path / "artifacts" / f"{artifact_id}.png").read_bytes() == payload


def test_artifact_http_retrieval_enforces_tenant_and_registration(
    tmp_path, monkeypatch,
):
    from app.main import app

    _, artifact_id, payload = create_artifact_conversation(tmp_path)
    reopened = state_store(tmp_path)
    auth = SecureDataAccess(data_dir=DATA, token_map=TOKEN_MAP)
    monkeypatch.setattr(app.state, "secure_data", auth, raising=False)
    monkeypatch.setattr(app.state, "conversation_store", reopened, raising=False)

    client = TestClient(app)
    path = f"/artifacts/{artifact_id}"
    alpha = client.get(path, headers={"X-Demo-Token": "demo-alpha"})
    assert alpha.status_code == 200
    assert alpha.headers["content-type"].startswith("image/png")
    assert alpha.content == payload

    beta = client.get(path, headers={"X-Demo-Token": "demo-beta"})
    assert beta.status_code == 404
    assert artifact_id not in beta.text

    missing_token = client.get(path)
    assert missing_token.status_code == 401

    # A file merely present under the artifact root is never sufficient.
    unregistered_id = uuid4().hex
    (tmp_path / "artifacts" / f"{unregistered_id}.png").write_bytes(payload)
    unregistered = client.get(
        f"/artifacts/{unregistered_id}",
        headers={"X-Demo-Token": "demo-alpha"},
    )
    assert unregistered.status_code == 404
    unknown = client.get(
        f"/artifacts/{uuid4().hex}",
        headers={"X-Demo-Token": "demo-alpha"},
    )
    assert unknown.status_code == 404


def test_missing_artifact_file_fails_closed_after_restart(tmp_path, monkeypatch):
    from app.main import app

    _, artifact_id, _ = create_artifact_conversation(tmp_path)
    (tmp_path / "artifacts" / f"{artifact_id}.png").unlink()
    monkeypatch.setattr(
        app.state, "secure_data",
        SecureDataAccess(data_dir=DATA, token_map=TOKEN_MAP), raising=False,
    )
    monkeypatch.setattr(app.state, "conversation_store", state_store(tmp_path), raising=False)
    response = TestClient(app).get(
        f"/artifacts/{artifact_id}", headers={"X-Demo-Token": "demo-alpha"},
    )
    assert response.status_code == 404
    assert str(tmp_path) not in response.text
