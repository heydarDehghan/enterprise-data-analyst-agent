"""Phase 4-C-C live qualification; never collected by the default test suite.

These tests make billable OpenRouter calls only with an explicit, separate opt-in.
They exercise the real HTTP adapter through POST /chat, not a scripted LLM.
"""

import ast
import csv
import json
import os
import shutil
import sqlite3
import subprocess
import unicodedata
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agent import EnvChatCompletionsClient
from app.config import load_llm_config
from app.main import create_app
from app.month_scope import requested_month
from app.sandbox import _valid_png
from app.structured_grounding import EvidenceScope
from live_tests.llm_diagnostics import PYTHON_PNG_NODE, collect_agent_observations


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TOKEN_MAP = ROOT / "demo_tokens.json.example"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "qwen/qwen3.8-27b"
IMAGE = "python:3.13-slim"

pytestmark = pytest.mark.skipif(
    os.environ.get("KAMIDSO_RUN_LIVE_AGENT_TESTS") != "1",
    reason="Set KAMIDSO_RUN_LIVE_AGENT_TESTS=1 to authorize live agent traffic",
)


@pytest.fixture(autouse=True)
def live_configuration(monkeypatch, request):
    if os.environ.get("LLM_CHAT_COMPLETIONS_URL") != ENDPOINT:
        pytest.fail("The exact approved OpenRouter endpoint is required")
    if os.environ.get("LLM_MODEL") != MODEL:
        pytest.fail("The exact approved OpenRouter model is required")
    if not os.environ.get("LLM_API_KEY"):
        pytest.fail("LLM_API_KEY must already be configured in the process")
    monkeypatch.setenv("LLM_MAX_CALLS_PER_REQUEST", "8")
    total_token_budget = 40_000 if request.node.nodeid == PYTHON_PNG_NODE else 24_000
    monkeypatch.setenv("LLM_MAX_TOTAL_TOKENS_PER_REQUEST", str(total_token_budget))


class CountingLiveClient:
    """Count real adapter calls without retaining credentials or provider payloads."""

    def __init__(self):
        self.adapter = EnvChatCompletionsClient()
        self.calls = 0
        self.usage_totals = []
        self.completion_totals = []

    def create(self, *, messages, tools):
        self.calls += 1
        response = self.adapter.create(messages=messages, tools=tools)
        usage = response.get("usage")
        self.usage_totals.append(
            usage.get("total_tokens") if isinstance(usage, dict) else None
        )
        self.completion_totals.append(
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        )
        return response


def configured_app(tmp_path, llm):
    return create_app(
        data_dir=DATA,
        token_map_path=TOKEN_MAP,
        schema_path=DATA / "schema.md",
        state_db_path=tmp_path / "conversations.db",
        artifact_root=tmp_path / "artifacts",
        docker_image=IMAGE,
        llm_client=llm,
    )


def snapshot(service, token, conversation_id):
    tenant = service.state.secure_data.authenticate(token)
    return service.state.conversation_store.get_conversation(tenant, conversation_id)


def tool_exchanges(messages):
    calls = {}
    exchanges = []
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                calls[call["id"]] = call["function"]
        elif message.get("role") == "tool":
            call = calls[message["tool_call_id"]]
            exchanges.append((
                call["name"], json.loads(call["arguments"]),
                json.loads(message["content"]),
            ))
    return exchanges


def ascii_digits(text):
    converted = "".join(
        str(unicodedata.decimal(char)) if char.isdecimal() else char
        for char in text
    )
    return converted.replace(",", "").replace("٬", "").replace("،", "")


def assert_persian_amount(answer, amount):
    assert any("\u0600" <= char <= "\u06ff" for char in answer)
    assert str(amount) in ascii_digits(answer)


def paid_september_total(database):
    uri = f"file:{(DATA / database).as_posix()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        return connection.execute(
            "SELECT SUM(amount_toman) FROM orders "
            "WHERE status = 'paid' AND substr(order_date, 1, 7) = '2026-09'"
        ).fetchone()[0]


def expected_tickets(filename, customer_id):
    with (DATA / filename).open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["ticket_id"] for row in csv.DictReader(handle)
            if row["customer_id"] == customer_id
        }


def assert_success_trace(payload, required_tools):
    trace = payload["trace"]
    assert payload["conversation_id"] and payload["trace_id"]
    assert 0 < len(trace) <= 8
    assert [item["step"] for item in trace] == list(range(1, len(trace) + 1))
    assert all(item["status"] == "success" for item in trace)
    assert required_tools <= {item["tool_name"] for item in trace}


@pytest.mark.parametrize(
    ("token", "database", "tickets_file", "customer_id", "foreign_prefix"),
    [
        ("demo-alpha", "alpha.db", "alpha_tickets.csv", "A001", "B"),
        ("demo-beta", "beta.db", "beta_tickets.csv", "B001", "A"),
    ],
)
def test_live_chat_calls_paid_sql_and_tenant_tickets(
    tmp_path, token, database, tickets_file, customer_id, foreign_prefix,
):
    llm = CountingLiveClient()
    service = configured_app(tmp_path, llm)
    question = (
        f"برای مشتری {customer_id} تیکت‌ها را با get_tickets بخوان. "
        "با query_sql مجموع واحدِ درآمد فقط سفارش‌های paid شهریور 2026 "
        "را از paid_orders محاسبه کن. عدد صحیح تومان را با رقم در پاسخ فارسی بنویس."
    )
    with TestClient(service) as client:
        assert client.post("/chat", json={"question": question}).status_code == 401
        assert llm.calls == 0
        response = client.post(
            "/chat", json={"question": question}, headers={"X-Demo-Token": token},
        )
    assert response.status_code == 200, response.status_code
    payload = response.json()
    assert_success_trace(payload, {"query_sql", "get_tickets"})
    assert_persian_amount(payload["answer"], paid_september_total(database))
    assert 1 <= llm.calls <= 8
    assert len(llm.completion_totals) == llm.calls
    completion_limit = load_llm_config().max_completion_tokens
    assert completion_limit is not None
    assert all(isinstance(count, int) and 0 <= count <= completion_limit
               for count in llm.completion_totals)
    exchanges = tool_exchanges(snapshot(service, token, payload["conversation_id"]).messages)
    sql_calls = [(args, result) for name, args, result in exchanges if name == "query_sql"]
    assert sql_calls
    assert all("paid_orders" in args["sql"].lower() for args, _ in sql_calls)
    assert any(
        paid_september_total(database) in row.values()
        for _, result in sql_calls for row in result["rows"]
    )
    ticket_calls = [result["tickets"] for name, _, result in exchanges
                    if name == "get_tickets"]
    assert ticket_calls
    assert any(
        {item["ticket_id"] for item in tickets} ==
        expected_tickets(tickets_file, customer_id)
        for tickets in ticket_calls
    )
    assert all(
        item["is_untrusted"] is True and
        not item["ticket_id"].startswith(foreign_prefix)
        for tickets in ticket_calls for item in tickets
    )


def docker_ready():
    if shutil.which("docker") is None:
        pytest.fail("Docker CLI is required for live Python execution")
    for command in (
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
    ):
        result = subprocess.run(command, capture_output=True, timeout=10, check=False)
        if result.returncode:
            pytest.fail("Docker daemon and the approved image are required")


def matches_python_revenue(stdout, total):
    try:
        output = json.loads(stdout)
    except (TypeError, ValueError):
        return False
    if not isinstance(output, dict) or output.keys() != {"metrics"}:
        return False
    metrics = output["metrics"]
    revenue = metrics.get("revenue") if isinstance(metrics, dict) else None
    return isinstance(revenue, str) and revenue.isdecimal() and Decimal(revenue) == total


def assert_python_revenue_slot(final_content, evidence_refs):
    slots = json.loads(final_content)["slots"]
    assert any(
        slot.get("type") == "numeric" and slot.get("unit") == "scalar"
        and slot.get("evidence_field") == "revenue" and slot.get("evidence_ref") in evidence_refs
        for slot in slots.values()
    ), "The final revenue slot must cite the verified Python result"


def _revenue_reference(accepted_json):
    slots = json.loads(accepted_json)["slots"]
    citations = [slot for slot in slots.values() if slot.get("evidence_field") == "revenue"]
    assert len(citations) == 1, "Expected exactly one revenue citation in the accepted answer"
    citation, = citations
    assert (citation.get("type"), citation.get("unit")) == ("numeric", "scalar")
    ref = citation.get("evidence_ref")
    assert isinstance(ref, str) and ref.strip(), "Expected an issued Python evidence ID"
    return ref


def _issued_tool_calls(messages):
    for message in messages:
        if message["role"] == "assistant":
            yield from message.get("tool_calls") or []


def _verified_python_record(state, scope, ref):
    records = [record for record in state.evidence if record.call_id == ref]
    assert len(records) == 1, "Expected exactly one persisted record for the cited evidence ID"
    record, = records
    assert record.scope == scope, "The cited evidence must belong to this tenant and conversation"
    assert record.tool_name == "execute_python" and record.succeeded is True
    return record


def _resolve_accepted_python_evidence_and_png(accepted_json, response, state, scope, month, total):
    ref = _revenue_reference(accepted_json)
    record = _verified_python_record(state, scope, ref)
    assert len(state.turns) == 1, "Expected only the first turn before the restart"
    turn, = state.turns
    assert (turn.trace_id, turn.status, turn.answer) == (response["trace_id"], "success", response["answer"])

    issued = [call["function"] for call in _issued_tool_calls(state.messages) if call["id"] == ref]
    tool_messages = [message for message in state.messages if message["role"] == "tool"]
    exchanges = [
        (step, message) for step, message in zip(turn.trace, tool_messages, strict=True)
        if message["tool_call_id"] == ref
    ]
    assert len(issued) == len(exchanges) == 1, "Expected one original tool exchange for the cited ID"
    function, = issued
    (step, message), = exchanges
    assert function["name"] == step.tool_name == "execute_python" and step.status == "success"
    result = json.loads(message["content"])
    assert result == record.result, "Persisted evidence must match the original tool result"
    assert type(result.get("exit_code")) is int and result["exit_code"] == 0 and result["stderr"] == ""
    assert matches_python_revenue(result["stdout"], total), "Expected Python revenue to match read-only SQLite"

    code = json.loads(function["arguments"])["code"]
    literals = {
        node.value for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    # Docker provides the tenant-scoped export. These source checks corroborate
    # the input and period alongside the independent revenue comparison.
    assert "/data/paid_orders.csv" in literals or {"/data", "paid_orders.csv"} <= literals
    assert month.gregorian_month in literals or (
        month.first in literals and bool({month.last, month.next_first} & literals)
    ), "Expected the cited Python code to specify the requested Gregorian month"

    pngs = [item for item in result["artifacts"] if (item["kind"], item["media_type"]) == ("image", "image/png")]
    assert len(pngs) == 1, "Expected one unambiguous PNG from the cited execution"
    png, = pngs
    artifact_id = png["artifact_id"]
    assert [item for item in response["artifacts"] if item["artifact_id"] == artifact_id] == [png]
    assert [item.model_dump() for item in state.artifacts if item.artifact_id == artifact_id] == [png]
    return record, png


def _download_verified_png(client, path):
    owner = client.get(path, headers={"X-Demo-Token": "demo-alpha"})
    assert owner.status_code == 200 and owner.headers["content-type"] == "image/png"
    assert _valid_png(owner.content), "Expected a valid PNG from the verified artifact"
    assert client.get(path, headers={"X-Demo-Token": "demo-beta"}).status_code == 404
    return owner.content


class _PythonPngLiveClient(CountingLiveClient):
    def __init__(self):
        super().__init__()
        self.evidence_manifests = []
        self.provider_calls_allowed = True

    def create(self, *, messages, tools):
        assert self.provider_calls_allowed, "Authorization probe must not contact the provider"
        manifest, = [
            line.removeprefix("KAMIDSO_EVIDENCE_V1 ") for line in messages[0]["content"].splitlines()
            if line.startswith("KAMIDSO_EVIDENCE_V1 ")
        ]
        self.evidence_manifests.append(json.loads(manifest))
        return super().create(messages=messages, tools=tools)


class _AcceptedFinal:
    """Match a successful chat response to the raw content grounded on that turn."""

    def __init__(self):
        self._events = []

    def __call__(self, event, **values):
        if event in ("turn", "provider_start", "grounding_start", "grounding_end"):
            self._events.append((event, values))

    def for_response(self, response):
        trace_id = response["trace_id"]
        expected = f"Expected exactly one accepted grounding attempt for trace {trace_id}"
        turns = [values for event, values in self._events if event == "turn"]
        assert len(turns) == 1 and self._events[0][0] == "turn", expected
        turn = turns[0]
        assert turn.get("trace_id") == trace_id and turn.get("conversation_id") == response["conversation_id"], expected

        starts = [i for i, (event, _) in enumerate(self._events) if event == "provider_start"]
        assert starts and starts[0] == 1, expected
        indices = [self._events[start][1].get("index") for start in starts]
        assert all(type(index) is int and index > 0 for index in indices), expected
        assert indices == sorted(set(indices)), expected

        accepted = []
        for start, end in zip(starts, [*starts[1:], len(self._events)], strict=True):
            grounding = self._events[start + 1:end]
            if not grounding:
                continue
            assert [event for event, _ in grounding] == ["grounding_start", "grounding_end"], expected
            (_, draft), (_, outcome) = grounding
            if outcome["rejection_reason"] is None:
                accepted.append((draft["raw_final_content"], outcome["rendered_answer"]))

        assert len(accepted) == 1, expected
        raw, rendered = accepted[0]
        assert isinstance(raw, str) and raw, expected
        assert isinstance(rendered, str) and rendered == response["answer"], expected
        return raw


def test_live_python_png_and_conversation_reuse(tmp_path):
    docker_ready()
    llm = _PythonPngLiveClient()
    service = configured_app(tmp_path, llm)
    total = paid_september_total("alpha.db")
    question = (
        "با execute_python در Docker برنامه Python بساز و اجرا کن. فقط از CSVهای "
        "مجاز همین مستأجر در /data بخوان. از /data/paid_orders.csv مجموع درآمد "
        "سفارش‌های paid را برای سپتامبر میلادی 2026 حساب کن "
        "(از 2026-09-01 شامل آن روز تا قبل از 2026-10-01). فقط با کتابخانه استاندارد "
        "Python و بدون نصب بسته، یک نمودار PNG معتبر و کوچک در "
        "/output/paid_revenue.png بساز. در stdout دقیقاً یک شیء JSON به شکل "
        '{"metrics":{"revenue":"<computed_total>"}} '
        "چاپ کن. مقدار revenue را از مجموع محاسبه‌شده به صورت رشته‌ای از رقم‌های صحیح "
        "پر کن و مقدار مورد انتظار را در کد ثابت ننویس. هیچ متن دیگری از جمله "
        "پیام اشکال‌زدایی را در stdout ننویس. سپس عدد دقیق تومان را در پاسخ فارسی "
        "با یک slot عددی با واحد scalar گزارش کن و برای فیلد revenue به شناسهٔ "
        "واقعی شاهد execute_python ارجاع بده."
    )
    first_final = _AcceptedFinal()
    with collect_agent_observations(first_final), TestClient(service) as client:
        response = client.post(
            "/chat", json={"question": question},
            headers={"X-Demo-Token": "demo-alpha"},
        )
    assert response.status_code == 200, response.status_code
    first = response.json()
    assert_success_trace(first, {"execute_python"})
    assert_persian_amount(first["answer"], total)
    first_raw = first_final.for_response(first)
    tenant = service.state.secure_data.authenticate("demo-alpha")
    assert tenant.display_label == "Alpha"
    scope = EvidenceScope(tenant.state_key, first["conversation_id"])
    month = requested_month(question)
    assert month is not None and month.gregorian_month == "2026-09"
    first_state = snapshot(service, "demo-alpha", first["conversation_id"])
    record, png = _resolve_accepted_python_evidence_and_png(first_raw, first, first_state, scope, month, total)
    evidence_refs = {record.call_id}
    assert_python_revenue_slot(first_raw, evidence_refs)
    path = f"/artifacts/{png['artifact_id']}"
    with TestClient(service) as client:
        image_bytes = _download_verified_png(client, path)

    restarted = configured_app(tmp_path, llm)
    followup_offset = len(llm.evidence_manifests)
    followup_body = {
        "question": (
            "عدد دقیق گزارش قبلی را با رقم دوباره بگو و نمودار قبلی را نگه دار. "
            "برای slot عددی از همان شاهد execute_python و فیلد revenue استفاده کن."
        ),
        "conversation_id": first["conversation_id"],
    }
    followup_final = _AcceptedFinal()
    with collect_agent_observations(followup_final), TestClient(restarted) as client:
        followup = client.post(
            "/chat", json=followup_body, headers={"X-Demo-Token": "demo-alpha"},
        )
    assert followup.status_code == 200, followup.status_code
    second = followup.json()
    assert second["conversation_id"] == first["conversation_id"]
    assert second["trace_id"] != first["trace_id"]
    assert_persian_amount(second["answer"], total)
    second_raw = followup_final.for_response(second)
    assert_python_revenue_slot(second_raw, evidence_refs)
    assert _revenue_reference(second_raw) == record.call_id
    assert png in second["artifacts"]
    assert restarted.state.secure_data.authenticate("demo-alpha").state_key == scope.tenant_key
    second_state = snapshot(restarted, "demo-alpha", first["conversation_id"])
    turns = second_state.turns
    assert [turn.trace_id for turn in turns] == [first["trace_id"], second["trace_id"]]
    assert [turn.status for turn in turns] == ["success", "success"]
    assert turns[0].trace
    assert _verified_python_record(second_state, scope, record.call_id) == record
    assert second_state.messages[:len(first_state.messages)] == first_state.messages
    followup_messages = second_state.messages[len(first_state.messages):]
    assert all(step.tool_name != "execute_python" for step in turns[1].trace), "Follow-up executed Python"
    assert all(
        call["function"]["name"] != "execute_python" for call in _issued_tool_calls(followup_messages)
    ), "Follow-up attempted a Python tool call"

    manifest = llm.evidence_manifests[followup_offset]
    assert manifest["context"]["user_request"] == {
        "period": month.gregorian_month, "period_start": month.first, "period_end": month.last,
    }
    advertised = [tool for tool in manifest["tools"] if tool["evidence_ref"] == record.call_id]
    assert len(advertised) == 1 and advertised[0]["tool_name"] == "execute_python"
    assert {"evidence_field": "revenue", "type": "numeric", "unit": "scalar"} in advertised[0]["fields"]

    llm.provider_calls_allowed = False
    with TestClient(restarted) as client:
        assert _download_verified_png(client, path) == image_bytes
        request = client.build_request(
            "POST", "/chat", json=followup_body, headers={"X-Demo-Token": "demo-beta"},
        )
        denied = client.send(request)
    assert denied.status_code == 404 and denied.json() == {"detail": "Conversation not found"}


def test_live_provider_call_budget_stops_after_one_tool_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MAX_CALLS_PER_REQUEST", "1")
    llm = CountingLiveClient()
    service = configured_app(tmp_path, llm)
    with TestClient(service) as client:
        response = client.post(
            "/chat", json={"question": (
                "با query_sql مجموع درآمد paid شهریور 2026 را از paid_orders "
                "حساب کن و عدد دقیق را با رقم پاسخ بده."
            )}, headers={"X-Demo-Token": "demo-alpha"},
        )
    assert response.status_code == 429
    assert llm.calls == 1
    assert response.json()["detail"] == "Request budget exceeded"
    tenant = service.state.secure_data.authenticate("demo-alpha")
    # The first provider completion must have caused a real tool step.
    with sqlite3.connect(tmp_path / "conversations.db") as connection:
        conversation_id = connection.execute(
            "SELECT conversation_id FROM conversations"
        ).fetchone()[0]
    turns = service.state.conversation_store.get_conversation(tenant, conversation_id).turns
    assert len(turns) == 1 and turns[0].status == "failed"
    assert any(item.tool_name == "query_sql" and item.status == "success"
               for item in turns[0].trace)


def test_live_total_token_budget_stops_after_one_completion(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOTAL_TOKENS_PER_REQUEST", "1")
    llm = CountingLiveClient()
    service = configured_app(tmp_path, llm)
    with TestClient(service) as client:
        response = client.post(
            "/chat", json={"question": "به فارسی یک جمله کوتاه بنویس."},
            headers={"X-Demo-Token": "demo-alpha"},
        )
    assert response.status_code == 429
    assert llm.calls == 1
    assert len(llm.usage_totals) == 1
    assert isinstance(llm.usage_totals[0], int) and llm.usage_totals[0] >= 1
    assert response.json()["detail"] == "Request budget exceeded"
