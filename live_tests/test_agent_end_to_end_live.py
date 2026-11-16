"""Phase 4-C-C live qualification; never collected by the default test suite.

These tests make billable OpenRouter calls only with an explicit, separate opt-in.
They exercise the real HTTP adapter through POST /chat, not a scripted LLM.
"""

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
from live_tests.llm_diagnostics import PYTHON_PNG_NODE


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


def test_live_python_png_and_conversation_reuse(tmp_path):
    docker_ready()
    final_contents = []

    class PythonRevenueClient(CountingLiveClient):
        def create(self, *, messages, tools):
            completion = super().create(messages=messages, tools=tools)
            choice = completion["choices"][0]
            if choice["finish_reason"] == "stop":
                final_contents.append(choice["message"].get("content"))
            return completion

    llm = PythonRevenueClient()
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
    with TestClient(service) as client:
        response = client.post(
            "/chat", json={"question": question},
            headers={"X-Demo-Token": "demo-alpha"},
        )
    assert response.status_code == 200, response.status_code
    first = response.json()
    assert_success_trace(first, {"execute_python"})
    assert_persian_amount(first["answer"], total)
    assert len(first["artifacts"]) >= 1
    artifact_id = first["artifacts"][0]["artifact_id"]
    first_state = snapshot(service, "demo-alpha", first["conversation_id"])
    python_results = [
        result for name, args, result in tool_exchanges(first_state.messages)
        if name == "execute_python" and "paid_orders.csv" in args["code"]
        and result["exit_code"] == 0 and matches_python_revenue(result["stdout"], total)
        and any(item["artifact_id"] == artifact_id for item in result["artifacts"])
    ]
    assert python_results, "Expected the retrieved PNG and correct structured revenue from Python"
    evidence_refs = {
        record.call_id for record in first_state.evidence
        if record.tool_name == "execute_python" and record.succeeded and record.result in python_results
    }
    assert evidence_refs, "Expected persisted evidence for the successful Python execution"
    first_final, = final_contents
    assert_python_revenue_slot(first_final, evidence_refs)
    with TestClient(service) as client:
        image = client.get(
            f"/artifacts/{artifact_id}", headers={"X-Demo-Token": "demo-alpha"},
        )
        foreign = client.get(
            f"/artifacts/{artifact_id}", headers={"X-Demo-Token": "demo-beta"},
        )
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert image.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert foreign.status_code == 404

    restarted = configured_app(tmp_path, llm)
    with TestClient(restarted) as client:
        followup = client.post(
            "/chat", json={
                "question": (
                    "عدد دقیق گزارش قبلی را با رقم دوباره بگو و نمودار قبلی را نگه دار. "
                    "برای slot عددی از همان شاهد execute_python و فیلد revenue استفاده کن."
                ),
                "conversation_id": first["conversation_id"],
            }, headers={"X-Demo-Token": "demo-alpha"},
        )
    assert followup.status_code == 200, followup.status_code
    second = followup.json()
    assert second["conversation_id"] == first["conversation_id"]
    assert second["trace_id"] != first["trace_id"]
    assert_persian_amount(second["answer"], total)
    _, followup_final = final_contents
    assert_python_revenue_slot(followup_final, evidence_refs)
    assert artifact_id in {item["artifact_id"] for item in second["artifacts"]}
    turns = snapshot(restarted, "demo-alpha", first["conversation_id"]).turns
    assert [turn.trace_id for turn in turns] == [first["trace_id"], second["trace_id"]]
    assert [turn.status for turn in turns] == ["success", "success"]
    assert turns[0].trace


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
