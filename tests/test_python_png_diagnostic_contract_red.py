"""Phase 4-C-C.5-A. Exercise Python and PNG diagnostics without provider traffic."""

import json
import struct
import zlib
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agent import EnvChatCompletionsClient
from app.main import create_app
from live_tests import conftest as live_fixtures
from live_tests.llm_diagnostics import capture_run
from test_agent_loop import call, numeric_slot, structured_turn, tool_turn
from test_beta_diagnostic_contract_red import BETA_NODE, fixture_scope, load_capture, reference_total
from test_evidence_reference_contract_red import EXAMPLE_PREFIX, manifest, system_block
from test_live_llm_integration import DATA, TOKEN_FILE
from test_llm_io_diagnostics_red import (
    CAPTURE_FLAG,
    CAPTURE_PATH,
    body_bytes,
    fake_provider,
    isolated_provider_environment,
    wire,
)
from test_python_sandbox import IMAGE, docker_ready


PYTHON_NODE = "live_tests/test_agent_end_to_end_live.py::test_live_python_png_and_conversation_reuse"
CALL_ID = "python-png-diagnostic"
QUESTION = "درآمد پرداخت‌شده سپتامبر ۲۰۲۶ را با پایتون محاسبه کن و نمودار بساز."
FOLLOWUP = "عدد گزارش قبلی را دوباره بگو و نمودار را نگه دار."

# Represents the code supplied by a scripted model response. Only Docker executes it.
PYTHON_CODE = """\
import csv
import json
import os
import struct
import zlib
from pathlib import Path

data = Path('/data')
assert Path('/.dockerenv').exists() and os.getuid() == 65534
assert {path.name for path in data.iterdir()} == {'customers.csv', 'paid_orders.csv', 'tickets.csv'}
exports = {}
for path in data.iterdir():
    with path.open(encoding='utf-8', newline='') as stream:
        exports[path.name] = list(csv.DictReader(stream))
assert all(rows and all(row['customer_id'].startswith('A') for row in rows)
           for rows in exports.values())
orders = exports['paid_orders.csv']
assert all(row['status'] == 'paid' for row in orders)
total = sum(int(row['amount_toman']) for row in orders
            if '2026-09-01' <= row['order_date'] < '2026-10-01')

def chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))

width, height = 64, 8
bar_width = min(width, total // 1_000_000)
row = b'\\x00' + b'\\x20' * bar_width + b'\\xff' * (width - bar_width)
image = b'\\x89PNG\\r\\n\\x1a\\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0))
image += chunk(b'IDAT', zlib.compress(row * height)) + chunk(b'IEND', b'')
Path('/output/paid_revenue.png').write_bytes(image)
print(json.dumps({'metrics': {'revenue': str(total)}}))
"""


@pytest.mark.parametrize(
    ("selectors", "accepted"),
    [
        ([PYTHON_NODE], True),
        ([PYTHON_NODE, BETA_NODE], False),
        ([PYTHON_NODE, PYTHON_NODE], False),
        ([PYTHON_NODE.split("::")[0]], False),
        ([PYTHON_NODE.removesuffix("_reuse")], False),
        ([PYTHON_NODE + "_lookalike"], False),
    ],
    ids=["exact", "mixed", "duplicate", "module", "partial", "lookalike"],
)
def test_collection_requires_exact_python_png_selector(monkeypatch, selectors, accepted):
    monkeypatch.setenv(CAPTURE_FLAG, "1")
    root = Path(__file__).resolve().parents[1]
    session = SimpleNamespace(config=SimpleNamespace(
        args=selectors, invocation_params=SimpleNamespace(dir=root), rootpath=root,
    ))
    if accepted:
        live_fixtures.pytest_collection_finish(session)
    else:
        with pytest.raises(pytest.UsageError):
            live_fixtures.pytest_collection_finish(session)


def test_single_python_png_item_reaches_capture_fixture(monkeypatch):
    monkeypatch.setenv(CAPTURE_FLAG, "1")
    seen = []

    def record_selection(*, node_id, monkeypatch):
        seen.append(node_id)
        return nullcontext()

    monkeypatch.setattr(live_fixtures, "capture_run", record_selection)
    with fixture_scope(monkeypatch, [PYTHON_NODE]):
        assert seen == [PYTHON_NODE]


def assert_decoded_chart(content, total):
    stream = BytesIO(content)
    assert stream.read(8) == b"\x89PNG\r\n\x1a\n"
    chunks = []
    while header := stream.read(8):
        size, kind = struct.unpack(">I4s", header)
        payload = stream.read(size)
        assert len(payload) == size
        assert stream.read(4) == struct.pack(">I", zlib.crc32(kind + payload))
        chunks.append((kind, payload))
    assert [kind for kind, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"]
    assert struct.unpack(">IIBBBBB", chunks[0][1]) == (64, 8, 8, 0, 0, 0, 0)
    pixels = zlib.decompress(chunks[1][1])
    assert len(pixels) == 8 * 65
    expected_width = total // 1_000_000
    expected_row = bytes([0] + [32] * expected_width + [255] * (64 - expected_width))
    assert pixels == expected_row * 8


def assert_python_capture(path, sent, replies, first, second, result, final_contents):
    capture = load_capture(path, PYTHON_NODE)
    assert capture["runtime_limits"] == {"max_calls": 8, "max_total_tokens": 24_000}
    calls = capture["provider_calls"]
    assert len(calls) == len({item["call_id"] for item in calls}) == 3
    assert [item["trace_id"] for item in calls] == [first["trace_id"], first["trace_id"], second["trace_id"]]
    assert all(item["conversation_id"] == first["conversation_id"] for item in calls)
    for item, request, (status, response) in zip(calls, sent, replies, strict=True):
        assert body_bytes(item["request"]) == request
        assert body_bytes(item["response"]) == response
        assert item["response"]["status_code"] == status
        assert item["request"]["headers"]["authorization"] == "[REDACTED]"

    execution, = capture["tool_executions"]
    assert (execution["provider_call_id"], execution["tool_call_id"], execution["tool_name"],
            execution["trace_id"], execution["conversation_id"], execution["status"]) == (
        calls[0]["call_id"], CALL_ID, "execute_python", first["trace_id"], first["conversation_id"], "success",
    )
    assert execution["arguments"] == {"code": PYTHON_CODE}
    assert json.loads(execution["result_text"]) == result
    second_request = json.loads(sent[1])
    tool_message, = (msg for msg in second_request["messages"] if msg["role"] == "tool")
    assert tool_message["content"] == execution["result_text"]
    for entry, provider, response, final in zip(
        capture["grounding"], calls[1:], [first, second], final_contents, strict=True,
    ):
        assert entry["provider_call_id"] == provider["call_id"]
        assert entry["trace_id"] == response["trace_id"]
        assert entry["conversation_id"] == response["conversation_id"]
        assert entry["raw_final_content"] == final
        assert entry["rendered_answer"] == response["answer"] and entry["rejection_reason"] is None
        slot, = entry["slot_decisions"]
        assert (slot["slot"], slot["evidence_ref"], slot["evidence_field"], slot["type"], slot["unit"],
                slot["decision"], str(slot["authoritative_value"])) == (
            "revenue", CALL_ID, "revenue", "numeric", "scalar", "accepted",
            json.loads(result["stdout"])["metrics"]["revenue"],
        )


@pytest.mark.parametrize("capture_enabled", [False, True], ids=["capture-off", "capture-on"])
def test_python_png_diagnostics_record_real_docker_and_recovered_evidence(
    docker_ready, tmp_path, monkeypatch, capture_enabled,
):
    total = reference_total("alpha.db")
    assert total == 53_250_000
    first_template = "درآمد پرداخت‌شده {{revenue}} تومان است و نمودار آماده شد."
    second_template = "درآمد گزارش قبلی {{revenue}} تومان بود."
    replies = [
        (200, wire(tool_turn(call("execute_python", {"code": PYTHON_CODE}, CALL_ID)), cost=0.00017)),
    ]
    final_contents = []
    for template in (first_template, second_template):
        completion = structured_turn(template, revenue=numeric_slot(total, CALL_ID, "revenue", "scalar"))
        final_contents.append(completion["choices"][0]["message"]["content"])
        replies.append((200, wire(completion, cost=0.00019)))
    sent = fake_provider(monkeypatch, replies)
    path = tmp_path / "python-png.json"
    monkeypatch.setenv(CAPTURE_PATH, str(path))
    if capture_enabled:
        monkeypatch.setenv(CAPTURE_FLAG, "1")
    options = dict(
        data_dir=DATA, token_map_path=TOKEN_FILE, schema_path=DATA / "schema.md",
        state_db_path=tmp_path / "conversation.db", artifact_root=tmp_path / "artifacts",
        docker_image=IMAGE, llm_client=EnvChatCompletionsClient(),
    )
    owner = {"X-Demo-Token": "demo-alpha"}
    foreign = {"X-Demo-Token": "demo-beta"}
    with capture_run(node_id=PYTHON_NODE, monkeypatch=monkeypatch):
        with TestClient(create_app(**options)) as client:
            response = client.post("/chat", headers=owner, json={"question": QUESTION})
            assert response.status_code == 200, response.text
            first = response.json()
            artifact, = first["artifacts"]
            url = f"/artifacts/{artifact['artifact_id']}"
            image = client.get(url, headers=owner)
            assert image.status_code == 200 and image.headers["content-type"] == "image/png"
            assert_decoded_chart(image.content, total)
            assert client.get(url, headers=foreign).status_code == 404

        restarted = create_app(**options)
        with TestClient(restarted) as client:
            followup = {"question": FOLLOWUP, "conversation_id": first["conversation_id"]}
            assert client.post("/chat", headers=foreign, json=followup).status_code == 404
            assert len(sent) == 2
            response = client.post("/chat", headers=owner, json=followup)
            assert response.status_code == 200, response.text
            second = response.json()
            current_image = client.get(url, headers=owner)
            assert current_image.status_code == 200 and current_image.content == image.content
            tenant = restarted.state.secure_data.authenticate("demo-alpha")
            state = restarted.state.conversation_store.get_conversation(tenant, first["conversation_id"])

    step, = first["trace"]
    assert (step["step"], step["tool_name"], step["status"], step["error"]) == (1, "execute_python", "success", None)
    assert first["answer"] == first_template.replace("{{revenue}}", str(total))
    assert second["answer"] == second_template.replace("{{revenue}}", str(total))
    assert second["conversation_id"] == first["conversation_id"] and second["trace_id"] != first["trace_id"]
    assert second["trace"] == [] and second["artifacts"] == [artifact]
    assert [(turn.trace_id, turn.answer, turn.failure_code) for turn in state.turns] == [
        (item["trace_id"], item["answer"], None) for item in (first, second)
    ]
    recovered, = state.evidence
    assert recovered.call_id == CALL_ID and recovered.scope.tenant_key == tenant.state_key
    assert recovered.scope.conversation_id == first["conversation_id"]
    result = recovered.result
    assert result["exit_code"] == 0 and result["stderr"] == ""
    assert json.loads(result["stdout"]) == {"metrics": {"revenue": str(total)}}
    assert result["artifacts"] == [artifact]
    requests = [json.loads(raw) for raw in sent]
    assert len(requests) == 3
    assert "execute_python" in {tool["function"]["name"] for tool in requests[0]["tools"]}
    assert manifest(requests[0]["messages"])["tools"] == []
    for request in requests[1:]:
        assert manifest(request["messages"])["tools"] == [{
            "evidence_ref": CALL_ID, "tool_name": "execute_python",
            "fields": [{"evidence_field": "revenue", "type": "numeric", "unit": "scalar"}],
        }]
        assert system_block(request["messages"], EXAMPLE_PREFIX)["scalar"]["unit"] == "scalar"
    if capture_enabled:
        assert_python_capture(path, sent, replies, first, second, result, final_contents)
    else:
        assert not path.exists()
