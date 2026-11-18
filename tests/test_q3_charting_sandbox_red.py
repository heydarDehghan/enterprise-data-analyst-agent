"""Q3-B contract for a chart-ready image through the real Docker and HTTP paths."""

import json
import os
import sqlite3
import subprocess
from contextlib import closing

from fastapi.testclient import TestClient

from app import config
from app.main import create_app
from app.sandbox import DockerPythonSandbox, _valid_png
from test_agent_loop import ScriptedLLM, call, numeric_slot, structured_turn, tool_result, tool_turn
from test_evidence_reference_contract_red import manifest
from test_python_sandbox import DATA, IMAGE, ROOT, docker_ready


QUESTION = "نمودار روند فروش پرداخت‌شده به تفکیک دسته محصول در اوت، سپتامبر و اکتبر ۲۰۲۶ را تولید کن."
CALL_ID = "q3-render-chart"

# A scripted provider supplies this sample program, and the real sandbox executes it.
CHART_CODE = """\
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

assert Path('/.dockerenv').is_file() and os.geteuid() == 65534
assert set(os.listdir('/sys/class/net')) == {'lo'}
assert 'LLM_API_KEY' not in os.environ
assert int(Path('/sys/fs/cgroup/memory.max').read_text()) <= 512 * 1024 * 1024
mounts = {fields[4]: fields[5].split(',') for fields in
          (line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines())}
assert all('ro' in mounts[path] for path in ('/', '/data', '/work'))

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

assert matplotlib.get_backend().lower() == 'agg'
assert Path(matplotlib.get_configdir()).is_relative_to('/tmp')
totals = defaultdict(int)
with Path('/data/paid_orders.csv').open(encoding='utf-8', newline='') as stream:
    for order in csv.DictReader(stream):
        assert order['status'] == 'paid' and order['customer_id'].startswith('A')
        month = order['order_date'][:7]
        if '2026-08' <= month <= '2026-10':
            totals[month, order['category']] += int(order['amount_toman'])

months = sorted({month for month, _ in totals})
categories = sorted({category for _, category in totals})
figure, axes = plt.subplots(figsize=(6, 4), dpi=100)
for category in categories:
    axes.plot(months, [totals[month, category] for month in months], marker='o', label=category)
axes.set(xlabel='Gregorian month', ylabel='Paid revenue (toman)')
axes.legend()
figure.tight_layout()
path = Path('/output/paid-revenue-by-category.png')
figure.savefig(path)
metrics = {
    f"revenue_{month.replace('-', '_')}_{line.get_label()}": str(int(value))
    for line in axes.get_lines()
    for month, value in zip(line.get_xdata(), line.get_ydata(), strict=True)
}
plt.close(figure)
with Image.open(path) as image:
    image.load()
    assert image.format == 'PNG' and image.size == (600, 400)
    assert any(lo < hi for lo, hi in image.convert('RGB').getextrema())
print(json.dumps({'metrics': metrics}))
"""


def expected_paid_series():
    uri = (DATA / "alpha.db").resolve().as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        rows = db.execute(
            "SELECT substr(order_date, 1, 7) AS month, category, SUM(amount_toman) "
            "FROM orders WHERE status = 'paid' AND order_date >= ? AND order_date < ? "
            "GROUP BY month, category",
            ("2026-08-01", "2026-11-01"),
        ).fetchall()
    assert len(rows) == 12
    assert {month for month, _, _ in rows} == {"2026-08", "2026-09", "2026-10"}
    assert len({category for _, category, _ in rows}) == 4
    return {f"revenue_{month.replace('-', '_')}_{category}": str(amount) for month, category, amount in rows}


def test_chart_image_renders_paid_series_and_serves_only_to_its_tenant(docker_ready, tmp_path, monkeypatch):
    image = os.environ.get("KAMIDSO_CHART_SANDBOX_IMAGE", IMAGE)
    local = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True, text=True, timeout=5, check=False,
    )
    assert local.returncode == 0 and local.stdout.strip().startswith("sha256:"), (
        "The chart test requires an existing local image and does not pull one"
    )
    monkeypatch.setattr(config, "_ENV_FILE", tmp_path / "unused.env")
    monkeypatch.delenv("LLM_MAX_TOTAL_TOKENS_PER_REQUEST", raising=False)
    monkeypatch.delenv("LLM_MAX_CALLS_PER_REQUEST", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "host-only-chart-secret-sentinel")
    expected = expected_paid_series()
    field, amount = next(iter(expected.items()))

    def final_answer(messages, tools):
        if "error" in tool_result(messages, CALL_ID):
            return structured_turn("ساخت نمودار انجام نشد.")
        return structured_turn("نمودار آماده است و مقدار نمونه {{amount}} تومان است.",
                               amount=numeric_slot(amount, CALL_ID, field, unit="scalar"))

    llm = ScriptedLLM(tool_turn(call("execute_python", {"code": CHART_CODE}, CALL_ID)), final_answer)
    service = create_app(
        data_dir=DATA, token_map_path=ROOT / "demo_tokens.json.example", schema_path=DATA / "schema.md",
        state_db_path=tmp_path / "state.db", artifact_root=tmp_path / "artifacts",
        docker_image=local.stdout.strip(), llm_client=llm,
    )
    sandbox = service.state.sandbox
    assert sandbox.limits.timeout_seconds == 20
    execute = DockerPythonSandbox.execute
    observed = []

    def record_execution(instance, code, tenant):
        assert code == CHART_CODE and tenant.display_label == "Alpha"
        result = execute(instance, code, tenant)
        observed.append(result)
        return result

    monkeypatch.setattr(DockerPythonSandbox, "execute", record_execution)
    owner = {"X-Demo-Token": "demo-alpha"}
    with TestClient(service) as client:
        response = client.post("/chat", headers=owner, json={"question": QUESTION})
        execution, = observed
        assert execution.exit_code == 0, execution.stderr
        assert execution.stderr == ""
        assert json.loads(execution.stdout) == {"metrics": expected}
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == f"نمودار آماده است و مقدار نمونه {amount} تومان است."
        assert [(entry["tool_name"], entry["status"]) for entry in body["trace"]] == [("execute_python", "success")]
        evidence, = manifest(llm.calls[-1]["messages"])["tools"]
        assert evidence["evidence_ref"] == CALL_ID
        assert {entry["evidence_field"] for entry in evidence["fields"]} == expected.keys()
        artifact, = body["artifacts"]
        produced, = execution.artifacts
        assert artifact["artifact_id"] == produced.artifact_id
        path = f"/artifacts/{produced.artifact_id}"
        served = client.get(path, headers=owner)
        assert served.status_code == 200 and served.headers["content-type"] == "image/png"
        assert _valid_png(served.content)
        assert served.content == (sandbox.artifact_root / f"{produced.artifact_id}.png").read_bytes()
        assert client.get(path, headers={"X-Demo-Token": "demo-beta"}).status_code == 404
        assert client.get(path).status_code == 401
