"""Phase 3-A contract for Docker-only generated Python execution.

Container tests require a running Docker daemon and an already present image.
They never pull images or execute supplied Python in the pytest process.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from app.secure_data import SecureDataAccess


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TOKEN_MAP = json.loads((ROOT / "demo_tokens.json.example").read_text(encoding="utf-8"))
IMAGE = "python:3.13-slim"
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"


@pytest.fixture
def docker_ready():
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI unavailable; real isolation not verified")
    daemon = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True, text=True, timeout=5, check=False,
    )
    if daemon.returncode:
        pytest.skip("Docker daemon unavailable; real isolation not verified")
    image = subprocess.run(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
        capture_output=True, text=True, timeout=5, check=False,
    )
    if image.returncode:
        pytest.skip("Sandbox image unavailable locally; no automatic pull")


def sandbox(artifact_root):
    from app.sandbox import DockerPythonSandbox

    return DockerPythonSandbox(image=IMAGE, artifact_root=artifact_root)


def tenant(token="demo-alpha"):
    return SecureDataAccess(data_dir=DATA, token_map=TOKEN_MAP).authenticate(token)


def test_default_sandbox_policy_is_bounded(tmp_path):
    instance = sandbox(tmp_path)
    assert instance.limits.timeout_seconds == 20
    assert instance.limits.memory_bytes == 512 * 1024 * 1024
    assert instance.limits.cpus <= 1
    assert instance.limits.pids <= 64


def test_docker_executes_python_and_returns_real_output(docker_ready, tmp_path):
    instance = sandbox(tmp_path)
    result = instance.execute("print(6 * 7)", tenant())

    assert result.exit_code == 0
    assert result.stdout.strip() == "42"
    assert result.stderr == ""
    assert result.artifacts == []


@pytest.mark.parametrize(
    ("token", "expected_prefix", "expected_revenue", "other_prefix"),
    (
        ("demo-alpha", "A", 53_250_000, "B"),
        ("demo-beta", "B", 106_500_000, "A"),
    ),
)
def test_container_receives_only_paid_tenant_exports(
    docker_ready, tmp_path, token, expected_prefix, expected_revenue, other_prefix
):
    code = """
import csv, json, os
from pathlib import Path
names = sorted(os.listdir('/data'))
with open('/data/paid_orders.csv', newline='', encoding='utf-8') as handle:
    rows = list(csv.DictReader(handle))
with open('/data/customers.csv', newline='', encoding='utf-8') as handle:
    customers = list(csv.DictReader(handle))
with open('/data/tickets.csv', newline='', encoding='utf-8') as handle:
    tickets = list(csv.DictReader(handle))
total = sum(int(r['amount_toman']) for r in rows if r['order_date'].startswith('2026-09'))
print(json.dumps({'names': names, 'paid': all(r['status'] == 'paid' for r in rows),
  'ids': [r['customer_id'] for r in rows + customers + tickets], 'revenue': total}))
"""
    before = (DATA / f"{token.removeprefix('demo-')}.db").read_bytes()
    result = sandbox(tmp_path).execute(code, tenant(token))
    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert payload["names"] == ["customers.csv", "paid_orders.csv", "tickets.csv"]
    assert payload["paid"] is True
    assert payload["revenue"] == expected_revenue
    assert payload["ids"] and all(x.startswith(expected_prefix) for x in payload["ids"])
    assert not any(x.startswith(other_prefix) for x in payload["ids"])
    assert (DATA / f"{token.removeprefix('demo-')}.db").read_bytes() == before


def test_container_has_no_network_host_secret_or_raw_database(docker_ready, tmp_path, monkeypatch):
    monkeypatch.setenv("KAMIDSO_SANDBOX_SECRET_PROBE", "host-only-sentinel")
    code = """
import json, os, socket
from pathlib import Path
sock = socket.socket()
sock.settimeout(1)
try:
    sock.connect(('1.1.1.1', 443))
    network = True
except OSError:
    network = False
finally:
    sock.close()
print(json.dumps({'network': network,
    'secret': os.environ.get('KAMIDSO_SANDBOX_SECRET_PROBE'),
    'db_files': list(Path('/data').glob('*.db')),
    'host_mount': Path('/host').exists(),
    'other_tenant': Path('/data/beta_tickets.csv').exists()}, default=str))
"""
    result = sandbox(tmp_path).execute(code, tenant("demo-alpha"))
    payload = json.loads(result.stdout)

    assert result.exit_code == 0
    assert payload == {
        "network": False, "secret": None, "db_files": [],
        "host_mount": False, "other_tenant": False,
    }


def test_container_cgroup_limits_are_enforced(docker_ready, tmp_path):
    code = """
import json
from pathlib import Path
def read(name):
    return Path('/sys/fs/cgroup', name).read_text().strip()
print(json.dumps({'memory': read('memory.max'), 'cpu': read('cpu.max'),
                  'pids': read('pids.max')}))
"""
    result = sandbox(tmp_path).execute(code, tenant())
    limits = json.loads(result.stdout)

    assert result.exit_code == 0
    assert int(limits["memory"]) <= 512 * 1024 * 1024
    quota, period = map(int, limits["cpu"].split())
    assert quota / period <= 1
    assert int(limits["pids"]) <= 64


def test_container_timeout_and_nonzero_error_are_reported(docker_ready, tmp_path):
    from app.sandbox import SandboxTimeoutError

    instance = sandbox(tmp_path)
    with pytest.raises(SandboxTimeoutError):
        instance.execute("while True: pass", tenant())
    failed = instance.execute("raise ValueError('analysis failed')", tenant())
    assert failed.exit_code != 0
    assert "ValueError" in failed.stderr


def test_valid_png_is_returned_and_invalid_artifact_is_rejected(docker_ready, tmp_path):
    from app.sandbox import InvalidArtifactError

    instance = sandbox(tmp_path)
    result = instance.execute(
        "import base64\nfrom pathlib import Path\n"
        f"Path('/output/plot.png').write_bytes(base64.b64decode('{PNG}'))\n",
        tenant(),
    )
    assert result.exit_code == 0
    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert artifact.media_type == "image/png"
    assert (tmp_path / f"{artifact.artifact_id}.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )

    with pytest.raises(InvalidArtifactError):
        instance.execute(
            "from pathlib import Path\nPath('/output/plot.png').write_text('not png')",
            tenant(),
        )
