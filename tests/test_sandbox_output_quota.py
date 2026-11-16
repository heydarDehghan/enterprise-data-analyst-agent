"""Phase 4-B.2-A RED contract for bounded sandbox output storage.

Acceptance criteria:
* /output is a size-bounded container filesystem, not a writable host bind.
* A 17 MiB probe hits a hard 16 MiB limit during execution; the probe is
  removed before the container exits, so the current RED run writes at most
  17 MiB temporarily and never attempts host disk exhaustion.
* A ninth tiny file is rejected during execution, not merely at collection.
* Valid PNG collection, timeout/error cleanup, and earlier security limits
  remain intact.
"""

import base64
import json

import pytest

from test_python_sandbox import IMAGE, PNG, docker_ready, tenant


OUTPUT_BYTES = 16 * 1024 * 1024
OUTPUT_FILES = 8


def test_output_policy_is_bounded_alongside_existing_security_limits(tmp_path):
    from app.sandbox import DockerPythonSandbox

    limits = DockerPythonSandbox(image=IMAGE, artifact_root=tmp_path).limits
    assert limits.output_bytes == OUTPUT_BYTES
    assert limits.output_files == OUTPUT_FILES
    assert limits.timeout_seconds == 20
    assert limits.memory_bytes == 512 * 1024 * 1024
    assert limits.cpus <= 1
    assert limits.pids <= 64


def test_real_container_output_is_not_a_writable_host_bind(docker_ready, tmp_path):
    from app.sandbox import DockerPythonSandbox

    code = """
import json, os
from pathlib import Path
mount = next(line for line in Path('/proc/self/mountinfo').read_text().splitlines()
             if line.split()[4] == '/output')
filesystem = mount.split(' - ', 1)[1].split()[0]
stats = os.statvfs('/output')
print(json.dumps({'filesystem': filesystem,
                  'capacity': stats.f_blocks * stats.f_frsize}))
"""
    result = DockerPythonSandbox(image=IMAGE, artifact_root=tmp_path).execute(code, tenant())
    observed = json.loads(result.stdout)

    assert result.exit_code == 0
    assert observed["filesystem"] == "tmpfs"
    assert observed["capacity"] <= OUTPUT_BYTES
    assert result.artifacts == []


def test_real_container_hits_hard_total_byte_limit_during_execution(docker_ready, tmp_path):
    from app.sandbox import DockerPythonSandbox

    code = """
import errno, json
from pathlib import Path
probe = Path('/output/quota-probe.bin')
block = b'x' * (1024 * 1024)
written = 0
quota_hit = False
with probe.open('wb', buffering=0) as handle:
    for _ in range(17):  # bounded: at most 17 MiB even in the RED implementation
        try:
            handle.write(block)
            written += len(block)
        except OSError as exc:
            quota_hit = exc.errno in (errno.ENOSPC, errno.EDQUOT, errno.EFBIG)
            break
size = probe.stat().st_size
probe.unlink()
print(json.dumps({'quota_hit': quota_hit, 'written': written, 'size': size}))
"""
    result = DockerPythonSandbox(image=IMAGE, artifact_root=tmp_path).execute(code, tenant())
    observed = json.loads(result.stdout)

    assert result.exit_code == 0
    assert observed["quota_hit"] is True
    assert observed["written"] <= OUTPUT_BYTES
    assert observed["size"] <= OUTPUT_BYTES
    assert result.artifacts == []
    assert list(tmp_path.iterdir()) == []


def test_real_container_rejects_ninth_file_during_execution(docker_ready, tmp_path):
    from app.sandbox import DockerPythonSandbox

    code = """
import errno, json
from pathlib import Path
root = Path('/output')
created = 0
quota_hit = False
for number in range(9):  # bounded: at most nine one-byte files in RED
    try:
        (root / f'probe-{number}.tmp').write_bytes(b'x')
        created += 1
    except OSError as exc:
        quota_hit = exc.errno in (errno.ENOSPC, errno.EDQUOT)
        break
for path in root.iterdir():
    path.unlink()
print(json.dumps({'created': created, 'quota_hit': quota_hit}))
"""
    result = DockerPythonSandbox(image=IMAGE, artifact_root=tmp_path).execute(code, tenant())
    observed = json.loads(result.stdout)

    assert result.exit_code == 0
    assert observed == {"created": OUTPUT_FILES, "quota_hit": True}
    assert result.artifacts == []
    assert list(tmp_path.iterdir()) == []


def test_valid_png_is_still_collected_after_bounded_execution(docker_ready, tmp_path):
    from app.sandbox import DockerPythonSandbox

    code = (
        "import base64\nfrom pathlib import Path\n"
        f"Path('/output/chart.png').write_bytes(base64.b64decode('{PNG}'))\n"
    )
    result = DockerPythonSandbox(image=IMAGE, artifact_root=tmp_path).execute(code, tenant())

    assert result.exit_code == 0
    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert artifact.media_type == "image/png"
    assert (tmp_path / f"{artifact.artifact_id}.png").read_bytes() == base64.b64decode(PNG)
    assert sorted(path.name for path in tmp_path.iterdir()) == [f"{artifact.artifact_id}.png"]


def test_invalid_output_is_cleaned_after_execution_error(docker_ready, tmp_path):
    from app.sandbox import DockerPythonSandbox, InvalidArtifactError

    code = (
        "from pathlib import Path\n"
        "Path('/output/partial.png').write_text('incomplete')\n"
        "raise ValueError('analysis failed')\n"
    )
    with pytest.raises(InvalidArtifactError):
        DockerPythonSandbox(image=IMAGE, artifact_root=tmp_path).execute(code, tenant())
    assert list(tmp_path.iterdir()) == []


def test_partial_output_is_cleaned_after_timeout(docker_ready, tmp_path):
    from app.sandbox import DockerPythonSandbox, SandboxLimits, SandboxTimeoutError

    instance = DockerPythonSandbox(
        image=IMAGE, artifact_root=tmp_path, limits=SandboxLimits(timeout_seconds=5),
    )
    code = (
        "from pathlib import Path\n"
        "Path('/output/partial.png').write_bytes(b'partial')\n"
        "while True: pass\n"
    )
    with pytest.raises(SandboxTimeoutError):
        instance.execute(code, tenant())
    assert list(tmp_path.iterdir()) == []
