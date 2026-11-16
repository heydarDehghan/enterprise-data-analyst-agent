"""Run generated analysis code in a disposable, tenant-scoped Docker container."""

import base64
import csv
import json
import struct
import subprocess
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from app.schemas import ArtifactMetadata


_CUSTOMERS = ("customer_id", "name", "segment", "created_at")
_PAID_ORDERS = ("order_id", "customer_id", "order_date", "amount_toman", "status", "category")
_TICKETS = ("ticket_id", "customer_id", "created_at", "category", "description", "is_untrusted")
_MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
_MAX_ARTIFACTS = 8
_OUTPUT_BYTES = 16 * 1024 * 1024
_MAX_OUTPUT_BYTES = 1024 * 1024
_MAX_CODE_BYTES = 1024 * 1024

# Docker cannot copy files from a tmpfs mount with `docker cp`. Read the
# bounded output from inside the running container, before removing it.
_READ_OUTPUT = """
import base64, json, os, stat
from pathlib import Path

paths = list(Path('/output').iterdir())
if len(paths) > 8:
    raise ValueError('Too many output artifacts')
items = []
for path in paths:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.suffix.lower() != '.png':
        raise ValueError('Only regular PNG artifacts are allowed')
    if info.st_size > 10 * 1024 * 1024:
        raise ValueError('PNG artifact exceeds size limit')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as handle:
        payload = handle.read(10 * 1024 * 1024 + 1)
    if len(payload) > 10 * 1024 * 1024:
        raise ValueError('PNG artifact exceeds size limit')
    items.append({'title': path.stem,
                  'payload': base64.b64encode(payload).decode('ascii')})
print(json.dumps(items))
"""


class SandboxTimeoutError(RuntimeError):
    """The generated program exceeded its wall clock budget."""


class InvalidArtifactError(ValueError):
    """The generated program produced an unsafe or invalid artifact."""


class SandboxOutputLimitError(RuntimeError):
    """The generated program exceeded its host output budget."""


@dataclass(frozen=True)
class SandboxLimits:
    timeout_seconds: int = 20
    memory_bytes: int = 512 * 1024 * 1024
    cpus: float = 1.0
    pids: int = 64
    output_bytes: int = _OUTPUT_BYTES
    output_files: int = _MAX_ARTIFACTS


@dataclass(frozen=True)
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    artifacts: list[ArtifactMetadata] = field(default_factory=list)


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _valid_png(data: bytes) -> bool:
    """Check PNG framing, required chunks, dimensions, and chunk CRCs."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    offset = 8
    chunks: list[bytes] = []
    while offset + 12 <= len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        end = offset + 12 + length
        if end > len(data):
            return False
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        if zlib.crc32(kind + payload) != crc:
            return False
        chunks.append(kind)
        if len(chunks) == 1:
            if kind != b"IHDR" or length != 13:
                return False
            width, height = struct.unpack_from(">II", payload)
            if width < 1 or height < 1 or width > 10_000 or height > 10_000:
                return False
        offset = end
        if kind == b"IEND":
            return length == 0 and offset == len(data) and b"IDAT" in chunks
    return False


class DockerPythonSandbox:
    def __init__(
        self,
        *,
        image: str,
        artifact_root: Path,
        limits: SandboxLimits | None = None,
    ) -> None:
        self.image = image
        self.artifact_root = Path(artifact_root).resolve()
        self.limits = limits or SandboxLimits()
        if (
            self.limits.timeout_seconds < 1 or self.limits.timeout_seconds > 20
            or self.limits.memory_bytes < 1 or self.limits.memory_bytes > 512 * 1024 * 1024
            or self.limits.cpus <= 0 or self.limits.cpus > 1
            or self.limits.pids < 1 or self.limits.pids > 64
            or self.limits.output_bytes < 1 or self.limits.output_bytes > _OUTPUT_BYTES
            or self.limits.output_files < 1 or self.limits.output_files > _MAX_ARTIFACTS
        ):
            raise ValueError("Sandbox limits exceed the approved policy")
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    def execute(self, code: str, tenant: object) -> SandboxResult:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Python code must be nonblank text")
        if len(code.encode("utf-8")) > _MAX_CODE_BYTES:
            raise ValueError("Python code exceeds size limit")

        with tempfile.TemporaryDirectory(prefix="sandbox-", dir=self.artifact_root) as temporary:
            root = Path(temporary)
            data = root / "data"
            work = root / "work"
            for directory in (data, work):
                directory.mkdir()

            # Only the agent-authorized paid snapshot and tenant tickets cross
            # the container boundary. Neither raw databases nor token maps do.
            _write_csv(data / "customers.csv", _CUSTOMERS, tenant.agent_query(
                "SELECT customer_id, name, segment, created_at FROM customers"
            ))
            _write_csv(data / "paid_orders.csv", _PAID_ORDERS, tenant.agent_query(
                "SELECT order_id, customer_id, order_date, amount_toman, status, category "
                "FROM paid_orders"
            ))
            _write_csv(data / "tickets.csv", _TICKETS, tenant.tickets())
            (work / "code.py").write_text(code, encoding="utf-8")

            name = f"kamidso-{uuid4().hex}"
            # The output tmpfs is charged to the container memory cgroup. Its
            # root consumes one inode, leaving exactly output_files entries.
            command = [
                "docker", "run", "-d", "--name", name,
                "--network", "none", "--ipc", "none", "--read-only",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--user", "65534:65534", "--memory", str(self.limits.memory_bytes),
                "--memory-swap", str(self.limits.memory_bytes),
                "--cpus", str(self.limits.cpus), "--pids-limit", str(self.limits.pids),
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
                "--tmpfs", (
                    f"/output:rw,noexec,nosuid,size={self.limits.output_bytes},"
                    f"nr_inodes={self.limits.output_files + 1},"
                    "uid=65534,gid=65534,mode=700"
                ),
                "--mount", f"type=bind,source={data},target=/data,readonly",
                "--mount", f"type=bind,source={work},target=/work,readonly",
                self.image, "sleep", "infinity",
            ]
            stdout_path = root / "stdout"
            stderr_path = root / "stderr"
            try:
                subprocess.run(command, capture_output=True, check=True, timeout=20)
                with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                    process = subprocess.Popen(
                        ["docker", "exec", name, "python", "-I", "-B", "/work/code.py"],
                        stdout=stdout, stderr=stderr,
                    )
                    deadline = time.monotonic() + self.limits.timeout_seconds
                    while process.poll() is None:
                        if (stdout_path.stat().st_size > _MAX_OUTPUT_BYTES
                                or stderr_path.stat().st_size > _MAX_OUTPUT_BYTES):
                            self._stop_container(name, process)
                            raise SandboxOutputLimitError("Python output exceeds size limit")
                        if time.monotonic() >= deadline:
                            self._stop_container(name, process)
                            raise SandboxTimeoutError("Python execution timed out")
                        time.sleep(0.05)
                    exit_code = process.returncode

                if (stdout_path.stat().st_size > _MAX_OUTPUT_BYTES
                        or stderr_path.stat().st_size > _MAX_OUTPUT_BYTES):
                    raise SandboxOutputLimitError("Python output exceeds size limit")

                artifacts = self._collect_artifacts(name)
                return SandboxResult(
                    stdout=stdout_path.read_text(encoding="utf-8", errors="replace"),
                    stderr=stderr_path.read_text(encoding="utf-8", errors="replace"),
                    exit_code=exit_code,
                    artifacts=artifacts,
                )
            finally:
                # A timed-out `docker run` can still have created the named
                # container, so cleanup must not depend on CLI success.
                subprocess.run(["docker", "rm", "-f", name], capture_output=True,
                               check=False, timeout=10)

    @staticmethod
    def _stop_container(name: str, process: subprocess.Popen) -> None:
        process.terminate()
        try:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True,
                           check=False, timeout=10)
        finally:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def _collect_artifacts(self, container: str) -> list[ArtifactMetadata]:
        result = subprocess.run(
            ["docker", "exec", container, "python", "-I", "-B", "-c", _READ_OUTPUT],
            capture_output=True, check=False, timeout=20,
        )
        if result.returncode:
            raise InvalidArtifactError("Output contains invalid artifacts")
        try:
            items = json.loads(result.stdout)
            if not isinstance(items, list) or len(items) > self.limits.output_files:
                raise ValueError("Too many output artifacts")
            validated = []
            total_bytes = 0
            for item in items:
                title = item["title"]
                if not isinstance(title, str):
                    raise ValueError("Invalid artifact title")
                payload = base64.b64decode(item["payload"], validate=True)
                total_bytes += len(payload)
                if (len(payload) > _MAX_ARTIFACT_BYTES
                        or total_bytes > self.limits.output_bytes
                        or not _valid_png(payload)):
                    raise ValueError("Invalid PNG artifact")
                validated.append((title, payload))
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidArtifactError("Output contains invalid artifacts") from exc

        artifacts: list[ArtifactMetadata] = []
        created: list[Path] = []
        try:
            for title, payload in validated:
                artifact_id = uuid4().hex
                target = self.artifact_root / f"{artifact_id}.png"
                created.append(target)
                target.write_bytes(payload)
                artifacts.append(ArtifactMetadata(
                    artifact_id=artifact_id, kind="image", title=title,
                    media_type="image/png",
                ))
        except Exception:
            for target in created:
                target.unlink(missing_ok=True)
            raise
        return artifacts
