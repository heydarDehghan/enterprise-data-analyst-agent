"""Opt-in, test-scoped recording of approved live provider runs."""

import base64
import json
import os
import re
import stat
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import httpx

from app.config import load_llm_config
from app.observations import Observer, observer


ALPHA_NODE = (
    "live_tests/test_agent_end_to_end_live.py::"
    "test_live_chat_calls_paid_sql_and_tenant_tickets[demo-alpha-alpha.db-alpha_tickets.csv-A001-B]"
)
BETA_NODE = (
    "live_tests/test_agent_end_to_end_live.py::"
    "test_live_chat_calls_paid_sql_and_tenant_tickets[demo-beta-beta.db-beta_tickets.csv-B001-A]"
)
PYTHON_PNG_NODE = "live_tests/test_agent_end_to_end_live.py::test_live_python_png_and_conversation_reuse"
APPROVED_DIAGNOSTIC_NODES = frozenset({ALPHA_NODE, BETA_NODE, PYTHON_PNG_NODE})
CAPTURE_PATH = "KAMIDSO_LLM_DIAGNOSTICS_PATH"
_MAX_BYTES = 16 * 1024 * 1024
_REDACTED = "[REDACTED]"
_agent_observation_listener: ContextVar[Observer | None] = ContextVar(
    "live_test_agent_observation_listener", default=None,
)


class DiagnosticCaptureError(RuntimeError):
    """The diagnostic output could not be captured completely."""


def capture_requested():
    return (os.environ.get("KAMIDSO_RUN_LIVE_AGENT_TESTS") == "1"
            and os.environ.get("KAMIDSO_CAPTURE_LLM_DIAGNOSTICS") == "1")


def _windows_private_fd(path):
    """Create an exclusive file with a protected owner and SYSTEM DACL."""
    import ctypes
    import msvcrt
    from ctypes import wintypes as win

    class SecurityAttributes(ctypes.Structure):
        _fields_ = [("length", win.DWORD), ("descriptor", win.LPVOID), ("inherit", win.BOOL)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    convert = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = (win.LPCWSTR, win.DWORD, ctypes.POINTER(win.LPVOID), ctypes.POINTER(win.DWORD))
    convert.restype = win.BOOL
    create = kernel.CreateFileW
    create.argtypes = (win.LPCWSTR, win.DWORD, win.DWORD, ctypes.POINTER(SecurityAttributes),
                       win.DWORD, win.DWORD, win.HANDLE)
    create.restype = win.HANDLE
    kernel.LocalFree.argtypes = (win.LPVOID,)
    kernel.LocalFree.restype = win.LPVOID
    kernel.CloseHandle.argtypes = (win.HANDLE,)
    kernel.CloseHandle.restype = win.BOOL

    descriptor = win.LPVOID()
    if not convert("D:P(A;;FA;;;OW)(A;;FA;;;SY)", 1, ctypes.byref(descriptor), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        attributes = SecurityAttributes(ctypes.sizeof(SecurityAttributes), descriptor, False)
        generic_write, create_new, normal_file = 0x40000000, 1, 0x80
        handle = create(str(path), generic_write, 0, ctypes.byref(attributes), create_new, normal_file, None)
        if handle == win.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.LocalFree(descriptor)
    try:
        return msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)
    except OSError:
        kernel.CloseHandle(handle)
        raise


def _private_output(path):
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Diagnostic capture requires an absolute local path")
    if os.name == "nt":
        import ctypes

        # Exclude network shares, mapped remote drives and Windows device names.
        kernel = ctypes.windll.kernel32
        root = ctypes.c_wchar_p(path.anchor)
        if (not re.fullmatch(r"[A-Za-z]:", path.drive)
                or kernel.GetDriveTypeW(root) not in (2, 3, 6)
                or any(":" in part or part.endswith((" ", ".")) for part in path.parts[1:])):
            raise ValueError("Diagnostic capture requires a regular local path")
        reserved = getattr(os.path, "isreserved", None)
        is_reserved = reserved(path) if reserved is not None else path.is_reserved()
        if is_reserved:
            raise ValueError("Diagnostic capture requires a regular local path")
        filesystem_flags = ctypes.c_uint32()
        persistent_acls = 0x8
        if (not kernel.GetVolumeInformationW(root, None, 0, None, None,
                                             ctypes.byref(filesystem_flags), None, 0)
                or not filesystem_flags.value & persistent_acls):
            raise ValueError("Diagnostic capture requires a filesystem that supports file ACLs")

    try:
        for directory in reversed(path.parents):
            info = directory.lstat()
            if (not stat.S_ISDIR(info.st_mode)
                    or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
                raise ValueError("Diagnostic path must have existing, non-symlink directories")
        if os.name == "nt":
            fd = _windows_private_fd(path)
        else:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags, 0o600)
        return os.fdopen(fd, "wb")
    except OSError:
        raise ValueError("Cannot create an unused diagnostic file in the configured directory") from None


class _Recorder:
    def __init__(self, node_id, api_key):
        self._credential = api_key
        variants = {api_key.encode("utf-8"), json.dumps(api_key)[1:-1].encode("utf-8")}
        self._credential_bytes = re.compile(b"|".join(re.escape(value) for value in sorted(
            variants, key=len, reverse=True
        )))
        self._body_bytes = 0
        self._turn = {}
        self._provider_call_id = None
        self._in_provider = False
        self._length_recovery = None
        self._in_length_recovery = False
        self._format_repair = None
        self._in_format_repair = False
        self._grounding = None
        self._slots = {}
        self._slot = None
        self._errors = []
        self.data = {
            "schema_version": 1, "run_id": uuid4().hex, "test_node_id": node_id,
            "runtime_limits": {}, "provider_calls": [], "tool_executions": [], "grounding": [],
        }

    def _guard(self, operation, action, *args):
        # Record failures outside the request path. Never let a capture failure
        # become a provider retry, tool failure, or HTTP application error.
        try:
            return action(*args)
        except Exception as exc:
            self._errors.append({"operation": operation, "error": type(exc).__name__})
            return None

    def observe(self, event, **values):
        self._guard(event, self._observe, event, values)
        listener = _agent_observation_listener.get()
        if listener is not None:
            self._guard("observation_listener", lambda: listener(event, **values))

    def _observe(self, event, values):
        if event == "turn":
            self._turn = {key: values[key] for key in ("conversation_id", "trace_id")}
            self._length_recovery = None
            self._format_repair = None
            self.data["runtime_limits"] = {
                key: values[key] for key in ("max_calls", "max_total_tokens")
            }
        elif event == "provider_start":
            self._in_provider = True
            self._in_length_recovery = values.get("length_recovery", False)
            self._in_format_repair = values.get("format_repair", False)
            self._provider_call_id = None
            self._provider_index = values["index"]
            if self._in_length_recovery and self._length_recovery is not None:
                self._length_recovery["attempts"] = 1
        elif event == "provider_end":
            self._in_provider = False
            self._in_length_recovery = False
            self._in_format_repair = False
        elif event == "length_completion":
            self._length_recovery = {
                **self._turn, "original_provider_call_id": self._provider_call_id,
                "recovery_provider_call_id": None, "attempts": 0, **values,
            }
            self.data.setdefault("length_recovery", []).append(self._length_recovery)
        elif event == "length_recovery_response" and self._length_recovery is not None:
            self._length_recovery.update(values)
        elif event == "turn_end":
            if self._length_recovery is not None:
                self._length_recovery.update(values)
            if self._format_repair is not None:
                self._format_repair.update(values)
        elif event == "tool":
            self.data["tool_executions"].append({
                **self._turn, "provider_call_id": self._provider_call_id, **values,
            })
        elif event == "grounding_start":
            self._grounding = {
                **self._turn, "provider_call_id": self._provider_call_id, **values,
                "template": None, "slots": None, "slot_decisions": [],
                "rendered_answer": None, "rejection_reason": None,
            }
            self.data["grounding"].append(self._grounding)
            self._slots = {}
            self._slot = None
        elif event == "grounding_parsed":
            self._grounding.update(values)
            if isinstance(values["slots"], dict):
                for name, proposed in values["slots"].items():
                    proposed = proposed if isinstance(proposed, dict) else {}
                    self._slots[name] = {
                        "slot": name, "proposed_value": proposed.get("value"),
                        "evidence_ref": proposed.get("evidence_ref"),
                        "evidence_field": proposed.get("evidence_field"), "row": proposed.get("row"),
                        "proposed_type": proposed.get("type"), "proposed_unit": proposed.get("unit"),
                        "type": None, "unit": None, "authoritative_value": None,
                        "decision": "not_evaluated", "reason_code": None,
                    }
                self._grounding["slot_decisions"] = list(self._slots.values())
        elif event == "slot_start":
            self._slot = self._slots[values["slot"]]
        elif event in ("authority", "slot_end"):
            self._slot.update(values)
        elif event == "grounding_end":
            self._grounding.update(values)
        elif event == "repair":
            self._grounding.setdefault("repair", {}).update(values)
        elif event == "format_repair":
            if self._format_repair is None:
                self._format_repair = {"recovery_provider_call_id": None}
                self._grounding["format_repair"] = self._format_repair
            self._format_repair.update(values)

    def _body(self, raw):
        self._body_bytes += len(raw)
        if self._body_bytes > _MAX_BYTES:
            raise DiagnosticCaptureError("HTTP body capture exceeds the diagnostic size limit")
        raw, redactions = self._credential_bytes.subn(_REDACTED.encode("ascii"), raw)
        try:
            body = {"encoding": "utf-8", "data": raw.decode("utf-8")}
        except UnicodeDecodeError:
            body = {"encoding": "base64", "data": base64.b64encode(raw).decode("ascii")}
        if redactions:
            body["credential_redactions"] = redactions
        return body

    def _begin_attempt(self, request):
        self._provider_call_id = f"{self.data['run_id']}:{len(self.data['provider_calls']) + 1}"
        if self._in_length_recovery and self._length_recovery is not None:
            self._length_recovery["recovery_provider_call_id"] = self._provider_call_id
        if self._in_format_repair and self._format_repair is not None:
            self._format_repair["recovery_provider_call_id"] = self._provider_call_id
        entry = {
            **self._turn, "index": self._provider_index, "call_id": self._provider_call_id,
            "request": {
                "method": request.method, "url": str(request.url),
                "headers": {"authorization": _REDACTED} if "authorization" in request.headers else {},
                "body": None,
            },
            "response": None, "transport_error": None,
        }
        self.data["provider_calls"].append(entry)
        entry["request"]["body"] = self._guard("request_body", lambda: self._body(request.content))
        return entry

    def _response(self, entry, response):
        entry["response"] = {
            "status_code": response.status_code,
            "provider_request_id": response.headers.get("x-request-id"),
            "body": self._guard("response_body", lambda: self._body(response.content)),
        }

    def instrument(self, factory):
        def client_factory(*args, **kwargs):
            client = factory(*args, **kwargs)
            send = client.send

            def observed_send(request, *send_args, **send_kwargs):
                if observer.get() is not self.sink or not self._in_provider:
                    return send(request, *send_args, **send_kwargs)
                entry = self._guard("request", self._begin_attempt, request)
                try:
                    response = send(request, *send_args, **send_kwargs)
                except Exception as exc:
                    if entry is not None:
                        entry["transport_error"] = {
                            "type": type(exc).__name__, "message": self._guard("transport_error", str, exc),
                        }
                    raise
                if entry is not None:
                    self._guard("response", self._response, entry, response)
                return response

            client.send = observed_send
            return client

        return client_factory

    def _write(self, handle, data):
        payload = json.dumps(data, ensure_ascii=False, indent=2, default=self._json_value)
        handle.write(payload.encode("utf-8", errors="backslashreplace") + b"\n")

    def _json_value(self, value):
        if isinstance(value, Decimal):
            return str(value)
        raise TypeError("Unsupported diagnostic value")

    def _redact(self, value):
        if isinstance(value, str):
            return value.replace(self._credential, _REDACTED)
        if isinstance(value, dict):
            return {self._redact(key): self._redact(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        return value

    def finish(self, handle):
        self.data["capture_errors"] = self._errors
        try:
            payload = json.dumps(self._redact(self.data), ensure_ascii=False, indent=2,
                                 default=self._json_value).encode("utf-8", errors="backslashreplace")
        except (ValueError, TypeError, RecursionError) as exc:
            self._errors.append({"operation": "document", "error": type(exc).__name__})
            payload = None
        if payload is not None and len(payload) + 1 <= _MAX_BYTES:
            handle.write(payload + b"\n")
        else:
            if payload is not None:
                self._errors.append({"operation": "document", "error": "DiagnosticSizeLimitExceeded"})
            # Keep a readable failure record instead of a truncated document
            # or one that appears to be a successful capture.
            self._write(handle, self._redact({
                key: [] if key in ("provider_calls", "tool_executions", "grounding") else value
                for key, value in self.data.items()
            }))
        handle.flush()
        if self._errors:
            raise DiagnosticCaptureError("LLM diagnostic capture incomplete; see capture_errors in the local file")


@contextmanager
def collect_agent_observations(listener: Observer):
    """Collect test events in memory alongside any active private recorder."""
    sink = observer.get()
    recorder = getattr(sink, "__self__", None)
    if isinstance(recorder, _Recorder) and sink is recorder.sink:
        # The HTTP capture checks this sink's identity. Register beside it.
        target = _agent_observation_listener
        token = target.set(listener)
    else:
        def forward(event, **values):
            if sink is not None:
                sink(event, **values)
            listener(event, **values)

        target = observer
        token = target.set(forward)
    try:
        yield
    finally:
        target.reset(token)


@contextmanager
def capture_run(*, node_id, monkeypatch):
    if not capture_requested():
        yield
        return
    if node_id not in APPROVED_DIAGNOSTIC_NODES:
        raise ValueError("LLM diagnostic capture requires an approved live test node")
    target = os.environ.get(CAPTURE_PATH)
    if not target:
        raise ValueError("Set KAMIDSO_LLM_DIAGNOSTICS_PATH for diagnostic capture")
    recorder = _Recorder(node_id, load_llm_config().api_key)
    recorder.sink = recorder.observe
    with _private_output(Path(target)) as output, monkeypatch.context() as patch:
        patch.setattr(httpx, "Client", recorder.instrument(httpx.Client))
        token = observer.set(recorder.sink)
        test_failure = None
        try:
            yield
        except BaseException as exc:
            test_failure = exc
            raise
        finally:
            observer.reset(token)
            try:
                recorder.finish(output)
            except (OSError, DiagnosticCaptureError) as exc:
                message = (str(exc) if isinstance(exc, DiagnosticCaptureError)
                           else "Unable to write the local LLM diagnostic file")
                if test_failure is not None:
                    test_failure.add_note(message)
                else:
                    raise DiagnosticCaptureError(message) from None
