"""Bounded server attestations for tool results already stored in turn history."""

import json
from dataclasses import asdict
from hashlib import sha256

from app.month_scope import MonthScope
from app.schemas import ToolTrace
from app.structured_grounding import EvidenceScope, FieldKind, RecordedToolResult, SqlField


MAX_EVIDENCE_TURNS = 32
_MAX_RESULT_BYTES = 64 * 1024
_MAX_PROOF_BYTES = 32 * 1024
_ELIGIBLE_TOOLS = {"query_sql", "get_tickets", "execute_python"}


def _tool_messages(messages: list[dict]) -> list[tuple[int, dict]]:
    return [(index, message) for index, message in enumerate(messages)
            if message.get("role") == "tool"]


def _result_digest(scope: EvidenceScope, index: int, message: dict) -> str | None:
    content = message.get("content")
    if type(content) is not str or len(content) > _MAX_RESULT_BYTES:
        return None
    raw = content.encode("utf-8")
    if len(raw) > _MAX_RESULT_BYTES:
        return None
    identity = json.dumps([scope.tenant_key, scope.conversation_id, index,
                           message.get("tool_call_id")], separators=(",", ":"))
    digest = sha256(identity.encode("utf-8"))
    digest.update(raw)
    return digest.hexdigest()


def encode_turn_evidence(
    scope: EvidenceScope,
    records: list[RecordedToolResult],
    messages: list[dict],
    trace: list[ToolTrace],
) -> str | None:
    """Attest only server-provided successes matched to their exact saved result."""
    tool_messages = _tool_messages(messages)
    if len(records) != len(trace) or len(records) != len(tool_messages) or len(records) > 8:
        return None
    proofs = []
    budget = _MAX_PROOF_BYTES - 2  # outer JSON array
    for step, (record, (index, message), execution) in enumerate(
        zip(records, tool_messages, trace), start=1,
    ):
        if (record.scope != scope or record.tool_name not in _ELIGIBLE_TOOLS
                or record.succeeded is not True or "error" in record.result
                or execution.status != "success" or execution.step != step
                or execution.tool_name != record.tool_name
                or message.get("tool_call_id") != record.call_id):
            continue
        if record.tool_name == "query_sql" and (
            not record.sql_provenance
            or record.sql_fields != {name: source.kind for name, source in record.sql_provenance.items()}
        ):
            continue
        digest = _result_digest(scope, index, message)
        if digest is None or message["content"] != json.dumps(record.result, ensure_ascii=False):
            continue
        entry = json.dumps({
            "step": step,
            "call_id": record.call_id,
            "tool": record.tool_name,
            "digest": digest,
            "window": asdict(record.sql_month) if record.sql_month is not None else None,
            "fields": {
                name: {"kind": source.kind.name, "table": source.table,
                       "column": source.column, "calculation": source.calculation}
                for name, source in record.sql_provenance.items()
            },
        }, separators=(",", ":"))
        if len(entry) + bool(proofs) <= budget:
            budget -= len(entry) + bool(proofs)
            proofs.append(entry)
    return "[" + ",".join(proofs) + "]" if proofs else None


def recover_turn_evidence(
    scope: EvidenceScope,
    proof: str | None,
    messages: list[dict],
    trace: list[ToolTrace],
) -> list[RecordedToolResult]:
    """Recover a result only when its server proof and execution trace agree."""
    if not proof or len(proof) > _MAX_PROOF_BYTES:
        return []
    try:
        entries = json.loads(proof)
    except (ValueError, RecursionError):
        return []
    if not isinstance(entries, list) or len(entries) > 8:
        return []
    tool_messages = _tool_messages(messages)
    recovered = []
    for entry in entries:
        try:
            step = entry["step"]
            if type(step) is not int or not 1 <= step <= min(len(trace), len(tool_messages)):
                continue
            index, message = tool_messages[step - 1]
            execution = trace[step - 1]
            name, call_id = entry["tool"], entry["call_id"]
            if (name not in _ELIGIBLE_TOOLS or execution.status != "success"
                    or execution.step != step or execution.tool_name != name
                    or type(call_id) is not str or not call_id.strip()
                    or call_id != message.get("tool_call_id") or not entry["digest"]
                    or entry["digest"] != _result_digest(scope, index, message)):
                continue
            result = json.loads(message["content"])
            if not isinstance(result, dict) or "error" in result:
                continue
            sources = {
                field: SqlField(FieldKind[source["kind"]], source["table"],
                                source["column"], source["calculation"])
                for field, source in entry["fields"].items()
            }
            window = MonthScope(**entry["window"]) if entry["window"] is not None else None
        except (KeyError, TypeError, ValueError, AttributeError, RecursionError):
            continue
        recovered.append(RecordedToolResult(
            scope, call_id, name, True, result, {field: source.kind for field, source in sources.items()},
            window, sources,
        ))
    return recovered
