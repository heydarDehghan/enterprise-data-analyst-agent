"""Bounded model-facing metadata for the existing structured slot contract."""

import json
import re
from collections.abc import Iterable

from app.structured_grounding import (
    EvidenceAddress, GroundingContext, RecordedToolResult, citable_fields, evidence_registry,
)


_EVIDENCE_PREFIX = "KAMIDSO_EVIDENCE_V1 "
_MAX_BLOCK_BYTES = 4_096
_MAX_FIELDS = 24
_MAX_FIELDS_PER_TOOL = 12
_TOOL_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_FIELD_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_TENANT_LABEL = re.compile(r"[A-Za-z0-9_-]{1,64}")
_EXAMPLE_ID = "<actual_tool_call_id>"
_CONTEXT_SOURCES = {"tenant_context", "user_request"}


def _compact(data: object) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"))


def _tool_slot(kind: str, value: str, field: str, **extra: object) -> dict[str, object]:
    return {"type": kind, "value": value, "evidence_ref": _EXAMPLE_ID, "evidence_field": field, **extra}


SLOT_EXAMPLES = "KAMIDSO_SLOT_EXAMPLES_V1 " + _compact({
    "tenant_label": {"source": "tenant_context", "value": "ExampleTenant"},
    "user_period": {"source": "user_request", "value": "2030-04"},
    "period_start": {"source": "user_request", "type": "date", "field": "period_start", "value": "2030-04-01"},
    "period_end": {"source": "user_request", "type": "date", "field": "period_end", "value": "2030-04-30"},
    "toman": _tool_slot("numeric", "1250", "paid_amount", unit="toman"),
    "count": _tool_slot("numeric", "4", "ticket_count", unit="count"),
    "percent": _tool_slot("numeric", "12.50", "order_share", unit="percent"),
    "scalar": _tool_slot("numeric", "3.25", "index", unit="scalar"),
    "identifier": _tool_slot("identifier", "AT042", "ticket_id", row=0),
    "date": _tool_slot("date", "2030-04-07", "created_at", row=0),
})


def _context_fields(context: GroundingContext) -> dict[str, dict[str, str]]:
    fields: dict[str, dict[str, str]] = {}
    if context.tenant_label and _TENANT_LABEL.fullmatch(context.tenant_label):
        fields["tenant_context"] = {"value": context.tenant_label}
    if context.month is not None:
        month = context.month
        fields["user_request"] = {"period_start": month.first, "period_end": month.last}
        if month.gregorian_month is not None:
            fields["user_request"]["period"] = month.gregorian_month
    return fields


def evidence_block(context: GroundingContext, records: Iterable[RecordedToolResult]) -> str:
    """Describe citable fields, favoring recent records within the prompt caps.

    The field list is a bounded guide. The full grounding registry stays intact.
    Tool data stays in the original tool messages, except the derived count.
    """
    tools: list[dict[str, object]] = []
    periods: dict[str, dict[str, str]] = {}
    manifest = {"context": _context_fields(context), "tools": tools,
                "verified_periods": periods, "truncated": False}
    field_count = 0
    for record in reversed(evidence_registry(context.scope, records).values()):
        if record is None or not _TOOL_ID.fullmatch(record.call_id) or record.call_id in _CONTEXT_SOURCES:
            continue
        fields: list[dict[str, object]] = []
        for name, row, kind, value in citable_fields(record, context.month):
            # Names and IDs can originate in model-generated SQL or Python.
            # Only short opaque identifiers enter the system message.
            if not _FIELD_NAME.fullmatch(name):
                continue
            if field_count == _MAX_FIELDS or len(fields) == _MAX_FIELDS_PER_TOOL:
                manifest["truncated"] = True
                break
            if not fields:
                tools.append({"evidence_ref": record.call_id, "tool_name": record.tool_name, "fields": fields})
                if record.tool_name == "query_sql" and record.sql_month is not None:
                    periods[record.call_id] = {"start": record.sql_month.first, "end": record.sql_month.last}
            field_type, unit = kind.value
            descriptor: dict[str, object] = {"evidence_field": name, "type": field_type}
            if unit is not None:
                descriptor["unit"] = unit
            if row is not None:
                descriptor["row"] = row
            if record.tool_name == "get_tickets" and name == "ticket_count":
                descriptor["value"] = value
            fields.append(descriptor)
            if len(_EVIDENCE_PREFIX) + len(_compact(manifest)) > _MAX_BLOCK_BYTES:
                fields.pop()
                if not fields:
                    tools.pop()
                    periods.pop(record.call_id, None)
                manifest["truncated"] = True
                break
            field_count += 1
        if field_count == _MAX_FIELDS and manifest["truncated"]:
            break
    # _compact emits ASCII, so character and byte lengths are equal.
    return _EVIDENCE_PREFIX + _compact(manifest)


def advertises_address(block: str, address: EvidenceAddress) -> bool:
    """Check an address against the bounded manifest actually sent to the model."""
    manifest = json.loads(block.removeprefix(_EVIDENCE_PREFIX))
    return any(
        (field["evidence_field"], field.get("row"), (field["type"], field.get("unit")))
        == (address.field, address.row, address.kind.value)
        for tool in manifest["tools"] if tool["evidence_ref"] == address.reference
        for field in tool["fields"]
    )
