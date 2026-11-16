"""Validate a model answer against server-owned evidence and render it as text.

Only the answer JSON is model input. Callers must construct the context from
authentication and conversation ownership, and supply actual execution records.
SQL field metadata must come from verified projection provenance and include only
unambiguous result columns. An alias or the model's proposed type is not evidence.
"""

import json
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum

from app.month_scope import MonthScope
from app.observations import observe


_SLOT_NAME = r"[A-Za-z_][A-Za-z_0-9]*"
_PLACEHOLDER = re.compile(r"\{\{(" + _SLOT_NAME + r")\}\}")
_NUMBER = re.compile(
    r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+|[0-9]{1,3}(?:٬[0-9]{3})+)(?:[.٫][0-9]+)?"
)
_MAX_NUMBER_LENGTH = 128
_MAX_JSON_LENGTH = 65_536


class StructuredGroundingError(ValueError):
    """Rejection details safe for internal status or logging."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class EvidenceScope:
    tenant_key: str
    conversation_id: str

    def __post_init__(self) -> None:
        if any(type(value) is not str or not value.strip()
               for value in (self.tenant_key, self.conversation_id)):
            raise ValueError("Evidence requires an authenticated conversation scope")


@dataclass(frozen=True, slots=True)
class GroundingContext:
    scope: EvidenceScope
    tenant_label: str | None = None
    month: MonthScope | None = None


class FieldKind(Enum):
    TOMAN = ("numeric", "toman")
    COUNT = ("numeric", "count")
    PERCENT = ("numeric", "percent")
    SCALAR = ("numeric", "scalar")
    DATE = ("date", None)
    IDENTIFIER = ("identifier", None)


@dataclass(frozen=True, slots=True)
class SqlField:
    kind: FieldKind
    table: str
    column: str
    calculation: str


@dataclass(frozen=True, slots=True)
class RecordedToolResult:
    """Private server input, never deserialized from a model answer or tool text."""

    scope: EvidenceScope
    call_id: str
    tool_name: str
    succeeded: bool
    result: Mapping[str, object] = field(repr=False)
    # Verified result-column types and, when known, the verified full SQL window.
    sql_fields: Mapping[str, FieldKind] = field(default_factory=dict)
    sql_month: MonthScope | None = None
    sql_provenance: Mapping[str, SqlField] = field(default_factory=dict)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON member")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise ValueError("Nonstandard JSON constant")


def _json_object(raw: object, code: str) -> dict[str, object]:
    if type(raw) is not str or len(raw) > _MAX_JSON_LENGTH:
        raise StructuredGroundingError(code)
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        raise StructuredGroundingError(code) from None
    if not isinstance(value, dict):
        raise StructuredGroundingError(code)
    return value


def _ascii_digits(text: str) -> str:
    return "".join(str(unicodedata.decimal(char)) if char.isdecimal() else char for char in text)


def _decimal(value: object) -> Decimal:
    # Each accepted representation has bounded digits. Avoid Decimal.normalize,
    # which can round a valid value under the active decimal context.
    if type(value) is str and len(value) <= _MAX_NUMBER_LENGTH:
        text = _ascii_digits(value)
        if not _NUMBER.fullmatch(text):
            raise StructuredGroundingError("unsupported_number")
        number = Decimal(text.replace(",", "").replace("٬", "").replace("٫", "."))
    elif type(value) in (int, float, Decimal):
        # SQLite REAL values are already approximate. Preserve the exposed value
        # when converting to Decimal and avoid any further binary float math.
        if type(value) is int and value.bit_length() > _MAX_NUMBER_LENGTH * 4:
            raise StructuredGroundingError("unsupported_number")
        number = Decimal(str(value)) if type(value) is float else Decimal(value)
    else:
        raise StructuredGroundingError("unsupported_number")
    if not number.is_finite():
        raise StructuredGroundingError("unsupported_number")
    parts = number.as_tuple()
    if (len(parts.digits) > _MAX_NUMBER_LENGTH or abs(parts.exponent) > _MAX_NUMBER_LENGTH
            or abs(number.adjusted()) > _MAX_NUMBER_LENGTH):
        raise StructuredGroundingError("unsupported_number")
    return number


def _render_number(number: Decimal) -> str:
    if number.is_zero():
        return "0"
    text = format(number, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _numeric_value(actual: object, kind: FieldKind) -> Decimal:
    number = _decimal(actual)
    if kind is FieldKind.COUNT and (number < 0 or number != number.to_integral_value()):
        raise StructuredGroundingError("unsupported_number")
    return number


def _iso_date(value: object) -> str:
    if type(value) is not str or len(value) != 10:
        raise StructuredGroundingError("unsupported_number")
    text = _ascii_digits(value)
    try:
        parsed = date.fromisoformat(text)
    except ValueError:
        raise StructuredGroundingError("unsupported_number") from None
    if parsed.isoformat() != text:
        raise StructuredGroundingError("unsupported_number")
    return text


def _select_row(rows: object, index: int | None) -> dict[str, object]:
    if not isinstance(rows, list) or not rows:
        raise StructuredGroundingError("unsupported_number")
    if index is None:
        if len(rows) != 1:
            raise StructuredGroundingError("unsupported_number")
        index = 0
    if index >= len(rows) or not isinstance(rows[index], dict):
        raise StructuredGroundingError("unsupported_number")
    return rows[index]


_TICKET_FIELDS = {
    "customer_id": FieldKind.IDENTIFIER,
    "ticket_id": FieldKind.IDENTIFIER,
    "created_at": FieldKind.DATE,
}


def _python_metrics(result: Mapping[str, object]) -> dict[str, object]:
    if type(result.get("exit_code")) is int and result["exit_code"] == 0:
        output = _json_object(result.get("stdout"), "unsupported_number")
        metrics = output.get("metrics")
        if output.keys() == {"metrics"} and isinstance(metrics, dict):
            return metrics
    raise StructuredGroundingError("unsupported_number")


def _tool_field(
    record: RecordedToolResult, name: str, row: int | None, month: MonthScope | None,
) -> tuple[FieldKind, object]:
    result = record.result
    if record.succeeded is not True or not isinstance(result, Mapping) or "error" in result:
        raise StructuredGroundingError("unsupported_number")
    if record.tool_name == "query_sql":
        kind = record.sql_fields.get(name)
        if not isinstance(kind, FieldKind):
            raise StructuredGroundingError("unsupported_number")
        if month is not None and (kind.value[0] == "numeric" or record.sql_month is not None):
            window = record.sql_month
            if window is None or (window.first, window.last) != (month.first, month.last):
                raise StructuredGroundingError("invalid_date_range")
        values = _select_row(result.get("rows"), row)
        if name in values:
            return kind, values[name]
    elif record.tool_name == "get_tickets":
        tickets = result.get("tickets")
        if name == "ticket_count" and row is None and isinstance(tickets, list):
            if all(isinstance(ticket, dict) for ticket in tickets):
                return FieldKind.COUNT, len(tickets)
        elif name in _TICKET_FIELDS:
            values = _select_row(tickets, row)
            if name in values:
                return _TICKET_FIELDS[name], values[name]
    elif record.tool_name == "execute_python" and row is None:
        metrics = _python_metrics(result)
        if type(metrics.get(name)) is str:
            return FieldKind.SCALAR, metrics[name]
    raise StructuredGroundingError("unsupported_number")


def _field_candidates(record: RecordedToolResult) -> Iterator[tuple[str, int | None]]:
    if record.tool_name == "query_sql":
        rows, names = record.result.get("rows"), record.sql_fields
    elif record.tool_name == "get_tickets":
        yield "ticket_count", None
        rows, names = record.result.get("tickets"), _TICKET_FIELDS
    else:
        return
    if isinstance(rows, list):
        for index in range(len(rows)):
            for name in names:
                yield name, index if len(rows) > 1 else None


def citable_fields(
    record: RecordedToolResult, month: MonthScope | None,
) -> Iterator[tuple[str, int | None, FieldKind, object]]:
    """Expose only addresses the grounding resolver can actually validate."""
    if record.succeeded is not True or not isinstance(record.result, Mapping) or "error" in record.result:
        return
    if record.tool_name == "execute_python":
        try:
            metrics = _python_metrics(record.result)
        except StructuredGroundingError:
            return
        for name, value in metrics.items():
            if type(value) is str:
                try:
                    _numeric_value(value, FieldKind.SCALAR)
                except StructuredGroundingError:
                    continue
                yield name, None, FieldKind.SCALAR, value
        return
    for name, row in _field_candidates(record):
        try:
            kind, value = _tool_field(record, name, row, month)
            if kind.value[0] == "numeric":
                _numeric_value(value, kind)
            elif kind is FieldKind.DATE:
                _iso_date(value)
            elif type(value) is not str or not value:
                continue
        except StructuredGroundingError:
            continue
        yield name, row, kind, value


def evidence_registry(
    scope: EvidenceScope, tool_results: Iterable[RecordedToolResult],
) -> dict[str, RecordedToolResult | None]:
    records: dict[str, RecordedToolResult | None] = {}
    for record in tool_results:
        if record.scope == scope:
            # Even an unsuccessful duplicate invalidates the entire reference.
            records[record.call_id] = None if record.call_id in records else record
    return records


def _context_value(slot: dict[str, object], value: str, context: GroundingContext) -> str:
    keys = slot.keys()
    if slot["source"] == "tenant_context" and keys == {"source", "value"}:
        observe("authority", authoritative_value=context.tenant_label, type="text", unit=None)
        if context.tenant_label is not None and value == context.tenant_label:
            return context.tenant_label
    elif slot["source"] == "user_request" and context.month is not None:
        month = context.month
        if keys == {"source", "value"} and month.gregorian_month is not None:
            observe("authority", authoritative_value=month.gregorian_month, type="period", unit=None)
            if _ascii_digits(value) == month.gregorian_month:
                return month.gregorian_month
        if keys == {"source", "value", "type", "field"} and slot["type"] == "date":
            if slot["field"] in ("period_start", "period_end"):
                expected = month.first if slot["field"] == "period_start" else month.last
                observe("authority", authoritative_value=expected, type="date", unit=None)
                if _iso_date(value) == expected:
                    return expected
    raise StructuredGroundingError("unsupported_number")


def _slot_value(
    slot: object, context: GroundingContext, records: Mapping[str, RecordedToolResult | None],
) -> str:
    if not isinstance(slot, dict) or type(slot.get("value")) is not str or not slot["value"]:
        raise StructuredGroundingError("invalid_template")
    value = slot["value"]
    if "source" in slot:
        return _context_value(slot, value, context)
    base = {"type", "value", "evidence_ref", "evidence_field"}
    required = base | {"unit"} if slot.get("type") == "numeric" else base
    if slot.keys() not in (required, required | {"row"}):
        raise StructuredGroundingError("invalid_template")
    ref, name, row = slot["evidence_ref"], slot["evidence_field"], slot.get("row")
    if (type(ref) is not str or not ref or type(name) is not str or not name
            or ("row" in slot and (type(row) is not int or row < 0))):
        raise StructuredGroundingError("invalid_template")
    record = records.get(ref)
    if record is None:
        raise StructuredGroundingError("unsupported_number")

    kind, actual = _tool_field(record, name, row, context.month)
    observe("authority", authoritative_value=actual, type=kind.value[0], unit=kind.value[1])
    if (slot["type"], slot.get("unit")) != kind.value:
        raise StructuredGroundingError("unsupported_number")
    if slot["type"] == "numeric":
        number = _numeric_value(actual, kind)
        if _decimal(value) == number:
            return _render_number(number)
    elif kind is FieldKind.DATE:
        actual_date = _iso_date(actual)
        if _iso_date(value) == actual_date:
            return actual_date
    elif type(actual) is str and value == actual:
        return actual
    raise StructuredGroundingError("unsupported_number")


def validate_and_render(
    answer_json: str,
    *,
    context: GroundingContext,
    tool_results: Iterable[RecordedToolResult] = (),
) -> str:
    """Replace validated slots in the original template using a single pass."""
    answer = _json_object(answer_json, "invalid_template")
    template, slots = answer.get("template"), answer.get("slots")
    observe("grounding_parsed", template=template, slots=slots)
    if (answer.keys() != {"template", "slots"} or not isinstance(template, str)
            or not template.strip() or not isinstance(slots, dict)):
        raise StructuredGroundingError("invalid_template")

    # split keeps the literal spans and the captured slot names in order.
    parts = _PLACEHOLDER.split(template)
    if any(char in "{}" or char.isnumeric() for text in parts[::2] for char in text):
        raise StructuredGroundingError("invalid_template")
    if set(parts[1::2]) != slots.keys():
        raise StructuredGroundingError("invalid_template")

    records = evidence_registry(context.scope, tool_results)
    values = {}
    for name, slot in slots.items():
        observe("slot_start", slot=name)
        try:
            values[name] = _slot_value(slot, context, records)
        except StructuredGroundingError as exc:
            observe("slot_end", decision="rejected", reason_code=exc.reason_code)
            raise
        observe("slot_end", decision="accepted", reason_code=None)
    parts[1::2] = [values[name] for name in parts[1::2]]
    if sum(map(len, parts)) > _MAX_JSON_LENGTH:
        raise StructuredGroundingError("invalid_template")
    rendered = "".join(parts)
    if any("\ud800" <= char <= "\udfff" for char in rendered):
        raise StructuredGroundingError("invalid_template")
    return rendered
