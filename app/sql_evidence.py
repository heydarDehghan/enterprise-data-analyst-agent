"""Qualify executed SQL projections as evidence using a restricted grammar.

This is not a SQL authorizer. Run the original query through TenantContext
first. Unrecognized SQL can still return results but earns no field metadata.
"""

import calendar
import re
from datetime import date, timedelta

from app.month_scope import MonthScope, _bound_sql
from app.structured_grounding import FieldKind, SqlField


_NAME = r"[a-z_][a-z_0-9]*"
_LITERAL = r"'[a-z0-9_ -]+'"
_ISO_DATE = r"'([0-9]{4}-[0-9]{2}-[0-9]{2})'"
_FLAGS = re.ASCII | re.IGNORECASE
_SELECT = re.compile(
    r"\s*SELECT\s+(?P<fields>.+?)\s+FROM\s+(?:main\.)?(?P<table>paid_orders|customers)"
    r"(?:\s+WHERE\s+(?P<where>.+?))?\s*;?\s*", _FLAGS | re.DOTALL,
)
_SUM = r"SUM\s*\(\s*amount_toman\s*\)"
_COUNT = r"COUNT\s*\(\s*\*\s*\)"
_SHARE = (
    rf"ROUND\s*\(\s*100\.0\s*\*\s*COUNT\s*\(\s*CASE\s+WHEN\s+customer_id\s*=\s*{_LITERAL}"
    rf"\s+THEN\s+1\s+END\s*\)\s*/\s*{_COUNT}\s*,\s*2\s*\)"
)
_PROJECTION = re.compile(
    rf"\s*(?:(?P<money>{_SUM}|COALESCE\s*\(\s*{_SUM}\s*,\s*0\s*\))|"
    rf"(?P<count>{_COUNT})|(?P<share>{_SHARE}))\s+AS\s+(?P<name>{_NAME})\s*(?P<sep>,|$)",
    _FLAGS,
)
_MONTH = re.compile(r"substr\s*\(\s*order_date\s*,\s*1\s*,\s*7\s*\)\s*=\s*'([0-9]{4}-[0-9]{2})'", _FLAGS)
_BETWEEN = re.compile(rf"order_date\s+BETWEEN\s+{_ISO_DATE}\s+AND\s+{_ISO_DATE}", _FLAGS)
_RANGE = re.compile(rf"order_date\s*>=\s*{_ISO_DATE}\s+AND\s+order_date\s*(<=|<)\s*{_ISO_DATE}", _FLAGS)
_FILTER = re.compile(rf"(?:customer_id|category|status)\s*=\s*{_LITERAL}", _FLAGS)
_SUFFIX = re.compile(rf"\s+AND\s+{_FILTER.pattern}\s*", _FLAGS)


def _window(predicate: str) -> MonthScope | None:
    """A complete supported date predicate, optionally followed by one filter."""
    try:
        for pattern in (_MONTH, _BETWEEN, _RANGE):
            match = pattern.match(predicate)
            if not match:
                continue
            remainder = predicate[match.end():]
            if remainder.strip() and not _SUFFIX.fullmatch(remainder):
                continue
            if pattern is _MONTH:
                month = match[1]
                first = date.fromisoformat(month + "-01")
                last = first.replace(day=calendar.monthrange(first.year, first.month)[1])
            else:
                month = None
                first = date.fromisoformat(match[1])
                last = date.fromisoformat(match.groups()[-1])
                if pattern is _RANGE and match[2] == "<":
                    last -= timedelta(days=1)
            if first <= last:
                return MonthScope("Verified SQL window", first.isoformat(), last.isoformat(),
                                  (last + timedelta(days=1)).isoformat(), month)
    except (ValueError, OverflowError):
        return None
    return None


def sql_provenance(
    sql: str, params: list, rows: list[dict],
) -> tuple[dict[str, SqlField], MonthScope | None]:
    """Return metadata only if every projection and predicate can be verified."""
    bound = _bound_sql(sql, params)
    query = _SELECT.fullmatch(bound) if bound is not None else None
    if query is None:
        return {}, None
    table = query["table"].lower()
    predicate = query["where"]
    window = _window(predicate.strip()) if predicate and table == "paid_orders" else None
    if predicate and window is None and not _FILTER.fullmatch(predicate.strip()):
        return {}, None

    fields = {}
    seen = set()
    projection = query["fields"]
    position = 0
    while position < len(projection):
        match = _PROJECTION.match(projection, position)
        if match is None:
            return {}, None
        name = match["name"]
        if name.lower() in seen or any(name not in row for row in rows):
            return {}, None
        seen.add(name.lower())
        if match["count"]:
            fields[name] = SqlField(FieldKind.COUNT, table, "*", match["count"])
        elif table == "paid_orders":
            if match["money"]:
                fields[name] = SqlField(FieldKind.TOMAN, table, "amount_toman", match["money"])
            else:
                fields[name] = SqlField(FieldKind.PERCENT, table, "customer_id", match["share"])
        position = match.end()
        if match["sep"] == "," and position == len(projection):
            return {}, None
    return fields, window
