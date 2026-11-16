"""Conservative calendar scope checks for agent-facing revenue evidence."""

import calendar
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any


@dataclass(frozen=True)
class MonthScope:
    label: str
    first: str
    last: str
    next_first: str
    gregorian_month: str | None = None

    @property
    def guidance(self) -> str:
        return (
            f"For this question, {self.label} covers Gregorian dates "
            f"{self.first} through {self.last} inclusive (or before "
            f"{self.next_first}). Use this exact period in revenue SQL. "
            "Gregorian September and the Persian month Shahrivar are different periods."
        )


def _ascii_digits(text: str) -> str:
    return "".join(
        str(unicodedata.decimal(char)) if char.isdecimal() else char
        for char in text
    )


def requested_month(question: str) -> MonthScope | None:
    normalized = _ascii_digits(question).lower()
    if "شهریور" in normalized and "1405" in normalized:
        if "2026" in normalized and any(
            name in normalized for name in ("september", "سپتامبر", "october", "اکتبر")
        ):
            return None  # A comparison needs more than one calendar scope.
        return MonthScope("Shahrivar 1405 Solar Hijri (شهریور ۱۴۰۵ شمسی)",
                          "2026-08-23", "2026-09-22", "2026-09-23")

    # The supplied 2026 demo questions use Gregorian year/month labels. The
    # older wording "شهریور ۲۰۲۶" is interpreted as that demo convention,
    # never as Shahrivar 1405 of the Persian calendar.
    if "2026" not in normalized:
        return None
    months = {
        8: ("august", "آگوست", "اوت"),
        9: ("september", "سپتامبر", "شهریور"),
        10: ("october", "اکتبر"),
    }
    mentioned = [
        month for month, names in months.items()
        if any(name in normalized for name in names) or f"2026-{month:02d}" in normalized
    ]
    if len(mentioned) != 1:
        return None  # Avoid imposing one month on a comparison question.
    month = mentioned[0]
    first = date(2026, month, 1)
    last = date(2026, month, calendar.monthrange(2026, month)[1])
    return MonthScope(
        f"Gregorian {first.strftime('%B')} 2026 (میلادی)",
        first.isoformat(), last.isoformat(),
        (last + timedelta(days=1)).isoformat(), f"2026-{month:02d}",
    )


def needs_month_scope(sql: str, rows: list[dict[str, Any]]) -> bool:
    return bool(re.search(r"\bamount_toman\b", sql, re.I)) or any(
        "amount_toman" in row for row in rows if isinstance(row, dict)
    )


def _bound_sql(sql: str, params: list[Any]) -> str | None:
    """Substitute positional values for predicate inspection, omitting comments."""
    output: list[str] = []
    index = position = 0
    while index < len(sql):
        if sql.startswith("--", index):
            end = sql.find("\n", index)
            index = len(sql) if end < 0 else end
            output.append(" ")
        elif sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            if end < 0:
                return None
            index = end + 2
            output.append(" ")
        elif sql[index] in ("'", '"'):
            quote = sql[index]
            start = index
            index += 1
            while index < len(sql):
                if sql[index] == quote:
                    if index + 1 < len(sql) and sql[index + 1] == quote:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            else:
                return None
            output.append(sql[start:index])
        elif sql[index] == "?":
            if position >= len(params):
                return None
            value = params[position]
            position += 1
            output.append("'" + value.replace("'", "''") + "'"
                          if isinstance(value, str) else str(value))
            index += 1
        else:
            output.append(sql[index])
            index += 1
    return "".join(output) if position == len(params) else None


def sql_matches_month(sql: str, params: list[Any], scope: MonthScope) -> bool:
    """Accept only recognizable whole-month predicates; unknown forms fail closed."""
    bound = _bound_sql(sql, params)
    if bound is None or re.search(r"\b(?:OR|UNION|INTERSECT|EXCEPT)\b", bound, re.I):
        return False
    if len(re.findall(r"\bpaid_orders\b", bound, re.I)) != 1:
        return False  # Do not validate one predicate for two revenue sources.
    date_references = len(re.findall(r"\border_date\b", bound, re.I))
    column = r"(?:\b[A-Za-z_][A-Za-z_0-9]*\.)?order_date\b"
    literal = r"'(\d{4}-\d{2}(?:-\d{2})?)'"
    all_dates = set(re.findall(literal, bound))

    if scope.gregorian_month is not None:
        substring = re.search(
            rf"\bsubstr\s*\(\s*{column}\s*,\s*1\s*,\s*7\s*\)\s*=\s*{literal}",
            bound, re.I,
        )
        if substring and date_references == 1 and substring.group(1) == scope.gregorian_month and (
            all_dates == {scope.gregorian_month}
        ):
            return True

    between = re.search(
        rf"{column}\s+BETWEEN\s+{literal}\s+AND\s+{literal}", bound, re.I,
    )
    if between and date_references == 1 and between.groups() == (scope.first, scope.last) and (
        all_dates == {scope.first, scope.last}
    ):
        return True

    lower = re.search(rf"{column}\s*>=\s*{literal}", bound, re.I)
    upper = re.search(rf"{column}\s*(<=|<)\s*{literal}", bound, re.I)
    if lower and upper and date_references == 2 and lower.group(1) == scope.first:
        operator, endpoint = upper.groups()
        expected = scope.next_first if operator == "<" else scope.last
        if endpoint == expected and all_dates == {scope.first, endpoint}:
            return True
    return False
