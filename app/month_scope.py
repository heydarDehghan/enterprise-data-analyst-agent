"""Conservative calendar scope checks for agent-facing revenue evidence."""

import calendar
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any


_DEMO_MONTH_NAMES = {
    8: ("august", "آگوست", "اوت"),
    9: ("september", "سپتامبر", "شهریور"),
    10: ("october", "اکتبر"),
}
_ISO_DATE = re.compile(r"\b[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}\b")
_DEMO_MONTH_ID = re.compile(r"\b2026-(0[1-9]|1[0-2])\b")
_NAMED_MONTHS = {name: month for month, names in _DEMO_MONTH_NAMES.items() for name in names}
_MONTH_WORD = re.compile(
    r"\b(?:" + "|".join(_NAMED_MONTHS)
    + r"|january|february|march|april|may|june|july|november|december"
    r"|ژانویه|فوریه|مارس|آوریل|مه|ژوئن|ژوئیه|ژولای|نوامبر|دسامبر)\b",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b[0-9]{4}\b")
_NON_GREGORIAN = re.compile(r"\b(?:هجری|شمسی|خورشیدی|قمری|hijri)\b", re.IGNORECASE)
_PREVIOUS_REPORT_SUBJECT = r"عدد (?:دقیق )?گزارش قبلی را (?:با رقم )?"
_PREVIOUS_REPORT_REQUEST = re.compile(
    _PREVIOUS_REPORT_SUBJECT + r"دوباره بگو"
    r"(?: و نمودار قبلی را نگه دار)?"
    r"(?:[.؟] برای slot عددی از همان شاهد execute_python و فیلد revenue استفاده کن)?[.؟]?",
)


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
            f"{self.first} through {self.last} inclusive. "
            f"For SQL and Python revenue filters use order_date >= {self.first} and order_date < {self.next_first}. "
            f"Use user_request.period_start = {self.first} and user_request.period_end = {self.last} "
            "(the inclusive final calendar day). "
            f"The exclusive cutoff {self.next_first} is only for filtering. "
            "Gregorian September and the Persian month Shahrivar are different periods."
        )


@dataclass(frozen=True)
class ComparisonScope:
    label: str
    month_windows: tuple[tuple[str, str], ...]

    @property
    def first(self) -> str:
        return min(window[0] for window in self.month_windows)

    @property
    def last(self) -> str:
        return max(window[1] for window in self.month_windows)


@dataclass(frozen=True)
class PurchaseDropScope(ComparisonScope):
    """A ranked decrease with baseline month first and comparison month second."""

    month_windows: tuple[tuple[str, str], tuple[str, str]]
    limit: int


def _ascii_digits(text: str) -> str:
    return "".join(
        str(unicodedata.decimal(char)) if char.isdecimal() else char
        for char in text
    )


def _month_window(month: int) -> tuple[str, str]:
    start = date(2026, month, 1)
    last = start.replace(day=calendar.monthrange(start.year, month)[1])
    return start.isoformat(), last.isoformat()


def _requested_three_months(question: str) -> ComparisonScope | None:
    # Use the original offsets to keep the display label verbatim.
    normalized = _ascii_digits(question)
    names = list(_MONTH_WORD.finditer(normalized))
    labels = [match[0].casefold() for match in names]
    months = [_NAMED_MONTHS.get(label, 0) for label in labels]
    if 0 in months or len(set(months)) != 3 or "شهریور" in labels:
        return None
    separator = r"(?:\s+2026)?(?:\s*[,،]\s*(?:(?:and|و)\s+)?|\s+(?:and|و)\s+)"
    if any(re.fullmatch(separator, normalized[left.end():right.start()], re.IGNORECASE) is None
           for left, right in zip(names, names[1:])):
        return None
    suffix = re.match(r"\s+(?:میلادی\s+)?2026\b(?![-./][0-9])(?:\s+میلادی\b)?", normalized[names[-1].end():])
    if suffix is None:
        return None
    start, end = names[0].start(), names[-1].end() + suffix.end()
    if _YEAR.search(normalized[:start] + " " + normalized[end:]):
        return None
    return ComparisonScope(question[start:end], tuple(_month_window(month) for month in months))


def requested_comparison(question: str) -> ComparisonScope | None:
    """Recognize an explicit pair or list of three named Gregorian demo months."""
    display = " ".join(question.split())
    normalized = _ascii_digits(display)
    if re.search(r"\b" + _PREVIOUS_REPORT_SUBJECT + r"(?:دوباره )?بگو\b", normalized):
        return None
    names = list(_MONTH_WORD.finditer(normalized))
    if _NON_GREGORIAN.search(normalized):
        return None
    if len(names) == 3:
        return _requested_three_months(question)
    if len(names) != 2:
        return None
    first, second = names
    month_names = (first[0].casefold(), second[0].casefold())
    first_month, second_month = (_NAMED_MONTHS.get(name) for name in month_names)
    if first_month is None or second_month is None or "شهریور" in month_names or first_month == second_month:
        return None

    # A shared trailing year or the same explicit year on both named months.
    between = normalized[first.end():second.start()]
    suffix = re.match(r" (?:میلادی )?2026\b(?![-./][0-9])(?: میلادی\b)?", normalized[second.end():])
    purchase_drop = (
        between == " نسبت به " and (first_month, second_month) == (10, 9)
        and normalized[:first.start()] == "5 مشتری با بیشترین افت مبلغ خرید در "
    )
    if suffix is None or (
        not purchase_drop and not re.fullmatch(r" (?:2026 )?(?:و|(?:را )?با|and|with) ", between, re.IGNORECASE)
    ):
        return None
    end = second.end() + suffix.end()
    if _YEAR.search(normalized[:first.start()] + " " + normalized[end:]):
        return None

    label = display[first.start():end]
    if purchase_drop:
        return PurchaseDropScope(label, (_month_window(second_month), _month_window(first_month)), limit=5)
    return ComparisonScope(label, (_month_window(first_month), _month_window(second_month)))


def references_previous_report(question: str) -> bool:
    """Recognize the supported repeat requests without allowing an extra date clause."""
    return _PREVIOUS_REPORT_REQUEST.fullmatch(" ".join(question.casefold().split())) is not None


def _explicit_date_window(question: str, dates: list[re.Match[str]]) -> tuple[date, date] | None:
    """Read an unambiguous interval and return its inclusive calendar endpoints."""
    if len(dates) != 2:
        return None
    start_token, end_token = dates
    try:
        start, end = date.fromisoformat(start_token.group()), date.fromisoformat(end_token.group())
    except ValueError:
        return None

    head = question[:start_token.start()].rstrip()
    connector = question[start_token.end():end_token.start()].strip()
    tail = question[end_token.end():].lstrip(" ,،")
    on_or_after = bool(re.search(r"\bon or after$", head))
    if not on_or_after and re.search(r"\b(?:after|before|بعد از|پیش از|قبل از)$", head):
        return None

    inclusive = bool(re.match(r"(?:inclusive|شامل)\b", tail))
    if connector in ("through", "to", "تا") and inclusive:
        return (start, end) if start <= end else None

    exclusive = (
        (connector == "and before" and (on_or_after or re.search(r"\bfrom$", head)))
        or (re.fullmatch(r"(?:شامل آن روز )?تا (?:قبل|پیش) از", connector) and re.search(r"\bاز$", head))
        or (connector in ("to", "inclusive to") and re.match(r"exclusive\b", tail))
    )
    if exclusive and not inclusive and start < end:
        return start, end - timedelta(days=1)
    return None


def requested_month(question: str) -> MonthScope | None:
    normalized = " ".join(_ascii_digits(question).lower().split())
    dates = list(_ISO_DATE.finditer(normalized))
    window = _explicit_date_window(normalized, dates) if dates else None
    if dates and window is None:
        return None

    # Remove calendar endpoints before scanning for independent month references.
    wording = _ISO_DATE.sub(" ", normalized)
    mentioned = {int(month) for month in _DEMO_MONTH_ID.findall(wording)}
    if "شهریور" in wording and "1405" in wording:
        if mentioned or ("2026" in normalized and any(
            name in wording
            for names in _DEMO_MONTH_NAMES.values()
            for name in names
            if name != "شهریور"
        )):
            return None  # A comparison needs more than one calendar scope.
        first, last = "2026-08-23", "2026-09-22"
        if window and tuple(day.isoformat() for day in window) != (first, last):
            return None
        return MonthScope("Shahrivar 1405 Solar Hijri (شهریور ۱۴۰۵ شمسی)", first, last, "2026-09-23")

    # The supplied 2026 demo questions use Gregorian year/month labels. The
    # older wording "شهریور ۲۰۲۶" is interpreted as that demo convention,
    # never as Shahrivar 1405 of the Persian calendar.
    if "2026" not in normalized:
        return None
    mentioned.update(month for month, names in _DEMO_MONTH_NAMES.items()
                     if any(name in wording for name in names))
    if window:
        mentioned.add(window[0].month)
    if len(mentioned) != 1:
        return None  # Avoid imposing one month on a comparison question.
    month, = mentioned
    if month not in _DEMO_MONTH_NAMES:
        return None
    first = date(2026, month, 1)
    last = date(2026, month, calendar.monthrange(2026, month)[1])
    if window and window != (first, last):
        return None
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
