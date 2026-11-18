"""Tenant-bound, read-only access to the supplied demo databases and tickets."""

import csv
import hashlib
import hmac
import re
import sqlite3
from calendar import monthrange
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

from fastapi import HTTPException, Request

from app.month_scope import PurchaseDropScope


class UnsafeQueryError(ValueError):
    """The SQL is outside the permitted read-only query surface."""


class QueryLimitExceeded(ValueError):
    """A query exceeded its row or execution budget."""


_ALLOWED_FUNCTIONS = frozenset(
    {"sum", "count", "avg", "min", "max", "coalesce", "substr", "lower", "upper", "length", "abs", "round"}
)
_MONTH = re.compile(r"\d{4}-(?:0[1-9]|1[0-2])\Z")
MAX_PURCHASE_DROP_ROWS = 5


def purchase_drop_scope(baseline_month: str, comparison_month: str, limit: int) -> PurchaseDropScope:
    if type(limit) is not int or not 1 <= limit <= MAX_PURCHASE_DROP_ROWS:
        raise ValueError("Invalid purchase-drop limit")
    months = baseline_month, comparison_month
    if any(not isinstance(month, str) or not month.isascii() or not _MONTH.fullmatch(month)
           for month in months):
        raise ValueError("Expected Gregorian YYYY-MM months")
    baseline, comparison = (date.fromisoformat(month + "-01") for month in months)
    if baseline >= comparison:
        raise ValueError("Baseline month must precede comparison month")

    def window(first: date) -> tuple[str, str]:
        last = first.replace(day=monthrange(first.year, first.month)[1])
        return first.isoformat(), last.isoformat()

    return PurchaseDropScope(
        f"{baseline_month} to {comparison_month}", (window(baseline), window(comparison)), limit,
    )


def _mapped_path(data_dir: Path, name: str, suffix: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(suffix):
        raise ValueError("Invalid tenant data filename")
    path = (data_dir / name).resolve()
    if path.parent != data_dir or not path.is_file():
        raise ValueError("Tenant data file unavailable")
    return path


def _is_permitted_sql_action(
    action: int, table: str | None, function: str | None, database: str | None
) -> bool:
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
        return True
    if action == sqlite3.SQLITE_READ:
        return table in ("customers", "orders") or database is None
    if action == sqlite3.SQLITE_FUNCTION:
        return (function or "").lower() in _ALLOWED_FUNCTIONS
    return False


def _is_permitted_agent_action(
    action: int,
    table: str | None,
    function: str | None,
    database: str | None,
    source: str | None,
) -> bool:
    # A CTE with either protected table name must not shadow the real table.
    if source in ("customers", "paid_orders"):
        return False
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
        return True
    if action == sqlite3.SQLITE_READ:
        return (database == "main" and table in ("customers", "paid_orders")) or database is None
    if action == sqlite3.SQLITE_FUNCTION:
        return (function or "").lower() in _ALLOWED_FUNCTIONS
    return False


def _paid_snapshot(database: Path) -> sqlite3.Connection:
    """Build an isolated connection containing only agent-visible records."""
    uri = f"file:{database.as_posix()}?mode=ro&immutable=1"
    with closing(sqlite3.connect(uri, uri=True)) as source:
        source.execute("PRAGMA query_only = ON")
        customers = source.execute(
            "SELECT customer_id, name, segment, created_at FROM customers"
        ).fetchall()
        paid_orders = source.execute(
            "SELECT order_id, customer_id, order_date, amount_toman, status, category "
            "FROM orders WHERE status = 'paid'"
        ).fetchall()

    connection = sqlite3.connect(":memory:")
    try:
        connection.row_factory = sqlite3.Row
        connection.execute(
            "CREATE TABLE customers (customer_id TEXT, name TEXT, segment TEXT, created_at TEXT)"
        )
        connection.execute(
            "CREATE TABLE paid_orders (order_id TEXT, customer_id TEXT, order_date TEXT, "
            "amount_toman INTEGER, status TEXT, category TEXT)"
        )
        connection.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", customers)
        connection.executemany("INSERT INTO paid_orders VALUES (?, ?, ?, ?, ?, ?)", paid_orders)
        connection.execute("PRAGMA query_only = ON")
        return connection
    except Exception:
        connection.close()
        raise


class SecureDataAccess:
    def __init__(
        self,
        *,
        data_dir: Path,
        token_map: Mapping[str, Mapping[str, str]],
        max_rows: int = 1000,
        max_sql_steps: int = 100_000,
    ) -> None:
        if max_rows < 1 or max_sql_steps < 1:
            raise ValueError("Query limits must be positive")
        root = Path(data_dir).resolve()
        self._resources = {
            token: (
                _mapped_path(root, files["db"], ".db"),
                _mapped_path(root, files["tickets"], ".csv"),
            )
            for token, files in token_map.items()
        }
        self._max_rows = max_rows
        self._max_sql_steps = max_sql_steps

    def authenticate(self, token: str | None) -> "TenantContext":
        if not token:
            raise HTTPException(status_code=401, detail="Invalid demo token")
        for configured_token, (database, tickets) in self._resources.items():
            if hmac.compare_digest(token, configured_token):
                return TenantContext(database, tickets, self._max_rows, self._max_sql_steps)
        raise HTTPException(status_code=401, detail="Invalid demo token")


def get_tenant_context(request: Request) -> "TenantContext":
    store = getattr(request.app.state, "secure_data", None)
    if not isinstance(store, SecureDataAccess):
        raise HTTPException(status_code=503, detail="Data access unavailable")
    return store.authenticate(request.headers.get("X-Demo-Token"))


class TenantContext:
    def __init__(
        self, database: Path, tickets_file: Path, max_rows: int, max_sql_steps: int
    ) -> None:
        self._database = database
        self._tickets_file = tickets_file
        self._max_rows = max_rows
        self._max_sql_steps = max_sql_steps

    @property
    def state_key(self) -> str:
        """Stable server-derived identity for state ownership, without a token."""
        resources = f"{self._database.resolve()}\0{self._tickets_file.resolve()}"
        return hashlib.sha256(resources.encode("utf-8")).hexdigest()

    @property
    def display_label(self) -> str | None:
        """Public label for a known demo dataset selected by authentication."""
        return {"alpha": "Alpha", "beta": "Beta"}.get(self._database.stem.lower())

    def query(self, sql: str, params: Sequence[object] = ()) -> list[dict[str, object]]:
        if not isinstance(sql, str) or not re.match(r"\A\s*(?:SELECT|WITH)\b", sql, re.I):
            raise UnsafeQueryError("Only read-only SELECT statements are allowed")

        uri = f"file:{self._database.as_posix()}?mode=ro&immutable=1"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            denied = False
            exceeded = False
            steps = 0

            def authorize(
                action: int,
                arg1: str | None,
                arg2: str | None,
                db_name: str | None,
                source: str | None,
            ) -> int:
                nonlocal denied
                if _is_permitted_sql_action(action, arg1, arg2, db_name):
                    return sqlite3.SQLITE_OK
                denied = True
                return sqlite3.SQLITE_DENY

            def progress() -> int:
                nonlocal steps, exceeded
                steps += 1
                if steps > self._max_sql_steps:
                    exceeded = True
                    return 1
                return 0

            connection.set_authorizer(authorize)
            connection.set_progress_handler(progress, 1)
            try:
                cursor = connection.execute(sql, tuple(params))
                rows = cursor.fetchmany(self._max_rows + 1)
            except sqlite3.Error as exc:
                if exceeded:
                    raise QueryLimitExceeded("SQL execution budget exceeded") from exc
                if denied or isinstance(exc, sqlite3.ProgrammingError):
                    raise UnsafeQueryError("SQL statement rejected") from exc
                raise
            finally:
                connection.set_progress_handler(None, 0)
                connection.set_authorizer(None)

        if len(rows) > self._max_rows:
            raise QueryLimitExceeded("SQL row limit exceeded")
        return [dict(row) for row in rows]

    def agent_query(self, sql: str, params: Sequence[object] = ()) -> list[dict[str, object]]:
        """Execute agent SQL against a paid-only, tenant-local snapshot."""
        if not isinstance(sql, str) or not re.match(r"\A\s*(?:SELECT|WITH)\b", sql, re.I):
            raise UnsafeQueryError("Only read-only SELECT statements are allowed")

        with closing(_paid_snapshot(self._database)) as connection:
            denied = False
            exceeded = False
            steps = 0

            def authorize(
                action: int,
                arg1: str | None,
                arg2: str | None,
                db_name: str | None,
                source_name: str | None,
            ) -> int:
                nonlocal denied
                if _is_permitted_agent_action(action, arg1, arg2, db_name, source_name):
                    return sqlite3.SQLITE_OK
                denied = True
                return sqlite3.SQLITE_DENY

            def progress() -> int:
                nonlocal steps, exceeded
                steps += 1
                if steps > self._max_sql_steps:
                    exceeded = True
                    return 1
                return 0

            connection.set_authorizer(authorize)
            connection.set_progress_handler(progress, 1)
            try:
                cursor = connection.execute(sql, tuple(params))
                rows = cursor.fetchmany(self._max_rows + 1)
            except sqlite3.Error as exc:
                if exceeded:
                    raise QueryLimitExceeded("SQL execution budget exceeded") from exc
                if denied or isinstance(exc, (sqlite3.ProgrammingError, sqlite3.OperationalError)):
                    raise UnsafeQueryError("Agent SQL statement rejected") from exc
                raise
            finally:
                connection.set_progress_handler(None, 0)
                connection.set_authorizer(None)

        if len(rows) > self._max_rows:
            raise QueryLimitExceeded("SQL row limit exceeded")
        return [dict(row) for row in rows]

    def customer_purchase_drops(
        self, *, baseline_month: str, comparison_month: str, limit: int,
    ) -> dict[str, object]:
        scope = purchase_drop_scope(baseline_month, comparison_month, limit)
        baseline, comparison = scope.month_windows
        # Snapshot dates are ISO calendar days. Bind both complete windows independently.
        rows = self.agent_query(
            """
            WITH monthly_totals AS (
                SELECT customer_id,
                    SUM(CASE WHEN order_date BETWEEN ?1 AND ?2 THEN amount_toman ELSE 0 END)
                        AS baseline_revenue_toman,
                    SUM(CASE WHEN order_date BETWEEN ?3 AND ?4 THEN amount_toman ELSE 0 END)
                        AS comparison_revenue_toman
                FROM paid_orders
                WHERE order_date BETWEEN ?1 AND ?2 OR order_date BETWEEN ?3 AND ?4
                GROUP BY customer_id
            )
            SELECT customer_id, baseline_revenue_toman, comparison_revenue_toman,
                baseline_revenue_toman - comparison_revenue_toman AS decrease_toman
            FROM monthly_totals
            WHERE baseline_revenue_toman > comparison_revenue_toman
            ORDER BY decrease_toman DESC, customer_id ASC
            LIMIT ?5
            """,
            (*baseline, *comparison, scope.limit),
        )
        return {"baseline_month": baseline_month, "comparison_month": comparison_month, "rows": rows}

    def paid_revenue(self, month: str) -> int:
        if not isinstance(month, str) or not _MONTH.fullmatch(month):
            raise ValueError("Expected a YYYY-MM month")
        rows = self.query(
            "SELECT COALESCE(SUM(amount_toman), 0) AS revenue "
            "FROM orders WHERE status = 'paid' AND substr(order_date, 1, 7) = ?",
            (month,),
        )
        return int(rows[0]["revenue"])

    def tickets(self, *, customer_id: str | None = None) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        with self._tickets_file.open(encoding="utf-8-sig", newline="") as handle:
            for record in csv.DictReader(handle):
                if customer_id is not None and record["customer_id"] != customer_id:
                    continue
                results.append({**record, "is_untrusted": True})
                if len(results) > self._max_rows:
                    raise QueryLimitExceeded("Ticket row limit exceeded")
        return results
