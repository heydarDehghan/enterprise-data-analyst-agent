import sqlite3
from contextlib import closing
from pathlib import Path


_PAID_REVENUE_SQL = (
    "SELECT COALESCE(SUM(amount_toman), 0) "
    "FROM orders "
    "WHERE substr(order_date, 1, 7) = ? AND status = 'paid'"
)


def _read_only_connection(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def paid_revenue_for_month(db_path: str | Path, month: str) -> int:
    with closing(_read_only_connection(db_path)) as connection:
        row = connection.execute(
            _PAID_REVENUE_SQL,
            (month,),
        ).fetchone()

    return int(row[0])


def percentage_change(*, previous: int, current: int) -> float:
    return (current - previous) / previous * 100
