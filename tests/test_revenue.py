import hashlib
import sqlite3
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

EXPECTED_REVENUE = {
    "alpha": {
        "2026-09": 53_250_000,
        "2026-10": 57_507_499,
        "change_percent": 7.9953032863849765,
    },
    "beta": {
        "2026-09": 106_500_000,
        "2026-10": 134_189_994,
        "change_percent": 25.999994366197183,
    },
}


def _read_only_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _independent_revenue(path: Path, month: str, *, paid_only: bool) -> int:
    status_clause = "AND status = 'paid'" if paid_only else ""
    with _read_only_connection(path) as connection:
        result = connection.execute(
            "SELECT COALESCE(SUM(amount_toman), 0) "
            "FROM orders "
            "WHERE substr(order_date, 1, 7) = ? "
            f"{status_clause}",
            (month,),
        ).fetchone()
    return int(result[0])


@pytest.mark.parametrize("tenant", ("alpha", "beta"))
def test_source_data_has_the_independently_verified_paid_only_totals(
    tenant: str,
) -> None:
    """Canceled and refunded orders must not contribute to revenue."""
    db_path = DATA_DIR / f"{tenant}.db"

    for month in ("2026-09", "2026-10"):
        paid_total = _independent_revenue(db_path, month, paid_only=True)
        all_status_total = _independent_revenue(db_path, month, paid_only=False)

        assert paid_total == EXPECTED_REVENUE[tenant][month]
        assert paid_total < all_status_total


@pytest.mark.parametrize("tenant", ("alpha", "beta"))
def test_paid_revenue_reader_returns_september_and_october_without_mutation(
    tenant: str,
) -> None:
    """Application revenue access is paid-only and leaves the source immutable."""
    from app.analytics import paid_revenue_for_month

    db_path = DATA_DIR / f"{tenant}.db"
    before_hash = _source_hash(db_path)

    september = paid_revenue_for_month(db_path, "2026-09")
    october = paid_revenue_for_month(db_path, "2026-10")

    assert september == EXPECTED_REVENUE[tenant]["2026-09"]
    assert october == EXPECTED_REVENUE[tenant]["2026-10"]
    assert _source_hash(db_path) == before_hash


@pytest.mark.parametrize("tenant", ("alpha", "beta"))
def test_september_to_october_percentage_change(tenant: str) -> None:
    """The change formula is (October - September) / September * 100."""
    from app.analytics import percentage_change

    september = EXPECTED_REVENUE[tenant]["2026-09"]
    october = EXPECTED_REVENUE[tenant]["2026-10"]

    actual = percentage_change(previous=september, current=october)

    assert actual == pytest.approx(
        EXPECTED_REVENUE[tenant]["change_percent"], rel=1e-12
    )
