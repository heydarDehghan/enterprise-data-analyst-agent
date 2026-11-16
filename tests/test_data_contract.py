import csv
import sqlite3
from datetime import date
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

TENANTS = (
    ("alpha", "A", 30, 450, 7),
    ("beta", "B", 30, 450, 2),
)

EXPECTED_DB_COLUMNS = {
    "customers": ["customer_id", "name", "segment", "created_at"],
    "orders": [
        "order_id",
        "customer_id",
        "order_date",
        "amount_toman",
        "status",
        "category",
    ],
}

EXPECTED_TICKET_COLUMNS = [
    "ticket_id",
    "customer_id",
    "created_at",
    "category",
    "description",
]


def _read_only_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def _read_tickets(path: Path) -> tuple[list[str] | None, list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def test_required_source_files_exist_and_are_readable() -> None:
    required_paths = [
        DATA_DIR / "schema.md",
        DATA_DIR / "alpha.db",
        DATA_DIR / "beta.db",
        DATA_DIR / "alpha_tickets.csv",
        DATA_DIR / "beta_tickets.csv",
        PROJECT_ROOT / "demo_tokens.json.example",
    ]

    for path in required_paths:
        assert path.is_file(), f"required source file is missing: {path}"
        with path.open("rb") as handle:
            assert handle.read(1), f"required source file is empty: {path}"


@pytest.mark.parametrize(
    ("tenant", "prefix", "expected_customers", "expected_orders", "expected_tickets"),
    TENANTS,
)
def test_tenant_dataset_integrity(
    tenant: str,
    prefix: str,
    expected_customers: int,
    expected_orders: int,
    expected_tickets: int,
) -> None:
    db_path = DATA_DIR / f"{tenant}.db"
    ticket_path = DATA_DIR / f"{tenant}_tickets.csv"

    with _read_only_connection(db_path) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []

        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert tables == set(EXPECTED_DB_COLUMNS)

        for table, expected_columns in EXPECTED_DB_COLUMNS.items():
            actual_columns = [
                row[1]
                for row in connection.execute(f'PRAGMA table_info("{table}")')
            ]
            assert actual_columns == expected_columns

        customers = {
            row[0] for row in connection.execute("SELECT customer_id FROM customers")
        }
        orders = connection.execute(
            "SELECT order_id, customer_id, order_date, amount_toman, status, category "
            "FROM orders"
        ).fetchall()

    assert len(customers) == expected_customers
    assert len(orders) == expected_orders
    assert all(customer_id.startswith(prefix) for customer_id in customers)

    allowed_statuses = {"paid", "cancelled", "refunded"}
    allowed_categories = {"software", "hardware", "services", "subscription"}
    allowed_months = {"2026-08", "2026-09", "2026-10"}

    for order_id, customer_id, order_date, amount, status, category in orders:
        assert order_id.startswith(f"{prefix}O")
        assert customer_id in customers
        assert customer_id.startswith(prefix)
        assert isinstance(amount, int) and amount > 0
        assert status in allowed_statuses
        assert category in allowed_categories
        assert date.fromisoformat(order_date).strftime("%Y-%m") in allowed_months

    ticket_columns, tickets = _read_tickets(ticket_path)
    assert ticket_columns == EXPECTED_TICKET_COLUMNS
    assert len(tickets) == expected_tickets

    for ticket in tickets:
        assert ticket["ticket_id"].startswith(f"{prefix}T")
        assert ticket["customer_id"] in customers
        assert ticket["customer_id"].startswith(prefix)
        date.fromisoformat(ticket["created_at"])
