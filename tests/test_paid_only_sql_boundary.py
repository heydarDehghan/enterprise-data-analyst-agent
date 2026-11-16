"""Phase 1-C: SQL available to an agent is paid-only and tenant-bound.

The previously approved query() contract remains available for trusted internal
use. An Agent Loop must use agent_query() and never receive query() as a tool.
"""

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TOKEN_MAP = json.loads((ROOT / "demo_tokens.json.example").read_text(encoding="utf-8"))


@pytest.fixture
def store():
    from app.secure_data import SecureDataAccess

    return SecureDataAccess(data_dir=DATA, token_map=TOKEN_MAP)


@pytest.mark.parametrize(
    ("token", "customer_id", "september_total"),
    (
        ("demo-alpha", "A001", 53_250_000),
        ("demo-beta", "B001", 106_500_000),
    ),
)
def test_dynamic_select_can_read_customers_and_paid_orders(
    store, token, customer_id, september_total
):
    tenant = store.authenticate(token)

    assert tenant.agent_query(
        "SELECT customer_id FROM customers WHERE customer_id = ?", (customer_id,)
    ) == [{"customer_id": customer_id}]
    assert tenant.agent_query(
        "SELECT SUM(amount_toman) AS revenue FROM paid_orders "
        "WHERE order_date >= ? AND order_date < ?",
        ("2026-09-01", "2026-10-01"),
    ) == [{"revenue": september_total}]


def test_legitimate_join_and_cte_remain_available(store):
    tenant = store.authenticate("demo-alpha")

    rows = tenant.agent_query(
        "WITH monthly AS ("
        "SELECT customer_id, SUM(amount_toman) AS revenue FROM paid_orders "
        "WHERE order_date >= ? AND order_date < ? GROUP BY customer_id"
        ") SELECT c.customer_id, monthly.revenue FROM customers AS c "
        "JOIN monthly ON monthly.customer_id = c.customer_id "
        "WHERE c.customer_id = ?",
        ("2026-09-01", "2026-10-01", "A001"),
    )

    assert rows == [{"customer_id": "A001", "revenue": 1_050_000}]


@pytest.mark.parametrize(
    "raw_sql",
    (
        "SELECT order_id FROM orders LIMIT 1",
        "SELECT SUM(amount_toman) FROM orders WHERE status = 'paid'",
        "SELECT COUNT(*) FROM main.orders",
        "SELECT o.status FROM orders AS o LIMIT 1",
        "SELECT * FROM (SELECT * FROM orders) AS raw LIMIT 1",
        "WITH raw AS (SELECT * FROM orders) SELECT * FROM raw LIMIT 1",
        "WITH paid_orders AS (SELECT * FROM orders) "
        "SELECT order_id FROM paid_orders LIMIT 1",
        "SELECT customer_id FROM customers WHERE EXISTS "
        "(SELECT 1 FROM orders WHERE orders.customer_id = customers.customer_id)",
        "SELECT name FROM sqlite_master",
    ),
)
def test_agent_sql_rejects_raw_orders_even_when_nested_or_aliased(store, raw_sql):
    from app.secure_data import UnsafeQueryError

    tenant = store.authenticate("demo-alpha")
    before = hashlib.sha256((DATA / "alpha.db").read_bytes()).hexdigest()

    with pytest.raises(UnsafeQueryError):
        tenant.agent_query(raw_sql)

    assert hashlib.sha256((DATA / "alpha.db").read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    ("token", "september_total"),
    (("demo-alpha", 53_250_000), ("demo-beta", 106_500_000)),
)
def test_cancelled_and_refunded_orders_cannot_enter_agent_sales_results(
    store, token, september_total
):
    tenant = store.authenticate(token)

    assert tenant.agent_query(
        "SELECT DISTINCT status FROM paid_orders"
    ) == [{"status": "paid"}]
    assert tenant.agent_query(
        "SELECT order_id FROM paid_orders WHERE status IN ('cancelled', 'refunded')"
    ) == []
    assert tenant.agent_query(
        "SELECT SUM(amount_toman) AS revenue FROM paid_orders "
        "WHERE substr(order_date, 1, 7) = ?",
        ("2026-09",),
    ) == [{"revenue": september_total}]


@pytest.mark.parametrize(
    ("token", "own_prefix", "other_customer"),
    (("demo-alpha", "A", "B001"), ("demo-beta", "B", "A001")),
)
def test_agent_sql_keeps_database_isolation(store, token, own_prefix, other_customer):
    tenant = store.authenticate(token)

    rows = tenant.agent_query(
        "SELECT customer_id FROM paid_orders ORDER BY customer_id LIMIT 10"
    )
    other = tenant.agent_query(
        "SELECT customer_id FROM customers WHERE customer_id = ?", (other_customer,)
    )

    assert rows
    assert all(row["customer_id"].startswith(own_prefix) for row in rows)
    assert other == []
