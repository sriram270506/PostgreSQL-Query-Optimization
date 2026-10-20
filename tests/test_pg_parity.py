"""Every SQL construct the engine accepts must give exactly PostgreSQL's answer, and every
construct it cannot execute exactly must be rejected (never silently approximated).

Parity tests need a live server: set PG_DSN (see .env.example) and `docker compose up -d`.
"""
import os

import pytest

from conftest import requires_data
from src.engine import SQLError

pytestmark = requires_data

ACCEPTED = [
    "SELECT COUNT(*) FROM employees",
    "SELECT COUNT(*) FROM employees WHERE city = 'Pune'",
    "SELECT COUNT(*) FROM employees WHERE city <> 'Pune'",
    "SELECT COUNT(*) FROM employees WHERE age < 30",
    "SELECT COUNT(*) FROM employees WHERE age <= 30",
    "SELECT COUNT(*) FROM employees WHERE age > 60",
    "SELECT COUNT(*) FROM employees WHERE age >= 60",
    "SELECT COUNT(*) FROM employees WHERE 30 < age",
    "SELECT COUNT(*) FROM employees WHERE salary BETWEEN 50000 AND 90000",
    "SELECT COUNT(*) FROM employees WHERE city IN ('Pune', 'Delhi')",
    "SELECT COUNT(*) FROM employees WHERE age >= 30 AND age <= 40 AND (city = 'Pune')",
    "SELECT COUNT(*) FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = 'Engineering'",
    "SELECT COUNT(*) FROM employees e INNER JOIN projects p ON e.dept_id = p.dept_id AND p.budget_tier = 5",
    "SELECT COUNT(*) FROM employees e, projects p WHERE e.dept_id = p.dept_id AND e.city = 'Delhi'",
    "SELECT COUNT(*) FROM employees e CROSS JOIN departments d WHERE d.region = 'South'",
    "SELECT COUNT(*) FROM departments d JOIN employees e ON d.dept_id = e.dept_id JOIN projects p ON d.dept_id = p.dept_id "
    "WHERE e.dept_id = p.dept_id AND e.city = 'Pune'",  # cyclic join graph
    "SELECT COUNT(*) FROM employees e JOIN assignments a ON e.emp_id = a.emp_id JOIN projects p ON p.proj_id = a.proj_id "
    "JOIN budgets b ON b.proj_id = p.proj_id WHERE p.priority = 'High' AND e.age BETWEEN 25 AND 35",
]
REJECTED = [
    "SELECT COUNT(e.salary) FROM employees e",
    "SELECT COUNT(DISTINCT city) FROM employees",
    "SELECT SUM(salary) FROM employees",
    "SELECT DISTINCT city FROM employees",
    "SELECT COUNT(*) FROM employees e LEFT JOIN projects p ON e.dept_id = p.dept_id",
    "SELECT COUNT(*) FROM employees e FULL OUTER JOIN projects p ON e.dept_id = p.dept_id",
    "SELECT COUNT(*) FROM employees e JOIN projects p USING (dept_id)",
    "SELECT COUNT(*) FROM employees NATURAL JOIN projects",
    "SELECT COUNT(*) FROM employees e JOIN projects p",
    "SELECT COUNT(*) FROM employees e JOIN projects p ON e.dept_id < p.dept_id",
    "SELECT city FROM employees LIMIT 5 OFFSET 3",
    "SELECT * FROM employees LIMIT -1",
    "SELECT COUNT(*) FROM employees WHERE city = 'Pune' OR age > 30",
    "SELECT COUNT(*) FROM employees WHERE name LIKE 'A%'",
    "SELECT COUNT(*) FROM employees GROUP BY city",
    "WITH x AS (SELECT * FROM employees) SELECT COUNT(*) FROM x",
]


@pytest.mark.parametrize("sql", REJECTED)
def test_unsupported_semantics_are_rejected(engine, sql):
    with pytest.raises(SQLError):
        engine.true_count(engine.parse(sql))


@pytest.fixture(scope="module")
def pg():
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        pytest.skip("PG_DSN not set: live PostgreSQL parity tests need the docker database")
    from src.engine import datasets
    from src.engine.live_pg import LivePG, dsn_for
    try:
        conn = LivePG(dsn_for(dsn, datasets.get("hr")))
    except Exception as e:
        pytest.skip(f"PostgreSQL unreachable: {e}")
    yield conn
    conn.close()


@pytest.mark.parametrize("sql", ACCEPTED)
def test_accepted_constructs_match_postgres(engine, pg, sql):
    pg.cur.execute(sql)
    assert engine.true_count(engine.parse(sql)) == pg.cur.fetchone()[0]


def test_row_fetch_returns_real_rows(engine, pg):
    sql = "SELECT e.emp_id, e.city FROM employees e WHERE e.city = 'Pune' AND e.age > 50 LIMIT 7"
    rows = engine.run(sql)["rows"]
    assert len(rows) == 7
    pg.cur.execute("SELECT emp_id FROM employees WHERE city = 'Pune' AND age > 50")
    allowed = {r[0] for r in pg.cur.fetchall()}
    assert set(rows["emp_id"].astype(int)) <= allowed  # single-table results drop the alias prefix
