"""Fuzzing: malformed or hostile input must produce a clean SQLError (API: 4xx), never a
crash, a hang or a 500."""
import random

import pytest

from conftest import requires_data
from src.engine import SQLError

pytestmark = requires_data

SEEDS = [
    "SELECT COUNT(*) FROM employees e JOIN projects p ON e.dept_id = p.dept_id WHERE e.city = 'Pune'",
    "SELECT e.name FROM employees e WHERE e.age BETWEEN 30 AND 40 AND e.city IN ('Delhi', 'Pune') LIMIT 5",
    "SELECT COUNT(*) FROM assignments a JOIN employees e ON a.emp_id = e.emp_id WHERE a.hours_per_week > 20",
]
JUNK = ["'", "(", ")", ";", "--", "/*", "OR", "AND", "NULL", "''", "1=1", "DROP TABLE employees", "\x00",
        "😀", "e.", "..", "BETWEEN", "IN ()", "9" * 400, "-", "UNION SELECT 1", "WHERE", "JOIN", ","]


def mutate(sql: str, rng: random.Random) -> str:
    toks = sql.split(" ")
    for _ in range(rng.randint(1, 4)):
        op = rng.random()
        i = rng.randrange(len(toks))
        if op < 0.4:
            toks.insert(i, rng.choice(JUNK))
        elif op < 0.7 and len(toks) > 1:
            toks.pop(i)
        else:
            toks[i] = rng.choice(JUNK)
    return " ".join(toks)


def test_parser_and_executor_never_crash(engine):
    rng = random.Random(0)
    for _ in range(600):
        sql = mutate(rng.choice(SEEDS), rng)
        try:
            q = engine.parse(sql)
            engine.true_count(q)
            if not q.count:
                engine.executor.fetch(q, 5)
        except SQLError:
            pass  # rejected cleanly


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import server
    with TestClient(server.app) as c:
        yield c


def test_api_never_500s(client):
    rng = random.Random(1)
    for _ in range(80):
        r = client.post("/api/run", json={"sql": mutate(rng.choice(SEEDS), rng)})
        assert r.status_code in (200, 400, 422), r.text
