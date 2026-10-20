"""Regression tests for production behaviour: limits, budgets, disconnected joins, stale
models, API protection and the feedback log."""
import json

import numpy as np
import pytest

from conftest import requires_data
from src.engine import SQLError
from src.engine.executor import MAX_ROWS

pytestmark = requires_data


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    import server
    with TestClient(server.app) as c:
        yield c


# ── limits and budgets ──────────────────────────────────────────────────────

def test_one_hard_row_cap(engine):
    assert len(engine.run("SELECT * FROM assignments LIMIT 5000")["rows"]) == MAX_ROWS
    assert len(engine.run("SELECT * FROM assignments LIMIT 5000", row_limit=50)["rows"]) == 50
    assert len(engine.run("SELECT * FROM assignments LIMIT 7", row_limit=500)["rows"]) == 7


def test_api_caps_rows_whatever_sql_says(client):
    b = client.post("/api/run", json={"sql": "SELECT * FROM assignments LIMIT 100000", "row_limit": 10}).json()
    assert b["rows"]["shown"] == 10 and b["count"] == 10000


@pytest.mark.parametrize("sql,msg", [
    ("SELECT COUNT(*) FROM employees a, employees b, employees c, employees d, employees e, employees f, "
     "employees g, employees h, employees i", "tables"),
    ("SELECT COUNT(*) FROM employees WHERE age IN (" + ", ".join(map(str, range(300))) + ")", "IN list"),
])
def test_complexity_caps(engine, sql, msg):
    with pytest.raises(SQLError, match=msg):
        engine.run(sql)


def test_time_budget_stops_execution(engine):
    with pytest.raises(SQLError, match="budget"):
        engine.run("SELECT COUNT(*) FROM employees e JOIN assignments a ON e.emp_id = a.emp_id", budget_s=-1)
    assert engine.executor.deadline is None  # always reset, even after a failure


def test_cyclic_join_refused_before_materialising(engine):
    sql = ("SELECT COUNT(*) FROM projects p JOIN employees e ON e.dept_id = p.dept_id "
           "JOIN assignments a ON a.emp_id = e.emp_id WHERE a.proj_id = p.proj_id")
    with pytest.raises(SQLError, match="in-memory fallback"):
        engine.run(sql)


def test_disconnected_query_runs_and_skips_plans(client):
    r = client.post("/api/run", json={"sql": "SELECT COUNT(*) FROM departments d, locations l, budgets b "
                                              "WHERE l.office_tier = 1 AND d.region = 'South' AND b.approved_amount < 20000"})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["plans"] is None and "cross product" in b["plans_note"]


# ── stale models ────────────────────────────────────────────────────────────

def test_stale_model_is_not_served_as_healthy():
    from src.estimator.service import Estimators
    S = Estimators("hr")
    if not S.ensembles:
        pytest.skip("no trained models")
    name = next(iter(S.ensembles))
    S.stale[name] = True
    q = S.engine.parse("SELECT COUNT(*) FROM employees WHERE city = 'Pune'")
    r = S.estimate_many([q], name)
    assert r["stale"] and r["fallback"].all()
    assert r["est"][0] == pytest.approx(S.estimate_many([q], "factorjoin")["est"][0])
    assert not S.healthy()


# ── API protection ──────────────────────────────────────────────────────────

def test_api_key_required_when_configured(client, monkeypatch):
    import server
    monkeypatch.setattr(server, "API_KEY", "s3cret")
    assert client.get("/api/schema").status_code == 401
    assert client.get("/api/schema", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/api/health").status_code == 200  # liveness stays open


def test_rate_limit(client, monkeypatch):
    import server
    monkeypatch.setattr(server, "RATE_PER_MIN", 2)
    server._hits.clear()
    codes = [client.post("/api/run", json={"sql": "SELECT COUNT(*) FROM departments"}).status_code for _ in range(4)]
    assert codes[:2] == [200, 200] and codes[2:] == [429, 429]
    server._hits.clear()


def test_refuses_public_bind_without_key():
    import server
    assert server._is_loopback("127.0.0.1") and server._is_loopback("localhost")
    assert not server._is_loopback("0.0.0.0")


# ── feedback log ────────────────────────────────────────────────────────────

def test_feedback_stores_no_raw_sql_and_survives_bad_lines(tmp_path, engine):
    from src.estimator.service import Estimators
    S = Estimators("hr", feedback_path=str(tmp_path / "fb.jsonl"))
    q = engine.parse("SELECT COUNT(*) FROM employees WHERE city = 'Pune'")
    S.log_feedback(q, 100, {"factorjoin": 120.0})
    with open(S.feedback_path, "a", encoding="utf8") as f:
        f.write("{not json\n")
        f.write(json.dumps({"est": "x"}) + "\n")
    rec = json.loads(open(S.feedback_path, encoding="utf8").readline())
    assert "sql" not in rec and "Pune" not in json.dumps(rec) and len(rec["qhash"]) == 16
    summ = S.feedback_summary()
    assert summ["n"] == 1 and summ["skipped_malformed"] == 2
    assert np.isclose(summ["estimators"]["factorjoin"]["median"], 1.2)
