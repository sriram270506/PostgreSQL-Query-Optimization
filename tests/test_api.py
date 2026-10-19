"""API contract tests: response shapes the UI reads, and 4xx (never 500) on bad input."""
import pytest

from conftest import requires_data

pytestmark = requires_data

pytest.importorskip("httpx", reason="httpx required by TestClient")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import server
    with TestClient(server.app) as c:
        yield c


def test_health(client):
    body = client.get("/api/health").json()
    assert "factorjoin" in body["estimators"] and "postgres" not in body["estimators"]  # emulation never served
    assert body["tables"]["employees"] > 0


def test_run_count_query_scores_every_estimator(client):
    r = client.post("/api/run", json={"sql": "SELECT COUNT(*) FROM employees e JOIN projects p ON e.dept_id = p.dept_id "
                                              "JOIN assignments a ON e.emp_id = a.emp_id WHERE e.city = 'Pune'"})
    assert r.status_code == 200
    b = r.json()
    assert b["count"] > 0 and b["rows"] is None
    names = {e["name"] for e in b["estimators"]}
    assert "factorjoin" in names and "postgres" not in names and b["best"] in names
    assert all(e["qerror"] >= 1 for e in b["estimators"])
    assert b["plans"]["optimal_cost"] > 0
    assert all(v["ratio"] >= 1 - 1e-9 for v in b["plans"]["estimators"].values())
    assert all(e["estimate"] <= b["upper_bound"] + 1e-6 for e in b["estimators"] if e["name"] != "postgres_live")
    assert b["count"] <= b["upper_bound"]


def test_run_select_returns_rows(client):
    b = client.post("/api/run", json={"sql": "SELECT * FROM departments", "row_limit": 3}).json()
    assert b["rows"]["shown"] == 3 and b["count"] == 6 and b["rows"]["truncated"]


@pytest.mark.parametrize("sql", ["DROP TABLE employees", "SELECT COUNT(*) FROM ghosts", "SELECT 1; DROP TABLE x"])
def test_bad_sql_is_400(client, sql):
    r = client.post("/api/run", json={"sql": sql})
    assert r.status_code == 400 and r.json()["detail"]


def test_tables_and_bn(client):
    assert client.get("/api/tables/employees?page=2&page_size=5").json()["page"] == 2
    assert client.get("/api/tables/ghosts").status_code == 404
    bn = client.get("/api/bn/employees").json()
    assert len(bn["edges"]) <= len(bn["nodes"]) - 1  # a forest (G-test may leave columns unlinked)
    for e in bn["edges"]:
        for row in e["cpt"]:
            assert abs(sum(row) - 1) <= len(row) * 5e-5 + 1e-9  # API rounds each cell to 4 decimals


def test_unknown_dataset_is_404(client):
    assert client.get("/api/health?dataset=nope").status_code == 404


def test_feedback_is_logged(client):
    client.post("/api/run", json={"sql": "SELECT COUNT(*) FROM employees WHERE city = 'Pune'"})
    fb = client.get("/api/feedback").json()
    assert fb["n"] >= 1 and "factorjoin" in fb["estimators"]
