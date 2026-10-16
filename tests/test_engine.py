"""Engine correctness: parser, exact counts vs an independent pandas oracle, row fetch,
planner emulation and the join-order optimizer."""
import numpy as np
import pytest

from conftest import requires_data

pytestmark = requires_data

from src.engine import SQLError, optimizer  # noqa: E402


def test_parse_aliases_joins_and_predicates(engine):
    q = engine.parse("SELECT COUNT(*) FROM employees e JOIN projects p ON e.dept_id = p.dept_id "
                     "WHERE e.city = 'Pune' AND p.budget_tier IN (1, 2) AND 30 < e.age")
    assert q.rels == {"e": "employees", "p": "projects"}
    assert [(j.a, j.acol, j.b, j.bcol) for j in q.joins] == [("e", "dept_id", "p", "dept_id")]
    ops = {(p.alias, p.col, p.op) for p in q.preds}
    assert ops == {("e", "city", "eq"), ("p", "budget_tier", "in"), ("e", "age", "gt")}  # literal-first flipped


@pytest.mark.parametrize("sql,msg", [
    ("SELECT COUNT(*) FROM nope", "does not exist"),
    ("SELECT COUNT(*) FROM employees WHERE city = 'A' OR age = 3", "OR"),
    ("SELECT COUNT(*) FROM employees e JOIN projects p ON e.dept_id = p.dept_id WHERE dept_id = 1", "ambiguous"),
    ("SELEC nonsense", "SELECT|syntax"),
    ("SELECT COUNT(*) FROM employees WHERE age = 'old'", "invalid input"),
])
def test_bad_sql_raises_sql_error(engine, sql, msg):
    with pytest.raises(SQLError, match=msg):
        engine.true_count(engine.parse(sql))


def test_to_sql_round_trips(engine):
    sql = ("SELECT COUNT(*) FROM projects p JOIN assignments a ON p.proj_id = a.proj_id "
           "JOIN employees e ON a.emp_id = e.emp_id WHERE p.priority = 'High' AND e.salary BETWEEN 50000 AND 90000")
    q = engine.parse(sql)
    assert engine.true_count(engine.parse(q.to_sql())) == engine.true_count(q)


def test_counts_match_pandas_oracle(engine, tables):
    e, p, a, b = tables["employees"], tables["projects"], tables["assignments"], tables["budgets"]
    sql = ("SELECT COUNT(*) FROM employees e JOIN projects p ON e.dept_id = p.dept_id "
           "JOIN assignments a ON e.emp_id = a.emp_id JOIN budgets b ON p.proj_id = b.proj_id "
           "WHERE e.city = 'Bangalore' AND p.budget_tier = 5 AND e.salary BETWEEN 82009 AND 142009")
    ee = e[(e.city == "Bangalore") & e.salary.between(82009, 142009)]
    m = ee.merge(p[p.budget_tier == 5], on="dept_id").merge(a, on="emp_id").merge(b, left_on="proj_id_x", right_on="proj_id")
    assert engine.true_count(engine.parse(sql)) == len(m)
    l = tables["locations"]
    assert engine.true_count(engine.parse("SELECT COUNT(*) FROM locations WHERE office_tier <> 2")) == int((l.office_tier != 2).sum())


def test_cyclic_join_matches_acyclic_semantics(engine, tables):
    # d-e, d-p and e-p on dept_id form a cycle; the extra edge is implied, so counts agree
    cyc = ("SELECT COUNT(*) FROM departments d JOIN employees e ON d.dept_id = e.dept_id "
           "JOIN projects p ON d.dept_id = p.dept_id WHERE e.dept_id = p.dept_id AND e.city = 'Pune' AND p.priority = 'Low'")
    tree = ("SELECT COUNT(*) FROM departments d JOIN employees e ON d.dept_id = e.dept_id "
            "JOIN projects p ON d.dept_id = p.dept_id WHERE e.city = 'Pune' AND p.priority = 'Low'")
    assert engine.true_count(engine.parse(cyc)) == engine.true_count(engine.parse(tree))


def test_fetch_rows_are_real_join_results(engine, tables):
    r = engine.run("SELECT e.emp_id, e.city, a.hours_per_week FROM employees e JOIN assignments a ON e.emp_id = a.emp_id "
                   "WHERE e.city = 'Delhi' AND a.hours_per_week > 40 LIMIT 25")
    rows = r["rows"]
    assert len(rows) == 25 and r["count"] > 25
    assert (rows["e.city"] == "Delhi").all() and (rows["a.hours_per_week"] > 40).all()
    a = tables["assignments"]
    for emp, h in zip(rows["e.emp_id"], rows["a.hours_per_week"]):
        assert ((a.emp_id == emp) & (a.hours_per_week == h)).any()


def test_planner_emulation_basics(engine, tables):
    # small categorical domain -> every value is an MCV, so equality estimates are exact
    l = tables["locations"]
    assert engine.planner.estimate(engine.parse("SELECT COUNT(*) FROM locations WHERE office_tier = 2")) == (l.office_tier == 2).sum()
    # independence assumption: Engineering x Bangalore is badly under-estimated
    q = engine.parse("SELECT COUNT(*) FROM employees WHERE dept_id = 1 AND city = 'Bangalore'")
    assert engine.planner.estimate(q) < engine.true_count(q) / 2


def test_optimizer_true_plan_is_never_beaten(engine):
    q = engine.parse("SELECT COUNT(*) FROM employees e JOIN projects p ON e.dept_id = p.dept_id "
                     "JOIN assignments a ON e.emp_id = a.emp_id JOIN budgets b ON p.proj_id = b.proj_id WHERE e.city = 'Pune'")
    subsets = optimizer.connected_subsets(q)
    true = {s: float(engine.true_count(q.sub(s))) for s in subsets}
    opt, _ = optimizer.best_plan(q, true)
    pg = {s: engine.planner.estimate(q.sub(s)) for s in subsets}
    assert optimizer.plan_true_cost(optimizer.best_plan(q, pg)[1], true) >= opt - 1e-6


def test_factorjoin_close_on_correlated_join(engine, fj):
    q = engine.parse("SELECT COUNT(*) FROM employees e JOIN projects p ON e.dept_id = p.dept_id WHERE e.city = 'Mumbai' AND p.budget_tier = 5")
    true = engine.true_count(q)
    assert abs(np.log(fj.estimate(q) / true)) < np.log(1.2)
    assert abs(np.log(fj.estimate(q) / true)) < abs(np.log(engine.planner.estimate(q) / true))
