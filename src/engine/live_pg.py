"""Talk to a real PostgreSQL server: estimates, extended statistics, hinted execution.

* explain_rows: the planner's row estimate for a query (parallelism off, so the top
  join node carries the whole-query estimate rather than a per-worker share).
* extended statistics: CREATE STATISTICS (ndistinct, dependencies, mcv) on each table's
  filter columns: PostgreSQL's own fix for correlated predicates, the strongest
  non-learned baseline.
* run_hinted: execute a query with pg_hint_plan Rows() hints that overwrite the planner's
  join cardinalities, to measure real runtime of the plan each estimator leads to.
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, Optional, Tuple

from .datasets import Dataset


def dsn_for(base: str, ds: Dataset) -> str:
    """base DSN points at any database; HR lives in 'carddb', STATS in 'stats'."""
    db = "stats" if ds.name.startswith("stats") else "carddb"
    return base.rsplit("/", 1)[0] + "/" + db


class LivePG:
    def __init__(self, dsn: str):
        import psycopg2
        self.conn = psycopg2.connect(dsn, connect_timeout=5)
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        self.cur.execute("SET max_parallel_workers_per_gather = 0")

    def close(self):
        self.conn.close()

    def explain_rows(self, sql: str) -> float:
        self.cur.execute("EXPLAIN (FORMAT JSON) " + sql)
        node = self.cur.fetchone()[0][0]["Plan"]
        while node.get("Node Type") == "Aggregate" and node.get("Plans"):
            node = node["Plans"][0]  # COUNT(*) aggregates to 1 row; the estimate is its input's
        return float(node["Plan Rows"])

    # ── extended statistics ─────────────────────────────────────────────────
    def create_extended_stats(self, ds: Dataset):
        for t, cols in ds.filters.items():
            if len(cols) < 2:
                continue
            self.cur.execute(f"DROP STATISTICS IF EXISTS xst_{t}")
            self.cur.execute(f"CREATE STATISTICS xst_{t} (ndistinct, dependencies, mcv) ON "
                             f"{', '.join(cols[:8])} FROM {t}")
        self.cur.execute("ANALYZE")

    def drop_extended_stats(self, ds: Dataset):
        for t in ds.filters:
            self.cur.execute(f"DROP STATISTICS IF EXISTS xst_{t}")
        self.cur.execute("ANALYZE")

    # ── hinted execution (pg_hint_plan) ─────────────────────────────────────
    def run_hinted(self, sql: str, rows: Optional[Dict[Tuple[str, ...], float]],
                   timeout_s: float = 30.0) -> Tuple[Optional[float], Optional[str]]:
        """Execute `sql` with Rows(alias... #n) hints for every given join; returns
        (execution ms, plan join order) or (None, None) on timeout."""
        self.cur.execute("LOAD 'pg_hint_plan'")
        self.cur.execute(f"SET statement_timeout = {int(timeout_s * 1000)}")
        hint = ""
        if rows:
            hint = "/*+ " + " ".join(f"Rows({' '.join(r)} #{max(int(round(n)), 1)})" for r, n in rows.items()) + " */ "
        try:
            self.cur.execute("EXPLAIN (ANALYZE, FORMAT JSON, TIMING OFF) " + hint + sql)
            res = self.cur.fetchone()[0][0]
            return float(res["Execution Time"]), _join_order(res["Plan"])
        except Exception as e:  # statement_timeout -> QueryCanceled
            if "statement timeout" in str(e) or "canceling statement" in str(e):
                return None, None
            raise
        finally:
            self.cur.execute("SET statement_timeout = 0")


def _join_order(node: dict) -> str:
    if "Plans" not in node:
        return node.get("Alias", "")
    kids = [_join_order(k) for k in node["Plans"]]
    kids = [k for k in kids if k]
    if node["Node Type"] in ("Hash Join", "Merge Join", "Nested Loop") and len(kids) == 2:
        return f"({kids[0]} ⋈ {kids[1]})"
    return kids[0] if kids else ""


def hint_aliases(subsets: Iterable[frozenset]) -> list:
    return [tuple(sorted(s)) for s in subsets]
