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

from typing import Dict, Iterable, Optional, Tuple

from .datasets import Dataset


STATEMENT_TIMEOUT_MS = 5000   # default for EXPLAIN; run_hinted sets its own per call
LOCK_TIMEOUT_MS = 2000


def dsn_for(base: str, ds: Dataset) -> str:
    """Same server and options (sslmode, host, user, ...) as `base`, but the dataset's
    database: HR lives in 'carddb', STATS in 'stats'."""
    from psycopg2.extensions import make_dsn, parse_dsn
    opts = parse_dsn(base)
    opts["dbname"] = "stats" if ds.name.startswith("stats") else "carddb"
    return make_dsn(**opts)


class LivePG:
    """One session with bounded statements and locks. Use as a context manager (or call
    close()) so the connection is released on every path."""

    def __init__(self, dsn: str):
        import psycopg2
        self.conn = psycopg2.connect(dsn, connect_timeout=5)
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        try:
            self.cur.execute("SET max_parallel_workers_per_gather = 0")
            self.cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
            self.cur.execute(f"SET lock_timeout = {LOCK_TIMEOUT_MS}")
        except Exception:
            self.conn.close()
            raise

    def close(self):
        if not self.conn.closed:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def server_info(self) -> dict:
        """Version and the planner settings that change estimates/plans (for provenance)."""
        self.cur.execute("SELECT version()")
        v = self.cur.fetchone()[0]
        self.cur.execute("SELECT name, setting FROM pg_settings WHERE name IN ('default_statistics_target', "
                         "'random_page_cost', 'work_mem', 'shared_buffers', 'max_parallel_workers_per_gather', "
                         "'join_collapse_limit', 'from_collapse_limit', 'geqo_threshold', 'effective_cache_size')")
        return {"version": v, "settings": dict(self.cur.fetchall())}

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
            if not self.conn.closed:
                self.cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")


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
