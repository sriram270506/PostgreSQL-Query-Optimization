"""Embedded SQL engine: parse -> exact execution + PostgreSQL-style planner estimates."""
import time
from dataclasses import replace
from typing import Any, Dict, Optional, Union

from . import datasets
from .datasets import Dataset
from .executor import Catalog, Executor
from .pg_planner import PGPlanner
from .sql import Join, Pred, Query, SQLError, parse

__all__ = ["Engine", "Query", "Pred", "Join", "SQLError", "Dataset"]


class Engine:
    def __init__(self, dataset: Union[str, Dataset] = "hr"):
        self.ds = datasets.get(dataset) if isinstance(dataset, str) else dataset
        self.catalog = Catalog(self.ds)
        self.executor = Executor(self.catalog)
        self.planner = PGPlanner(self.catalog)

    @property
    def schema(self):
        return self.catalog.schema

    def parse(self, sql: str) -> Query:
        q = parse(sql, self.schema)
        ts = self.ds.timestamps
        q.preds = [replace(p, ts=p.col in ts.get(q.rels[p.alias], [])) for p in q.preds]
        return q

    def true_count(self, q: Query) -> int:
        return self.executor.count(q)[0]

    def run(self, sql: str, row_limit: Optional[int] = None) -> Dict[str, Any]:
        """Execute `sql`: exact count, fetched rows, per-relation actual vs planner rows."""
        q = self.parse(sql)
        t0 = time.perf_counter()
        count, filtered, subtree = self.executor.count(q)
        count_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        frame = None if q.count else self.executor.fetch(q, row_limit)
        fetch_ms = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        pg_est = self.planner.estimate(q.sub(q.rels))
        pg_ms = (time.perf_counter() - t2) * 1000
        return {
            "query": q, "count": count, "count_ms": count_ms,
            "rows": frame, "fetch_ms": fetch_ms,
            "pg_estimate": pg_est, "pg_ms": pg_ms,
            "scan": {a: {"table": t, "actual": filtered[a], "pg": self.planner.rel_rows(q, a),
                         "total": self.catalog.tables[t].n} for a, t in q.rels.items()},
            "subtree": subtree,
        }
