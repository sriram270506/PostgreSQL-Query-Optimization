"""Random query workload over the schema join graph.

Each query: a random connected, acyclic subset of JOIN_EDGES (1-5 relations), 1-2
predicates on most relations (=, <>, IN, <, >, BETWEEN with widths from 2% to 60% of the
domain), literals drawn from the real data so selectivities span many orders of magnitude.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np

from src.engine.executor import JOIN_EDGES, Catalog
from src.engine.sql import Join, Pred, Query

ALIAS = {"departments": "d", "employees": "e", "projects": "p",
         "assignments": "a", "locations": "l", "budgets": "b"}
FILTERS = {
    "departments": ["dept_name", "region"],
    "employees": ["age", "experience_years", "salary", "city", "dept_id"],
    "projects": ["priority", "budget_tier", "dept_id"],
    "assignments": ["hours_per_week"],
    "locations": ["city", "office_tier"],
    "budgets": ["approved_amount"],
}
SIZE_WEIGHTS = {1: 0.2, 2: 0.3, 3: 0.25, 4: 0.15, 5: 0.1}


def category(n_rels: int) -> str:
    return "1-table" if n_rels == 1 else f"{n_rels}-way join" if n_rels < 4 else "4+ way join"


class WorkloadGenerator:
    def __init__(self, catalog: Catalog, seed: int = 42):
        self.cat = catalog
        self.rng = random.Random(seed)
        self.np = np.random.default_rng(seed)

    def _join_tree(self, size: int):
        rels = [self.rng.choice(list(self.cat.tables))]
        joins: List[Join] = []
        while len(rels) < size:
            frontier = [e for e in JOIN_EDGES if (e[0] in rels) != (e[2] in rels)]
            if not frontier:
                break
            t1, c1, t2, c2 = self.rng.choice(frontier)
            rels.append(t2 if t1 in rels else t1)
            joins.append(Join(ALIAS[t1], c1, ALIAS[t2], c2))
        return rels, joins

    def _pred(self, table: str, alias: str, col: str) -> Pred:
        tb = self.cat.tables[table]
        vals = tb.cols[col]
        uniq = np.unique(vals)
        r = self.rng.random()
        if not tb.numeric[col] or len(uniq) <= 10:
            pick = lambda: _py(self.rng.choice(list(uniq)))  # noqa: E731
            if r < 0.55 or len(uniq) < 3:
                return Pred(alias, col, "eq", pick())
            if r < 0.8:
                k = self.rng.randint(2, min(3, len(uniq) - 1))
                return Pred(alias, col, "in", tuple(sorted({pick() for _ in range(k)}, key=str)))
            if tb.numeric[col]:
                return Pred(alias, col, self.rng.choice(["lt", "gt", "le", "ge"]), pick())
            return Pred(alias, col, "ne", pick())
        lo_v, hi_v = float(vals.min()), float(vals.max())
        q = float(np.quantile(vals, self.rng.random()))
        if r < 0.6:
            width = (hi_v - lo_v) * float(np.exp(self.np.uniform(np.log(0.02), np.log(0.6))))
            return Pred(alias, col, "between", (int(q), int(min(q + width, hi_v))))
        return Pred(alias, col, self.rng.choice(["lt", "le", "gt", "ge"]), int(q))

    def query(self, size: Optional[int] = None) -> Query:
        size = size or self.rng.choices(list(SIZE_WEIGHTS), weights=list(SIZE_WEIGHTS.values()))[0]
        rels, joins = self._join_tree(size)
        preds: List[Pred] = []
        for t in rels:
            if self.rng.random() < 0.7 or (t == rels[-1] and not preds):
                cols = self.rng.sample(FILTERS[t], k=min(len(FILTERS[t]), self.rng.choice([1, 1, 2])))
                preds += [self._pred(t, ALIAS[t], c) for c in cols]
        return Query(rels={ALIAS[t]: t for t in rels}, joins=joins, preds=preds)


def _py(v):
    if isinstance(v, (np.floating, float)):
        return int(v) if float(v).is_integer() else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return str(v)
