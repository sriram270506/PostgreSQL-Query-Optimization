"""Random query workload over a dataset's join graph.

Each query: a random connected, acyclic subset of the dataset's join edges (1-5
relations), 1-2 predicates on most relations (=, <>, IN, <, >, BETWEEN, or a >=/<= pair
as STATS-CEB writes ranges), literals drawn from the real data so selectivities span many
orders of magnitude. Each query also records its "template" (relations + filtered
columns) so experiments can hold whole templates out.
"""
from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
import pandas as pd

from src.engine.executor import Catalog
from src.engine.sql import Join, Pred, Query

SIZE_WEIGHTS = {1: 0.2, 2: 0.3, 3: 0.25, 4: 0.15, 5: 0.1}


def category(n_rels: int) -> str:
    return "1-table" if n_rels == 1 else f"{n_rels}-way join" if n_rels < 4 else "4+ way join"


def template(q: Query) -> str:
    """Relations + filtered columns, ignoring literals and operators."""
    return "|".join(sorted(q.rels.values())) + "#" + ",".join(sorted({f"{q.rels[p.alias]}.{p.col}" for p in q.preds}))


class WorkloadGenerator:
    def __init__(self, catalog: Catalog, seed: int = 42, sizes=None):
        self.cat = catalog
        self.ds = catalog.ds
        self.rng = random.Random(seed)
        self.np = np.random.default_rng(seed)
        self.sizes = sizes or SIZE_WEIGHTS

    def _join_tree(self, size: int):
        rels = [self.rng.choice(list(self.cat.tables))]
        joins: List[Join] = []
        A = self.ds.aliases
        while len(rels) < size:
            frontier = [e for e in self.ds.join_edges if (e[0] in rels) != (e[2] in rels)]
            if not frontier:
                break
            t1, c1, t2, c2 = self.rng.choice(frontier)
            rels.append(t2 if t1 in rels else t1)
            joins.append(Join(A[t1], c1, A[t2], c2))
        return rels, joins

    def _pred(self, table: str, alias: str, col: str) -> List[Pred]:
        tb = self.cat.tables[table]
        vals = tb.cols[col]
        vals = vals[np.asarray(pd.notna(vals))]
        ts = col in self.ds.timestamps.get(table, [])
        uniq = np.unique(vals)
        r = self.rng.random()
        P = lambda op, v: Pred(alias, col, op, v, ts)  # noqa: E731
        if not tb.numeric[col] or len(uniq) <= 10:
            pick = lambda: _py(self.rng.choice(list(uniq)))  # noqa: E731
            if r < 0.55 or len(uniq) < 3:
                return [P("eq", pick())]
            if r < 0.8:
                k = self.rng.randint(2, min(3, len(uniq) - 1))
                return [P("in", tuple(sorted({pick() for _ in range(k)}, key=str)))]
            if tb.numeric[col]:
                return [P(self.rng.choice(["lt", "gt", "le", "ge"]), pick())]
            return [P("ne", pick())]
        lo_v, hi_v = float(vals.min()), float(vals.max())
        q = float(np.quantile(vals, self.rng.random()))
        if r < 0.6:
            width = (hi_v - lo_v) * float(np.exp(self.np.uniform(np.log(0.02), np.log(0.6))))
            lo, hi = int(q), int(min(q + width, hi_v))
            if self.rng.random() < 0.5:  # STATS-CEB style: two one-sided predicates
                return [P("ge", lo), P("le", hi)]
            return [P("between", (lo, hi))]
        return [P(self.rng.choice(["lt", "le", "gt", "ge"]), int(q))]

    def query(self, size: Optional[int] = None) -> Query:
        size = size or self.rng.choices(list(self.sizes), weights=list(self.sizes.values()))[0]
        rels, joins = self._join_tree(size)
        A = self.ds.aliases
        preds: List[Pred] = []
        for t in rels:
            cols = self.ds.filters.get(t, [])
            if cols and (self.rng.random() < 0.7 or (t == rels[-1] and not preds)):
                pick = self.rng.sample(cols, k=min(len(cols), self.rng.choice([1, 1, 2])))
                for c in pick:
                    preds += self._pred(t, A[t], c)
        return Query(rels={A[t]: t for t in rels}, joins=joins, preds=preds)


def _py(v):
    if isinstance(v, (np.floating, float)):
        return int(v) if float(v).is_integer() else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return str(v)
