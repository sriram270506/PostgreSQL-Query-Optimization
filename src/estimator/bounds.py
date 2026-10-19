"""Provable cardinality upper bounds (no data scan at query time).

For an acyclic join rooted at relation r, every result tuple is one row of r extended,
edge by edge, by at most maxdeg(child.key) child rows, where maxdeg is the largest number
of rows sharing one key value in the unfiltered child table:
    |Q| <= |T_r| * prod_{edges r->...->c} maxdeg_c(key_c)
Minimised over all roots. Filters only remove rows, so the bound holds for every filter.
Any estimate above it is impossible and gets clamped.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.engine.executor import Catalog
from src.engine.sql import Query


class Bounds:
    def __init__(self, catalog: Catalog):
        self.cat = catalog
        self.n = {t: tb.n for t, tb in catalog.tables.items()}
        self.maxdeg: Dict[Tuple[str, str], int] = {}

    def _deg(self, table: str, col: str) -> int:
        key = (table, col)
        if key not in self.maxdeg:
            v = self.cat.tables[table].cols[col]
            v = v[np.asarray(pd.notna(v))]
            self.maxdeg[key] = int(pd.Series(v).value_counts().max()) if len(v) else 0
        return self.maxdeg[key]

    def upper(self, q: Query) -> float:
        best = np.inf
        for root in q.rels:
            seen, stack, bound = {root}, [root], float(self.n[q.rels[root]])
            while stack:
                x = stack.pop()
                for j in q.joins:
                    if x not in (j.a, j.b):
                        continue
                    _, y, ycol = j.other(x)
                    if y in seen:
                        continue
                    seen.add(y)
                    stack.append(y)
                    bound *= self._deg(q.rels[y], ycol)
            for a in q.rels:  # disconnected parts multiply (cross product)
                if a not in seen:
                    bound *= self.n[q.rels[a]]
            best = min(best, bound)
        return float(best)

    def clamp(self, q: Query, est: float) -> Tuple[float, bool]:
        ub = self.upper(q)
        return (max(1.0, min(est, ub)), est > ub)
