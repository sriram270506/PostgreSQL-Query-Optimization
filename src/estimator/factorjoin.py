"""FactorJoin (Wu et al., SIGMOD 2023) over per-table Chow-Liu BNs.

Join keys that can be equated (dept_id, emp_id, proj_id, city) share one binning.
For a join tree, each relation sends its parent a vector over the shared key bins:
    m_T(b) = N_T * P_BN(key = b, filters_T, soft evidence from T's own children)
where a child's message enters T's BN as soft evidence m_child(b) / ndv(b) on the child's
join-key node (a row with key x matches m(b)/ndv(b) child results when keys are uniform
inside bin b). The root's total mass is the estimate. Exact BN inference inside a table,
bin-level uniformity across tables: the FactorJoin approximation.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.engine.executor import JOIN_EDGES, Catalog
from src.engine.sql import Join, Query
from .bn import ChowLiuBN, Discretizer

KEY_BINS = 64


def _key_groups() -> List[set]:
    groups: List[set] = []
    for t1, c1, t2, c2 in JOIN_EDGES:
        a, b = (t1, c1), (t2, c2)
        hit = [g for g in groups if a in g or b in g]
        merged = set().union(*hit) | {a, b} if hit else {a, b}
        groups = [g for g in groups if g not in hit] + [merged]
    return groups


class FactorJoin:
    def __init__(self, catalog: Catalog):
        self.cat = catalog
        self.group_of: Dict[Tuple[str, str], int] = {}
        self.group_disc: List[Discretizer] = []
        self.group_ndv: List[np.ndarray] = []
        for gi, g in enumerate(_key_groups()):
            vals = np.concatenate([catalog.tables[t].cols[c] for t, c in g if t in catalog.tables])
            numeric = all(catalog.tables[t].numeric[c] for t, c in g if t in catalog.tables)
            d = Discretizer(vals, numeric, n_bins=KEY_BINS)
            uniq = np.unique(vals)
            self.group_disc.append(d)
            self.group_ndv.append(np.maximum(np.bincount(d.bin(uniq)[d.bin(uniq) >= 0], minlength=d.k), 1))
            for tc in g:
                self.group_of[tc] = gi
        self.bns: Dict[str, ChowLiuBN] = {}
        for t, tb in catalog.tables.items():
            kd = {c: self.group_disc[self.group_of[(t, c)]] for c in tb.cols if (t, c) in self.group_of}
            self.bns[t] = ChowLiuBN(tb, kd)

    # ── single relation ────────────────────────────────────────────────────
    def table_estimate(self, q: Query, alias: str) -> float:
        t = q.rels[alias]
        bn = self.bns[t]
        L, scale = bn.likelihood(q.preds_of(alias), self.cat.tables[t])
        return bn.n * bn.prob(L) * scale

    def selectivity(self, q: Query, alias: str) -> float:
        return self.table_estimate(q, alias) / max(self.bns[q.rels[alias]].n, 1)

    # ── join tree ───────────────────────────────────────────────────────────
    def _shared_group(self, q: Query, j: Join) -> Optional[int]:
        ga = self.group_of.get((q.rels[j.a], j.acol))
        gb = self.group_of.get((q.rels[j.b], j.bcol))
        return ga if ga is not None and ga == gb else None

    def _message(self, q: Query, alias: str, parent: Optional[Join], fallback: List[float]):
        t = q.rels[alias]
        bn = self.bns[t]
        L, scale = bn.likelihood(q.preds_of(alias), self.cat.tables[t])
        for j in q.joins:
            if j is parent or alias not in (j.a, j.b):
                continue
            mycol, child, _ = j.other(alias)
            g = self._shared_group(q, j)
            m = self._message(q, child, j, fallback)
            if g is None:
                # shortcut: join on columns outside a key group uses PG's 1/max(ndv); add the
                # pair to JOIN_EDGES to model it properly
                nd = max(len(np.unique(self.cat.tables[t].cols[mycol])),
                         len(np.unique(self.cat.tables[q.rels[child]].cols[j.other(child)[0]])))
                fallback.append(float(np.sum(m)) / nd)
                continue
            soft = m / self.group_ndv[g]
            L[mycol] = L[mycol] * soft if mycol in L else soft
        if parent is None:
            return bn.n * scale * bn.prob(L)
        mycol = parent.other(alias)[0]
        g = self._shared_group(q, parent)
        if g is None:  # parent joins us on an unmodelled column: send our total mass
            return np.array(bn.n * scale * bn.prob(L))
        return bn.n * scale * bn.belief(mycol, L)

    def estimate(self, q: Query) -> float:
        # split into connected components; the product covers cross joins
        remaining, total = set(q.rels), 1.0
        while remaining:
            root = next(a for a in q.rels if a in remaining)
            comp, stack = {root}, [root]
            while stack:
                x = stack.pop()
                for j in q.joins:
                    if x in (j.a, j.b):
                        y = j.b if x == j.a else j.a
                        if y not in comp:
                            comp.add(y)
                            stack.append(y)
            remaining -= comp
            fb: List[float] = []
            sub = q.sub(comp)
            if len(sub.joins) > len(comp) - 1:
                # shortcut: cyclic join graphs drop the cycle-closing edges
                keep, seen, grew = [], {root}, True
                while grew:
                    grew = False
                    for j in sub.joins:
                        if j not in keep and (j.a in seen) != (j.b in seen):
                            keep.append(j)
                            seen |= {j.a, j.b}
                            grew = True
                sub.joins = keep
            est = float(np.sum(self._message(sub, root, None, fb)))
            for f in fb:
                est *= f
            total *= est
        return max(1.0, total)

    def timed(self, q: Query) -> Tuple[float, float]:
        t0 = time.perf_counter()
        est = self.estimate(q)
        return est, (time.perf_counter() - t0) * 1000
