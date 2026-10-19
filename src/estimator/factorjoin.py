"""FactorJoin (Wu et al., SIGMOD 2023) over per-table Chow-Liu BNs.

Join keys that can be equated (e.g. every user-id column) share one binning: heavy-hitter
keys get their own bin, the rest equal-row-mass ranges. For a join tree each relation
sends its parent a vector over the shared key bins:
    m_T(b) = N_T * P_BN(key = b, filters_T, soft evidence from T's own children)
A child's message enters T's BN as soft evidence m_child(b) / ndv(b) on the child's
join-key node (a row with key x matches m(b)/ndv(b) child results when keys are uniform
inside bin b). The root's total mass is the estimate: exact BN inference inside a table,
bin-level uniformity across tables.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.engine.executor import Catalog
from src.engine.sql import Join, Query
from .bn import ChowLiuBN, Discretizer

CACHE_SIZE = 50_000


def _key_groups(edges) -> List[set]:
    groups: List[set] = []
    for t1, c1, t2, c2 in edges:
        a, b = (t1, c1), (t2, c2)
        hit = [g for g in groups if a in g or b in g]
        merged = set().union(*hit) | {a, b} if hit else {a, b}
        groups = [g for g in groups if g not in hit] + [merged]
    return groups


class FactorJoin:
    def __init__(self, catalog: Catalog):
        self.cat = catalog
        ds = catalog.ds
        self.group_of: Dict[Tuple[str, str], int] = {}
        self.group_disc: List[Discretizer] = []
        self.group_ndv: List[np.ndarray] = []
        for gi, g in enumerate(_key_groups(ds.join_edges)):
            members = [(t, c) for t, c in sorted(g) if t in catalog.tables]
            vals = np.concatenate([catalog.tables[t].cols[c] for t, c in members])
            numeric = all(catalog.tables[t].numeric[c] for t, c in members)
            d = Discretizer.fit_key(vals, numeric)
            self.group_disc.append(d)
            self.group_ndv.append(d.ndv())
            for tc in g:
                self.group_of[tc] = gi
        self.bns: Dict[str, ChowLiuBN] = {}
        for t, tb in catalog.tables.items():
            kd = {c: self.group_disc[self.group_of[(t, c)]] for c in tb.cols if (t, c) in self.group_of}
            self.bns[t] = ChowLiuBN(tb, ds.bn_columns(t), kd)
        self._cache: "OrderedDict[str, float]" = OrderedDict()
        self._lcache: "OrderedDict[tuple, tuple]" = OrderedDict()

    # ── single relation ────────────────────────────────────────────────────
    def _likelihood(self, q: Query, alias: str):
        t = q.rels[alias]
        key = (t, tuple(sorted(p.to_sql().split(".", 1)[1] for p in q.preds_of(alias))))
        hit = self._lcache.get(key)
        if hit is None:
            hit = self.bns[t].likelihood(q.preds_of(alias), self.cat.tables[t])
            self._lcache[key] = hit
            if len(self._lcache) > CACHE_SIZE:
                self._lcache.popitem(last=False)
        L, scale = hit
        return dict(L), scale  # callers add soft evidence; never mutate the cached copy

    def table_estimate(self, q: Query, alias: str) -> float:
        bn = self.bns[q.rels[alias]]
        L, scale = self._likelihood(q, alias)
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
        L, scale = self._likelihood(q, alias)
        for j in q.joins:
            if j is parent or alias not in (j.a, j.b):
                continue
            mycol, child, _ = j.other(alias)
            g = self._shared_group(q, j)
            m = self._message(q, child, j, fallback)
            if g is None or mycol not in bn.disc:
                # shortcut: join on columns outside a key group uses PG's 1/max(ndv); add the
                # pair to the dataset's join_edges to model it properly
                nd = max(len(np.unique(self.cat.tables[t].cols[mycol])),
                         len(np.unique(self.cat.tables[q.rels[child]].cols[j.other(child)[0]])))
                fallback.append(float(np.sum(m)) / nd)
                continue
            soft = m / self.group_ndv[g]
            if bn.disc[mycol].null_bin:
                soft = np.append(soft, 0.0)  # NULL keys never join
            L[mycol] = L[mycol] * soft if mycol in L else soft
        if parent is None:
            return bn.n * scale * bn.prob(L)
        mycol = parent.other(alias)[0]
        g = self._shared_group(q, parent)
        if g is None or mycol not in bn.disc:  # parent joins us on an unmodelled column
            return np.array(bn.n * scale * bn.prob(L))
        belief = bn.n * scale * bn.belief(mycol, L)
        return belief[: bn.disc[mycol].k_value]  # drop the NULL bin

    def estimate(self, q: Query) -> float:
        q = q.canonical()  # same root and message order however the query was written
        key = q.key() + "|" + ";".join(sorted(j.to_sql() for j in q.joins))
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        est = self._estimate(q)
        self._cache[key] = est
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return est

    def _estimate(self, q: Query) -> float:
        remaining, total = set(q.rels), 1.0
        while remaining:  # connected components multiply (cross joins)
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
