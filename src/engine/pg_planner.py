"""PostgreSQL planner row-estimate emulation (no server needed).

Reproduces the pieces of PostgreSQL that decide `rows=` in EXPLAIN for this query class:
  * ANALYZE (analyze.c compute_scalar_stats): null_frac, n_distinct, MCV list
    (values > 1.25x average frequency, up to default_statistics_target=100), and a
    100-bucket equi-depth histogram over the non-MCV values.
  * eqsel / neqsel / scalarineqsel (selfuncs.c): MCV hit, else (1-sumMCV-nullfrac)/(nd-nMCV);
    range = MCV part + linear-interpolated histogram part.
  * clauselist_selectivity: independence product, with range pairs (BETWEEN) combined
    as s_lo + s_hi - 1.
  * scalararraysel: IN (...) as a sum of disjoint equality selectivities.
  * eqjoinsel_inner: MCV-matching join selectivity, else (1-nf1)(1-nf2)/max(nd1, nd2).
  * Join size = product of base-rel rows x product of join selectivities (clamp_row_est).
Correlations are ignored, exactly as in stock PostgreSQL without CREATE STATISTICS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .executor import Catalog
from .sql import Pred, Query

STATS_TARGET = 100
DEFAULT_EQ_SEL = 0.005
DEFAULT_INEQ_SEL = 1.0 / 3.0
DEFAULT_RANGE_INEQ_SEL = 0.005


@dataclass
class ColStats:
    numeric: bool
    null_frac: float
    n_distinct: float
    mcv_vals: np.ndarray
    mcv_freqs: np.ndarray
    hist: np.ndarray  # histogram bounds over non-MCV values (sorted); empty if none


def analyze_column(values: np.ndarray, numeric: bool) -> ColStats:
    n = len(values)
    if numeric:
        nn = values[~np.isnan(values)]
    else:
        nn = values[values != None]  # noqa: E711
    null_frac = 1.0 - len(nn) / max(n, 1)
    vals, counts = np.unique(nn, return_counts=True)
    nd = len(vals)
    if nd <= STATS_TARGET and (counts > 1).all():
        mcv_idx = np.argsort(-counts, kind="stable")  # all values fit: keep every one
    else:
        avg = len(nn) / max(nd, 1)
        mincount = max(avg * 1.25, 2)
        order = np.argsort(-counts, kind="stable")[:STATS_TARGET]
        mcv_idx = order[counts[order] >= mincount]
    mcv_vals, mcv_freqs = vals[mcv_idx], counts[mcv_idx] / max(n, 1)
    hist = np.array([])
    if numeric:
        rest = nn[~np.isin(nn, mcv_vals)]
        if len(np.unique(rest)) >= 2:
            rest = np.sort(rest)
            nb = min(STATS_TARGET, len(rest) - 1)
            hist = rest[np.linspace(0, len(rest) - 1, nb + 1).round().astype(int)]
    return ColStats(numeric, null_frac, float(nd), mcv_vals, mcv_freqs, hist)


class PGPlanner:
    def __init__(self, catalog: Catalog):
        self.cat = catalog
        self.stats: Dict[str, Dict[str, ColStats]] = {
            t: {c: analyze_column(tb.cols[c], tb.numeric[c]) for c in tb.cols}
            for t, tb in catalog.tables.items()
        }
        self.reltuples = {t: tb.n for t, tb in catalog.tables.items()}

    # ── restriction selectivity ─────────────────────────────────────────────
    def _eqsel(self, s: ColStats, v) -> float:
        if s.numeric:
            try:
                v = float(v)
            except (TypeError, ValueError):
                return 0.0
        else:
            v = str(v)
        hit = np.flatnonzero(s.mcv_vals == v)
        if len(hit):
            return float(s.mcv_freqs[hit[0]])
        others = s.n_distinct - len(s.mcv_vals)
        if others <= 0:
            return 0.0  # value is not among the full MCV list: PG assumes ~0 rows
        sel = (1.0 - s.mcv_freqs.sum() - s.null_frac) / others
        return float(min(max(sel, 0.0), 1.0))

    def _ineqsel(self, s: ColStats, op: str, v) -> float:
        if not s.numeric:
            return DEFAULT_INEQ_SEL
        v = float(v)
        lt = op in ("lt", "le")
        incl = op in ("le", "ge")
        # MCV part
        if lt:
            m = (s.mcv_vals < v) | (incl & (s.mcv_vals == v))
        else:
            m = (s.mcv_vals > v) | (incl & (s.mcv_vals == v))
        mcv_sel = float(s.mcv_freqs[m].sum())
        rest = 1.0 - s.mcv_freqs.sum() - s.null_frac
        if len(s.hist) < 2:
            return mcv_sel
        h = s.hist
        if v <= h[0]:
            frac = 0.0
        elif v >= h[-1]:
            frac = 1.0
        else:  # ineq_histogram_selectivity: bucket position + linear interpolation
            i = int(np.searchsorted(h, v, "right")) - 1
            lo, hi = h[i], h[i + 1]
            within = (v - lo) / (hi - lo) if hi > lo else 0.5
            frac = (i + within) / (len(h) - 1)
        hsel = frac if lt else 1.0 - frac
        return mcv_sel + hsel * rest

    def pred_sel(self, table: str, p: Pred) -> float:
        s = self.stats[table][p.col]
        if p.op == "eq":
            return self._eqsel(s, p.value)
        if p.op == "ne":
            return max(0.0, 1.0 - self._eqsel(s, p.value) - s.null_frac)
        if p.op == "in":
            return min(1.0, sum(self._eqsel(s, v) for v in p.value))
        if p.op == "between":
            return self._range(s, p.value[0], p.value[1])
        return self._ineqsel(s, p.op, p.value)

    def _range(self, s: ColStats, lo, hi) -> float:
        if not s.numeric:
            return DEFAULT_RANGE_INEQ_SEL
        sel = self._ineqsel(s, "ge", lo) + self._ineqsel(s, "le", hi) - 1.0
        if sel > 0:
            return sel
        return DEFAULT_RANGE_INEQ_SEL if sel < -0.01 else 1e-10  # as in clauselist_selectivity

    def rel_rows(self, q: Query, alias: str) -> float:
        t = q.rels[alias]
        sel = 1.0
        for p in q.preds_of(alias):
            sel *= self.pred_sel(t, p)
        return max(1.0, round(self.reltuples[t] * sel))  # clamp_row_est

    # ── join selectivity ────────────────────────────────────────────────────
    def join_sel(self, t1: str, c1: str, t2: str, c2: str) -> float:
        s1, s2 = self.stats[t1][c1], self.stats[t2][c2]
        nd1, nd2 = s1.n_distinct, s2.n_distinct
        if len(s1.mcv_vals) and len(s2.mcv_vals) and s1.numeric == s2.numeric:
            match = np.isin(s1.mcv_vals, s2.mcv_vals)
            match2 = np.isin(s2.mcv_vals, s1.mcv_vals)
            f2 = dict(zip(s2.mcv_vals.tolist(), s2.mcv_freqs.tolist()))
            matchprod = sum(f * f2[v] for v, f in zip(s1.mcv_vals[match].tolist(), s1.mcv_freqs[match]))
            mf1, mf2 = s1.mcv_freqs[match].sum(), s2.mcv_freqs[match2].sum()
            unm1, unm2 = s1.mcv_freqs.sum() - mf1, s2.mcv_freqs.sum() - mf2
            oth1 = 1.0 - s1.null_frac - s1.mcv_freqs.sum()
            oth2 = 1.0 - s2.null_frac - s2.mcv_freqs.sum()
            nmatch = int(match.sum())
            t1sel = matchprod
            if nd2 > len(s2.mcv_vals):
                t1sel += unm1 * oth2 / (nd2 - len(s2.mcv_vals))
            if nd2 > nmatch:
                t1sel += oth1 * (oth2 + unm2) / (nd2 - nmatch)
            t2sel = matchprod
            if nd1 > len(s1.mcv_vals):
                t2sel += unm2 * oth1 / (nd1 - len(s1.mcv_vals))
            if nd1 > nmatch:
                t2sel += oth2 * (oth1 + unm1) / (nd1 - nmatch)
            return float(min(t1sel, t2sel))
        return (1.0 - s1.null_frac) * (1.0 - s2.null_frac) / max(nd1, nd2, 1.0)

    def estimate(self, q: Query) -> float:
        rows = 1.0
        for a in q.rels:
            rows *= self.rel_rows(q, a)
        for j in q.joins:
            rows *= self.join_sel(q.rels[j.a], j.acol, q.rels[j.b], j.bcol)
        return float(max(1.0, round(rows)))

    def explain(self, q: Query) -> Dict[str, float]:
        return {a: self.rel_rows(q, a) for a in q.rels}
