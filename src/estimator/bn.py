"""Per-table Chow-Liu Bayesian network (forest) over discretised columns.

Structure: Chow & Liu (1968) maximum-likelihood tree, with two corrections:
  * plug-in MI is biased upward by ~(kx-1)(ky-1)/2N even for independent columns, so edge
    weights use the Miller-Madow corrected MI;
  * an edge is kept only if a G-test rejects independence (G = 2N*MI ~ chi2 with
    (kx-1)(ky-1) dof, p < 0.001). Unrelated columns stay disconnected (a forest), so a
    surrogate key is not tied to salary by sampling noise.
Parameters: MLE counts with a tiny Laplace prior (alpha=1e-3).
NULLs are their own state; a filter never matches it, a join key never joins on it.
Inference: exact sum-product per tree; independent components multiply.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.engine.executor import Table, pred_rows
from src.engine.sql import Pred

N_BINS = 16
KEY_BINS = 128
G_TEST_P = 1e-3
MIN_ROWS_FOR_TEST = 1000  # tiny tables (6 departments) can't reach significance; keep plain Chow-Liu


class Discretizer:
    """Value -> bin. Two modes:
    * 'edges': continuous numeric, quantile edges (unseen values fall in the enclosing bin);
    * 'lut':  explicit value table (categoricals, small domains, join keys). Numeric unseen
      values map to the bin of the nearest smaller known value; unseen categories to -1.
    A per-table view may append one NULL bin (index k-1)."""

    def __init__(self, mode: str, numeric: bool, edges=None, uniq=None, lut=None, k_value: int = 0,
                 null_bin: bool = False):
        self.mode, self.numeric = mode, numeric
        self.edges, self.uniq, self.lut = edges, uniq, lut
        self.k_value = k_value
        self.null_bin = null_bin
        self.k = k_value + int(null_bin)

    @staticmethod
    def fit(values: np.ndarray, numeric: bool, n_bins: int = N_BINS) -> "Discretizer":
        v = values[np.asarray(pd.notna(values))]
        if not numeric:
            uniq = np.unique(v.astype(str))
            return Discretizer("lut", False, uniq=uniq, lut=np.arange(len(uniq)), k_value=len(uniq))
        uniq = np.unique(v)
        if len(uniq) <= 2 * n_bins:
            return Discretizer("lut", True, uniq=uniq, lut=np.arange(len(uniq)), k_value=len(uniq))
        edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
        return Discretizer("edges", True, edges=edges, k_value=len(edges) - 1)

    @staticmethod
    def fit_key(values: np.ndarray, numeric: bool, n_bins: int = KEY_BINS) -> "Discretizer":
        """Join-key bins: heavy hitters (>= 1/n_bins of all rows) get a bin of their own, the
        rest are cut into equal-row-mass ranges, so no bin hides a huge key next to tiny ones."""
        v = values[np.asarray(pd.notna(values))]
        if not numeric:
            v = v.astype(str)
        uniq, cnt = np.unique(v, return_counts=True)
        if len(uniq) <= n_bins:
            return Discretizer("lut", numeric, uniq=uniq, lut=np.arange(len(uniq)), k_value=len(uniq))
        heavy = np.flatnonzero(cnt >= cnt.sum() / n_bins)[: n_bins // 4]
        lut = np.full(len(uniq), -1)
        lut[heavy] = np.arange(len(heavy))
        rest = np.flatnonzero(lut < 0)
        n_rest = n_bins - len(heavy)
        cum = (np.cumsum(cnt[rest]) - cnt[rest]) / cnt[rest].sum()  # mass before each value
        lut[rest] = len(heavy) + np.minimum((cum * n_rest).astype(int), n_rest - 1)
        return Discretizer("lut", numeric, uniq=uniq, lut=lut, k_value=n_bins)

    def view(self, has_null: bool) -> "Discretizer":
        return Discretizer(self.mode, self.numeric, self.edges, self.uniq, self.lut, self.k_value, has_null)

    def bin(self, values: np.ndarray) -> np.ndarray:
        null = np.asarray(pd.isna(values))
        out = np.full(len(values), (self.k - 1) if self.null_bin else -1, dtype=np.int64)
        v = values[~null]
        if len(v) == 0:
            return out
        if self.mode == "edges":
            b = np.clip(np.searchsorted(self.edges, v.astype(float), "right") - 1, 0, self.k_value - 1)
        elif self.numeric:
            idx = np.clip(np.searchsorted(self.uniq, v.astype(float), "right") - 1, 0, len(self.uniq) - 1)
            b = self.lut[idx]
        else:
            s = v.astype(str)
            idx = np.clip(np.searchsorted(self.uniq, s), 0, len(self.uniq) - 1)
            b = np.where(self.uniq[idx] == s, self.lut[idx], -1)
        out[~null] = b
        return out

    def ndv(self) -> np.ndarray:
        """Distinct values per value-bin (for join-key uniformity)."""
        if self.mode == "lut":
            return np.maximum(np.bincount(self.lut, minlength=self.k_value), 1)
        return np.ones(self.k_value)

    def labels(self) -> List[str]:
        if self.mode == "edges":
            e = self.edges
            out = [f"{e[i]:.0f}-{e[i + 1]:.0f}" for i in range(self.k_value)]
        elif self.k_value == len(self.uniq):
            out = [str(int(c)) if self.numeric and float(c).is_integer() else str(c) for c in self.uniq]
        else:
            out = [f"key bin {i}" for i in range(self.k_value)]
        return out + (["NULL"] if self.null_bin else [])


def _joint(x: np.ndarray, y: np.ndarray, kx: int, ky: int) -> np.ndarray:
    m = (x >= 0) & (y >= 0)
    return np.bincount(x[m] * ky + y[m], minlength=kx * ky).reshape(kx, ky).astype(float)


def mutual_info(joint: np.ndarray) -> float:
    n = joint.sum()
    if n == 0:
        return 0.0
    p = joint / n
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())


class ChowLiuBN:
    def __init__(self, table: Table, cols: List[str], key_disc: Dict[str, Discretizer],
                 alpha: float = 1e-3, p_value: float = G_TEST_P):
        self.table = table.name
        self.n = table.n
        self.cols = [c for c in cols if c in table.cols]
        self.disc: Dict[str, Discretizer] = {}
        for c in self.cols:
            base = key_disc.get(c) or Discretizer.fit(table.cols[c], table.numeric[c])
            self.disc[c] = base.view(bool(np.asarray(pd.isna(table.cols[c])).any()))
        self.rowbins = {c: self.disc[c].bin(table.cols[c]) for c in self.cols}
        self.bincount = {c: np.bincount(self.rowbins[c][self.rowbins[c] >= 0], minlength=self.disc[c].k)
                         for c in self.cols}
        k = {c: self.disc[c].k for c in self.cols}

        # candidate edges: Miller-Madow MI, kept only if the G-test rejects independence
        cand: List[Tuple[float, str, str]] = []
        self.pair_mi: Dict[Tuple[str, str], Tuple[float, float, bool]] = {}
        for i, a in enumerate(self.cols):
            for b in self.cols[i + 1:]:
                j = _joint(self.rowbins[a], self.rowbins[b], k[a], k[b])
                n = j.sum()
                mi = mutual_info(j)
                ra, rb = int((j.sum(1) > 0).sum()), int((j.sum(0) > 0).sum())
                dof = max((ra - 1) * (rb - 1), 1)
                mi_mm = mi - dof / (2 * max(n, 1))
                if table.n < MIN_ROWS_FOR_TEST:
                    significant, mi_mm = mi > 0, mi
                else:
                    significant = bool(n > 0 and 2 * n * mi > chi2.ppf(1 - p_value, dof))
                self.pair_mi[(a, b)] = (mi, mi_mm, significant)
                if significant and mi_mm > 0:
                    cand.append((mi_mm, a, b))
        parent = {c: c for c in self.cols}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        self.edges: List[Tuple[str, str, float]] = []  # (a, b, corrected MI); Kruskal max forest
        for w, a, b in sorted(cand, reverse=True):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
                self.edges.append((a, b, float(w)))
        self.nbrs: Dict[str, List[str]] = {c: [] for c in self.cols}
        self.pair: Dict[Tuple[str, str], np.ndarray] = {}
        for a, b, _ in self.edges:
            self.nbrs[a].append(b)
            self.nbrs[b].append(a)
            j = _joint(self.rowbins[a], self.rowbins[b], k[a], k[b]) + alpha
            j /= j.sum()
            self.pair[(a, b)], self.pair[(b, a)] = j, j.T
        self.marg = {}
        for c in self.cols:
            m = self.bincount[c] + alpha
            self.marg[c] = m / m.sum()
        self.comp: Dict[str, int] = {}
        self.roots: List[str] = []
        for c in self.cols:
            if c in self.comp:
                continue
            stack = [c]
            self.comp[c] = len(self.roots)
            while stack:
                x = stack.pop()
                for y in self.nbrs[x]:
                    if y not in self.comp:
                        self.comp[y] = len(self.roots)
                        stack.append(y)
            self.roots.append(c)
        self._cpt: Dict[Tuple[str, str], np.ndarray] = {}

    def cpt(self, child: str, parent: str) -> np.ndarray:
        """P(child | parent), shape (k_parent, k_child): the MLE conditional table."""
        key = (child, parent)
        if key not in self._cpt:
            j = self.pair[(parent, child)]
            self._cpt[key] = j / j.sum(1, keepdims=True)
        return self._cpt[key]

    def likelihood(self, preds: List[Pred], table: Table) -> Tuple[Dict[str, np.ndarray], float]:
        """Evidence vectors L_c(bin) = P(pred holds | column in bin), from the rows inside each
        bin (NULL bin -> 0). Predicates on columns outside the BN become an independent factor."""
        L: Dict[str, np.ndarray] = {}
        scale = 1.0
        for p in preds:
            rows = pred_rows(table, p)  # sorted-index lookup, cost ~ number of matching rows
            if p.col not in self.disc:
                scale *= len(rows) / max(table.n, 1)
                continue
            b = self.rowbins[p.col][rows]
            hit = np.bincount(b[b >= 0], minlength=self.disc[p.col].k)
            vec = hit / np.maximum(self.bincount[p.col], 1)
            L[p.col] = L[p.col] * vec if p.col in L else vec
        return L, scale

    def _lam(self, node: str, parent: Optional[str], L: Dict[str, np.ndarray]) -> np.ndarray:
        lam = L.get(node, np.ones(self.disc[node].k)).copy()
        for c in self.nbrs[node]:
            if c != parent:
                lam *= self.cpt(c, node) @ self._lam(c, node, L)
        return lam

    def _comp_prob(self, ci: int, L: Dict[str, np.ndarray]) -> float:
        r = self.roots[ci]
        return float((self.marg[r] * self._lam(r, None, L)).sum())

    def belief(self, var: str, L: Dict[str, np.ndarray]) -> np.ndarray:
        """P(var = bin, evidence): its tree rooted at `var`, times the other components."""
        other = 1.0
        for ci in {self.comp[c] for c in L} - {self.comp[var]}:
            other *= self._comp_prob(ci, L)
        return self.marg[var] * self._lam(var, None, L) * other

    def prob(self, L: Dict[str, np.ndarray]) -> float:
        out = 1.0
        for ci in {self.comp[c] for c in L}:
            out *= self._comp_prob(ci, L)
        return out
