"""Per-table Chow-Liu Bayesian network over discretised columns.

Structure: maximum spanning tree on pairwise mutual information (Chow & Liu 1968), the
tree that maximises data likelihood among all tree-shaped BNs.
Parameters: MLE counts with a tiny Laplace prior (alpha=1e-3): enough to avoid log(0), small
enough that combinations absent from the data (Engineering in the West) stay ~impossible. Inference: exact sum-product on the tree with per-column likelihood
vectors as (soft) evidence.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.engine.executor import Table, pred_mask
from src.engine.sql import Pred

N_BINS = 16


class Discretizer:
    """Maps a column to bins. Small domains / categoricals: one bin per value."""

    def __init__(self, values: np.ndarray, numeric: bool, n_bins: int = N_BINS):
        self.numeric = numeric
        self.cats: Optional[np.ndarray] = None
        self.edges: Optional[np.ndarray] = None
        if not numeric or len(np.unique(values)) <= 2 * n_bins:
            self.cats = np.unique(values)
        else:
            self.edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
        self.k = len(self.cats) if self.cats is not None else len(self.edges) - 1

    def bin(self, values: np.ndarray) -> np.ndarray:
        if self.cats is not None:
            idx = np.searchsorted(self.cats, values)
            idx = np.clip(idx, 0, self.k - 1)
            return np.where(self.cats[idx] == values, idx, -1)
        return np.clip(np.searchsorted(self.edges, values, "right") - 1, 0, self.k - 1)

    def labels(self) -> List[str]:
        if self.cats is not None:
            return [str(c if not isinstance(c, float) or not c.is_integer() else int(c)) for c in self.cats]
        e = self.edges
        return [f"{e[i]:.0f}-{e[i + 1]:.0f}" for i in range(self.k)]


def mutual_info(x: np.ndarray, y: np.ndarray, kx: int, ky: int) -> float:
    joint = np.zeros((kx, ky))
    np.add.at(joint, (x, y), 1)
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


class ChowLiuBN:
    def __init__(self, table: Table, key_disc: Dict[str, "Discretizer"], alpha: float = 1e-3,
                 exclude=("name",)):
        self.table = table.name
        self.n = table.n
        self.cols = [c for c in table.cols if c not in exclude]
        self.disc = {c: key_disc.get(c) or Discretizer(table.cols[c], table.numeric[c]) for c in self.cols}
        self.rowbins = {c: self.disc[c].bin(table.cols[c]) for c in self.cols}
        self.bincount = {c: np.bincount(self.rowbins[c][self.rowbins[c] >= 0], minlength=self.disc[c].k)
                         for c in self.cols}
        k = {c: self.disc[c].k for c in self.cols}
        n = len(self.cols)
        mi = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                mi[i, j] = mi[j, i] = mutual_info(self.rowbins[self.cols[i]], self.rowbins[self.cols[j]],
                                                  k[self.cols[i]], k[self.cols[j]])
        self.mi = mi
        # Prim's maximum spanning tree
        self.edges: List[Tuple[str, str, float]] = []  # (parent, child, MI)
        if n > 1:
            inside, best, par = {0}, mi[0].copy(), np.zeros(n, dtype=int)
            while len(inside) < n:
                cand = [v for v in range(n) if v not in inside]
                v = max(cand, key=lambda x: best[x])
                inside.add(v)
                self.edges.append((self.cols[par[v]], self.cols[v], float(mi[par[v], v])))
                for u in range(n):
                    if u not in inside and mi[v, u] > best[u]:
                        best[u], par[u] = mi[v, u], v
        self.nbrs: Dict[str, List[str]] = {c: [] for c in self.cols}
        self.pair: Dict[Tuple[str, str], np.ndarray] = {}
        for p, c, _ in self.edges:
            self.nbrs[p].append(c)
            self.nbrs[c].append(p)
            j = np.full((k[p], k[c]), alpha)
            np.add.at(j, (self.rowbins[p], self.rowbins[c]), 1)
            j /= j.sum()
            self.pair[(p, c)], self.pair[(c, p)] = j, j.T
        self.marg = {}
        for c in self.cols:
            m = self.bincount[c] + alpha
            self.marg[c] = m / m.sum()

    def cpt(self, child: str, parent: str) -> np.ndarray:
        """P(child | parent), shape (k_parent, k_child): the MLE conditional table."""
        j = self.pair[(parent, child)]
        return j / j.sum(1, keepdims=True)

    def likelihood(self, preds: List[Pred], table: Table) -> Tuple[Dict[str, np.ndarray], float]:
        """Evidence vectors L_c(bin) = P(pred holds | column in bin), from the data inside each
        bin. Predicates on columns outside the BN become an independent scale factor."""
        L: Dict[str, np.ndarray] = {}
        scale = 1.0
        for p in preds:
            m = pred_mask(table, p)
            if p.col not in self.disc:
                scale *= m.mean()
                continue
            b = self.rowbins[p.col]
            ok = b >= 0
            hit = np.bincount(b[m & ok], minlength=self.disc[p.col].k)
            vec = hit / np.maximum(self.bincount[p.col], 1)
            L[p.col] = L[p.col] * vec if p.col in L else vec
        return L, scale

    def _lam(self, node: str, parent: Optional[str], L: Dict[str, np.ndarray]) -> np.ndarray:
        lam = L.get(node, np.ones(self.disc[node].k)).copy()
        for c in self.nbrs[node]:
            if c != parent:
                lam *= self.cpt(c, node) @ self._lam(c, node, L)
        return lam

    def belief(self, var: str, L: Dict[str, np.ndarray]) -> np.ndarray:
        """P(var = bin, evidence): tree rooted at `var`, one upward sum-product pass."""
        return self.marg[var] * self._lam(var, None, L)

    def prob(self, L: Dict[str, np.ndarray]) -> float:
        return float(self.belief(self.cols[0], L).sum()) if L else 1.0
