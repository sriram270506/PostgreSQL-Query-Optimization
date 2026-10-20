"""MSCN baseline (Kipf et al., "Learned Cardinalities", CIDR 2019).

A query is three sets: tables (one-hot + a bitmap of which of 1,000 sampled rows pass
that table's filters), joins (one-hot edge) and predicates (one-hot column, one-hot
operator, normalised literal). Each set goes through its own 2-layer MLP, is average-pooled,
and the three pooled vectors feed an output MLP predicting log(1 + rows). No database
statistics, no query structure beyond the sets: the standard learned baseline.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.engine.executor import Catalog, pred_mask
from src.engine.sql import OPS, Query

N_SAMPLES = 1000


class MSCNFeaturizer:
    def __init__(self, catalog: Catalog, seed: int = 0):
        self.cat = catalog
        ds = catalog.ds
        self.tables = list(catalog.tables)
        self.edges = sorted({tuple(sorted([(a, b), (c, d)])) for a, b, c, d in ds.join_edges})
        self.edge_idx = {e: i for i, e in enumerate(self.edges)}
        self.columns = sorted({(t, c) for t in self.tables for c in catalog.tables[t].cols})
        self.col_idx = {c: i for i, c in enumerate(self.columns)}
        rng = np.random.default_rng(seed)
        self.samples = {t: rng.choice(tb.n, size=min(N_SAMPLES, tb.n), replace=False)
                        for t, tb in catalog.tables.items()}
        self.range = {(t, c): (float(np.nanmin(tb.cols[c])), float(np.nanmax(tb.cols[c])))
                      for t, tb in catalog.tables.items() for c in tb.cols if tb.numeric[c]}
        self.t_dim = len(self.tables) + N_SAMPLES
        self.j_dim = len(self.edges) + 1  # +1: join outside the schema's edge list
        self.p_dim = len(self.columns) + len(OPS) + 3

    def featurize(self, q: Query) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = q.canonical()
        T = np.zeros((len(q.rels), self.t_dim), dtype=np.float32)
        for i, (a, t) in enumerate(q.rels.items()):
            T[i, self.tables.index(t)] = 1
            tb = self.cat.tables[t]
            idx = self.samples[t]
            m = np.ones(len(idx), dtype=bool)
            for p in q.preds_of(a):
                m &= pred_mask(tb, p, idx)  # only the sampled rows, not the whole table
            T[i, len(self.tables):len(self.tables) + len(idx)] = m
        J = np.zeros((max(len(q.joins), 1), self.j_dim), dtype=np.float32)
        for i, j in enumerate(q.joins):
            e = tuple(sorted([(q.rels[j.a], j.acol), (q.rels[j.b], j.bcol)]))
            J[i, self.edge_idx.get(e, self.j_dim - 1)] = 1
        P = np.zeros((max(len(q.preds), 1), self.p_dim), dtype=np.float32)
        for i, p in enumerate(q.preds):
            t = q.rels[p.alias]
            P[i, self.col_idx[(t, p.col)]] = 1
            P[i, len(self.columns) + OPS.index(p.op)] = 1
            vals = p.value if p.op in ("between", "in") else (p.value,)
            if (t, p.col) in self.range:
                mn, mx = self.range[(t, p.col)]
                try:
                    norm = [(float(v) - mn) / (mx - mn or 1) for v in vals]
                except (TypeError, ValueError):
                    norm = [0.5]
                P[i, -3:] = [min(norm), max(norm), len(vals) / 10]
            else:
                P[i, -1] = len(vals) / 10
        return T, J, P


class SetConv(nn.Module):
    def __init__(self, t_dim: int, j_dim: int, p_dim: int, hid: int):
        super().__init__()
        mlp = lambda d: nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU())  # noqa: E731
        self.t, self.j, self.p = mlp(t_dim), mlp(j_dim), mlp(p_dim)
        self.out = nn.Sequential(nn.Linear(3 * hid, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.use_tree = False

    @staticmethod
    def _pool(x, mask):
        return (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

    def forward(self, b):
        h = torch.cat([self._pool(self.t(b["T"]), b["Tm"]), self._pool(self.j(b["J"]), b["Jm"]),
                       self._pool(self.p(b["P"]), b["Pm"])], 1)
        return self.out(h).squeeze(1)


class MSCNData:
    def __init__(self, feats: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], y=None):
        self.feats, self.y = feats, y

    def batch(self, idx) -> Dict[str, torch.Tensor]:
        out = {}
        for k, part in (("T", 0), ("J", 1), ("P", 2)):
            arrs = [self.feats[i][part] for i in idx]
            n = max(a.shape[0] for a in arrs)
            X = np.zeros((len(arrs), n, arrs[0].shape[1]), dtype=np.float32)
            M = np.zeros((len(arrs), n), dtype=np.float32)
            for r, a in enumerate(arrs):
                X[r, :a.shape[0]] = a
                M[r, :a.shape[0]] = 1 if a.any() else 0
            out[k], out[k + "m"] = torch.from_numpy(X), torch.from_numpy(M)
        if self.y is not None:
            out["y"] = torch.from_numpy(np.log1p(self.y[np.asarray(idx)])).float()
        return out


def predict(model: SetConv, data: MSCNData, idx=None, bs: int = 512) -> np.ndarray:
    idx = np.arange(len(data.feats)) if idx is None else np.asarray(idx)
    model.eval()
    out = []
    with torch.inference_mode():
        for s in range(0, len(idx), bs):
            out.append(model(data.batch(idx[s:s + bs])).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train(cfg: Dict, data: MSCNData, tr_idx, va_idx, dims, max_epochs=100, patience=12, seed=0, report=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SetConv(*dims, cfg["hidden"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    from .neural import make_loss, val_score
    loss_fn = make_loss(cfg)
    best, best_state, bad, hist = math.inf, None, 0, []
    for ep in range(max_epochs):
        model.train()
        perm = rng.permutation(np.asarray(tr_idx))
        tot = 0.0
        for s in range(0, len(perm), cfg["batch"]):
            b = data.batch(perm[s:s + cfg["batch"]])
            loss = loss_fn(model(b), b["y"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b["y"])
        pv = predict(model, data, va_idx)
        val = val_score(np.abs(np.log(np.maximum(np.expm1(pv), 1)) - np.log(np.maximum(data.y[va_idx], 1))), cfg)
        hist.append({"epoch": ep, "train_loss": tot / len(perm), "val_mean_log_q": val})
        if report:
            report(ep, val)
        if val < best - 1e-4:
            best, bad, best_state = val, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_val": best, "history": hist}


class StackedSetConv(nn.Module):
    """M MSCN members in one pass (leading member dimension on every weight)."""

    def __init__(self, models: List[SetConv]):
        super().__init__()
        st = lambda get: torch.stack([get(m).detach() for m in models])  # noqa: E731
        self.sets = {k: [(st(lambda m, k=k, i=i: getattr(m, k)[i].weight), st(lambda m, k=k, i=i: getattr(m, k)[i].bias))
                         for i in (0, 2)] for k in ("t", "j", "p")}
        self.out = [(st(lambda m, i=i: m.out[i].weight), st(lambda m, i=i: m.out[i].bias)) for i in (0, 2)]

    @staticmethod
    def _lin(x, W, b):  # x (B,S,i) or (M,B,S,i); W (M,o,i)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return torch.matmul(x, W.transpose(1, 2).unsqueeze(1)) + b[:, None, None, :]

    @torch.inference_mode()
    def forward(self, b) -> torch.Tensor:
        pooled = []
        for k, key in (("t", "T"), ("j", "J"), ("p", "P")):
            (W1, b1), (W2, b2) = self.sets[k]
            h = torch.relu(self._lin(torch.relu(self._lin(b[key], W1, b1)), W2, b2))  # (M,B,S,H)
            m = b[key + "m"][None, :, :, None]
            pooled.append((h * m).sum(2) / m.sum(2).clamp(min=1))
        z = torch.cat(pooled, 2)  # (M,B,3H)
        (W1, b1), (W2, b2) = self.out
        z = torch.relu(torch.matmul(z, W1.transpose(1, 2)) + b1[:, None, :])
        return (torch.matmul(z, W2.transpose(1, 2)) + b2[:, None, :]).squeeze(2)
