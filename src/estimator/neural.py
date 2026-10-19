"""TreeLSTM query encoder + fusion MLP.

Query -> plan-shaped tree:   AGG
                              └ JOIN(e.dept_id = p.dept_id)
                                  ├ SCAN employees ── PRED city = 'Pune'
                                  └ SCAN projects  ── PRED budget_tier IN (1, 2)
Each node gets a feature vector: node type, table, column, operator, normalised literal
range, categorical literal one-hots, plus two statistics slots (predicate selectivity;
table BN/PG selectivity; FactorJoin/PG estimate of the join subtree). A Child-Sum TreeLSTM (Tai et
al. 2015) folds the tree bottom-up; the root state h_root summarises the whole query.

The fusion MLP sees [h_root | statistics features] where the statistics are the
per-table Bayesian-network selectivities, the FactorJoin estimate and the PostgreSQL
estimate. It predicts a residual on top of log FactorJoin:
    log(1+card) = log(1+FJ) + MLP([h_root, stats])
With use_tree=False the same head runs on the statistics alone (the "MLP" ablation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.engine.executor import Catalog, pred_count
from src.engine.pg_planner import PGPlanner
from src.engine.sql import OPS, Query
from .factorjoin import FactorJoin

NODE_TYPES = ["AGG", "JOIN", "SCAN", "PRED"]


# ─── featurisation ───────────────────────────────────────────────────────────

@dataclass
class TreeSample:
    x: np.ndarray            # (n_nodes, F) node features
    parent: np.ndarray       # (n_nodes,) parent index, -1 for root (node 0)
    height: np.ndarray       # (n_nodes,) 0 for leaves
    labels: List[str] = field(default_factory=list)  # human-readable node names (viz)
    meta: List[dict] = field(default_factory=list)    # kind, table, col, op, aliases under node


class Featurizer:
    def __init__(self, catalog: Catalog, fj: FactorJoin, pg: PGPlanner):
        self.cat, self.fj, self.pg = catalog, fj, pg
        self.tables = list(catalog.tables)
        self.columns = sorted({f"{t}.{c}" for t, tb in catalog.tables.items() for c in tb.cols})
        self.col_idx = {c: i for i, c in enumerate(self.columns)}
        self.vocab = sorted({str(v) for t, tb in catalog.tables.items() for c in catalog.ds.filters.get(t, [])
                             if not tb.numeric[c] for v in np.unique(tb.cols[c][pd.notna(tb.cols[c])])})
        self.vocab_idx = {v: i for i, v in enumerate(self.vocab)}
        self.range = {(t, c): (float(np.nanmin(tb.cols[c])), float(np.nanmax(tb.cols[c])))
                      for t, tb in catalog.tables.items() for c in tb.cols if tb.numeric[c]}
        nt, nc = len(self.tables), len(self.columns)
        self.slices = {"type": (0, 4), "table": (4, 4 + nt), "column": (4 + nt, 4 + nt + nc),
                       "op": (4 + nt + nc, 4 + nt + nc + len(OPS))}
        o = self.slices["op"][1]
        self.slices["literal"] = (o, o + 4)  # lo, hi, width, n_values/10
        self.slices["value"] = (o + 4, o + 4 + len(self.vocab))
        # per-node statistics: PRED -> log selectivity; SCAN -> BN / PG log selectivity;
        # JOIN, AGG -> log FactorJoin / PG estimate of the subtree (scaled to ~[0, 1])
        self.slices["stat"] = (o + 4 + len(self.vocab), o + 6 + len(self.vocab))
        self.dim = o + 6 + len(self.vocab)
        self.stats_names = ([f"present_{t}" for t in self.tables] + [f"bn_logsel_{t}" for t in self.tables]
                            + ["log_fj", "log_pg", "n_tables", "n_preds"])

    def _node(self, kind: str, table: Optional[str] = None, cols=(), op=None, pred=None) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        v[NODE_TYPES.index(kind)] = 1
        if table:
            v[self.slices["table"][0] + self.tables.index(table)] = 1
        for c in cols:
            v[self.slices["column"][0] + self.col_idx[c]] = 1
        if op:
            v[self.slices["op"][0] + OPS.index(op)] = 1
        if pred is not None:
            lo_i = self.slices["literal"][0]
            vals = pred.value if pred.op in ("between", "in") else (pred.value,)
            key = (table, pred.col)
            if key in self.range:
                mn, mx = self.range[key]
                norm = [(float(x) - mn) / (mx - mn or 1) for x in vals]
                lo, hi = min(norm), max(norm)
                if pred.op in ("lt", "le"):
                    lo = 0.0
                elif pred.op in ("gt", "ge"):
                    lo, hi = hi, 1.0
                v[lo_i:lo_i + 3] = [lo, hi, hi - lo]
            else:
                for x in vals:
                    if str(x) in self.vocab_idx:
                        v[self.slices["value"][0] + self.vocab_idx[str(x)]] = 1
            v[lo_i + 3] = len(vals) / 10
        return v

    def tree(self, q: Query) -> TreeSample:
        q = q.canonical()
        feats, parent, labels, meta = [], [], [], []

        def add(vec, par, label, m):
            feats.append(vec)
            parent.append(par)
            labels.append(label)
            meta.append(m)
            return len(feats) - 1

        root = add(self._node("AGG"), -1, "COUNT(*)", {"kind": "AGG"})

        def scan(alias, par):
            t = q.rels[alias]
            s = add(self._node("SCAN", t), par, f"SCAN {t}", {"kind": "SCAN", "table": t, "alias": alias})
            for p in q.preds_of(alias):
                add(self._node("PRED", t, [f"{t}.{p.col}"], p.op, p), s, p.to_sql(),
                    {"kind": "PRED", "table": t, "alias": alias, "col": p.col, "op": p.op, "pred": p})
            return s

        # left-deep join tree in BFS order from the first relation
        first = next(iter(q.rels))
        order, placed, pending = [], {first}, list(q.joins)
        while pending:
            j = next((j for j in pending if (j.a in placed) != (j.b in placed)), None)
            if j is None:
                break
            pending.remove(j)
            new = j.b if j.a in placed else j.a
            placed.add(new)
            order.append((j, new))
        if not order:
            scan(first, root)
        else:
            # build top-down: outermost join is the last one added
            def build(k, par):
                if k < 0:
                    return scan(first, par)
                j, new = order[k]
                cols = [f"{q.rels[j.a]}.{j.acol}", f"{q.rels[j.b]}.{j.bcol}"]
                n = add(self._node("JOIN", None, cols, "eq"), par, f"JOIN {j.to_sql()}", {"kind": "JOIN"})
                build(k - 1, n)
                scan(new, n)
                return n
            build(len(order) - 1, root)
        parent_a = np.array(parent)
        height = np.zeros(len(parent_a), dtype=int)
        for i in range(len(parent_a) - 1, 0, -1):  # children are always added after parents
            p = parent_a[i]
            height[p] = max(height[p], height[i] + 1)
        under = [{m["alias"]} if m["kind"] == "SCAN" else set() for m in meta]
        for i in range(len(parent_a) - 1, 0, -1):
            under[parent_a[i]] |= under[i]
        for m, u in zip(meta, under):
            m["aliases"] = sorted(u)
        st = self.slices["stat"][0]
        for v, m in zip(feats, meta):
            if m["kind"] == "PRED":
                tb = self.cat.tables[m["table"]]
                v[st] = math.log(max(pred_count(tb, m["pred"]) / max(tb.n, 1), 1e-6)) / 14
            elif m["kind"] == "SCAN":
                v[st] = math.log(max(self.fj.selectivity(q, m["alias"]), 1e-9)) / 14
                v[st + 1] = math.log(self.pg.rel_rows(q, m["alias"]) / max(self.cat.tables[m["table"]].n, 1)) / 14
            else:
                sub = q.sub(m["aliases"])
                v[st] = math.log1p(self.fj.estimate(sub)) / 20
                v[st + 1] = math.log1p(self.pg.estimate(sub)) / 20
        return TreeSample(np.stack(feats), parent_a, height, labels, meta)

    def stats(self, q: Query) -> np.ndarray:
        q = q.canonical()
        v = np.zeros(len(self.stats_names), dtype=np.float32)
        nt = len(self.tables)
        for a, t in q.rels.items():
            i = self.tables.index(t)
            v[i] = 1
            v[nt + i] = math.log(max(self.fj.selectivity(q, a), 1e-9))
        v[2 * nt] = math.log1p(self.fj.estimate(q))
        v[2 * nt + 1] = math.log1p(self.pg.estimate(q))
        v[2 * nt + 2] = len(q.rels)
        v[2 * nt + 3] = len(q.preds)
        return v


# ─── model ───────────────────────────────────────────────────────────────────

class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim: int, embed: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Sequential(nn.Linear(in_dim, embed), nn.ReLU())
        self.W_iou = nn.Linear(embed, 3 * hidden)
        self.U_iou = nn.Linear(hidden, 3 * hidden, bias=False)
        self.W_f = nn.Linear(embed, hidden)
        self.U_f = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, parent, height, trace: bool = False):
        """x: (N, F); parent: (N,) long, -1 for roots; height: (N,). Returns h, c (N, H)."""
        N, H = x.shape[0], self.hidden
        e = self.embed(x)
        h = x.new_zeros(N, H)
        c = x.new_zeros(N, H)
        tr = {"i": x.new_zeros(N, H), "o": x.new_zeros(N, H), "u": x.new_zeros(N, H),
              "f": x.new_zeros(N, H)} if trace else None
        child = torch.arange(N, device=x.device)
        for lvl in range(int(height.max()) + 1):
            idx = (height == lvl).nonzero(as_tuple=True)[0]
            kids = child[(parent >= 0) & (height[parent.clamp(min=0)] == lvl)]
            kp = parent[kids]
            h_sum = x.new_zeros(N, H).index_add(0, kp, h[kids])[idx]
            iou = self.W_iou(e[idx]) + self.U_iou(h_sum)
            i, o, u = iou.chunk(3, dim=1)
            i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
            f = torch.sigmoid(self.W_f(e[kp]) + self.U_f(h[kids]))      # one forget gate per child
            fc = x.new_zeros(N, H).index_add(0, kp, f * c[kids])[idx]
            c_new = i * u + fc
            h_new = o * torch.tanh(c_new)
            c = c.index_copy(0, idx, c_new)
            h = h.index_copy(0, idx, h_new)
            if trace:
                tr["i"][idx], tr["o"][idx], tr["u"][idx] = i.detach(), o.detach(), u.detach()
                tr["f"] = tr["f"].index_copy(0, kids, f.detach())
        return h, c, tr


class FusionModel(nn.Module):
    def __init__(self, node_dim: int, stats_dim: int, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.use_tree = cfg.get("use_tree", True)
        self.tree = ChildSumTreeLSTM(node_dim, cfg["embed"], cfg["hidden"]) if self.use_tree else None
        d = stats_dim + (cfg["hidden"] if self.use_tree else 0)
        act = nn.GELU if cfg.get("activation") == "gelu" else nn.ReLU
        layers: List[nn.Module] = []
        for _ in range(cfg["mlp_layers"]):
            layers += [nn.Linear(d, cfg["mlp_width"]), act(), nn.Dropout(cfg["dropout"])]
            d = cfg["mlp_width"]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(d, 1)
        nn.init.zeros_(self.out.weight)  # start exactly at the FactorJoin estimate
        nn.init.zeros_(self.out.bias)

    def forward(self, batch, trace: bool = False):
        parts = [batch["stats"]]
        tr = {}
        if self.use_tree:
            h, c, gates = self.tree(batch["x"], batch["parent"], batch["height"], trace)
            parts.insert(0, h[batch["roots"]])
            if trace:
                tr.update(h=h.detach(), c=c.detach(), gates=gates)
        z = torch.cat(parts, dim=1)
        acts = []
        for m in self.mlp:
            z = m(z)
            if trace and isinstance(m, (nn.ReLU, nn.GELU)):
                acts.append(z.detach())
        resid = self.out(z).squeeze(1)
        pred = batch["log_fj"] + resid
        if trace:
            tr.update(mlp_acts=acts, resid=resid.detach())
            return pred, tr
        return pred


# ─── batching / data ─────────────────────────────────────────────────────────

class Dataset:
    """Pre-featurised workload; `stats` are standardised with train-split moments."""

    def __init__(self, trees: List[TreeSample], stats: np.ndarray, y: Optional[np.ndarray],
                 mean: np.ndarray, std: np.ndarray, log_fj_col: int):
        self.trees = trees
        self.raw_stats = stats
        self.stats = (stats - mean) / std
        self.log_fj = stats[:, log_fj_col]
        self.y = y

    def batch(self, idx, need_tree: bool = True) -> Dict[str, torch.Tensor]:
        xs, par, hts, roots, off = [], [], [], [], 0
        for i in (idx if need_tree else idx[:1]):
            t = self.trees[i]
            xs.append(t.x)
            par.append(np.where(t.parent >= 0, t.parent + off, -1))
            hts.append(t.height)
            roots.append(off)
            off += len(t.parent)
        b = {"x": torch.from_numpy(np.concatenate(xs)),
             "parent": torch.from_numpy(np.concatenate(par)).long(),
             "height": torch.from_numpy(np.concatenate(hts)).long(),
             "roots": torch.tensor(roots, dtype=torch.long) if need_tree else torch.zeros(len(idx), dtype=torch.long),
             "stats": torch.from_numpy(self.stats[idx]).float(),
             "log_fj": torch.from_numpy(self.log_fj[idx]).float()}
        if self.y is not None:
            b["y"] = torch.from_numpy(np.log1p(self.y[idx])).float()
        return b


def log_qerror(pred_log1p: np.ndarray, true: np.ndarray) -> np.ndarray:
    est = np.maximum(np.expm1(pred_log1p), 1.0)
    return np.abs(np.log(est) - np.log(np.maximum(true, 1.0)))


def predict(model: FusionModel, ds: Dataset, idx=None, bs: int = 512) -> np.ndarray:
    idx = np.arange(len(ds.trees)) if idx is None else np.asarray(idx)
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            out.append(model(ds.batch(idx[s:s + bs], model.use_tree)).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train(cfg: Dict, ds: Dataset, tr_idx, va_idx, node_dim: int, max_epochs: int = 60,
          patience: int = 8, seed: int = 0, report=None) -> Tuple[FusionModel, Dict]:
    """Adam + early stopping on validation mean log Q-error. `report(epoch, val)` may raise
    to prune (Optuna)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = FusionModel(node_dim, ds.stats.shape[1], cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.HuberLoss(delta=1.0) if cfg.get("loss") == "huber" else nn.MSELoss()
    best, best_state, bad, hist = math.inf, None, 0, []
    tr_idx = np.asarray(tr_idx)
    for ep in range(max_epochs):
        model.train()
        perm = rng.permutation(tr_idx)
        tot = 0.0
        for s in range(0, len(perm), cfg["batch"]):
            b = ds.batch(perm[s:s + cfg["batch"]], model.use_tree)
            loss = loss_fn(model(b), b["y"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * len(b["roots"])
        val = float(log_qerror(predict(model, ds, va_idx), ds.y[va_idx]).mean())
        hist.append({"epoch": ep, "train_loss": tot / len(perm), "val_mean_log_q": val})
        if report:
            report(ep, val)
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_val": best, "history": hist}


# ─── stacked ensemble (serving) ──────────────────────────────────────────────

class StackedFusion(nn.Module):
    """M same-config FusionModels evaluated in one pass: every weight gets a leading
    member dimension, so the tree is walked once for all members instead of M times.
    Numerically equivalent to running the members one by one (tested)."""

    def __init__(self, models: List[FusionModel]):
        super().__init__()
        m0 = models[0]
        self.use_tree = m0.use_tree
        self.act = torch.nn.functional.gelu if m0.cfg.get("activation") == "gelu" else torch.relu
        st = lambda get: torch.stack([get(m).detach() for m in models])  # noqa: E731
        if self.use_tree:
            self.We, self.be = st(lambda m: m.tree.embed[0].weight), st(lambda m: m.tree.embed[0].bias)
            self.Wiou, self.biou = st(lambda m: m.tree.W_iou.weight), st(lambda m: m.tree.W_iou.bias)
            self.Uiou = st(lambda m: m.tree.U_iou.weight)
            self.Wf, self.bf = st(lambda m: m.tree.W_f.weight), st(lambda m: m.tree.W_f.bias)
            self.Uf = st(lambda m: m.tree.U_f.weight)
            self.H = m0.tree.hidden
        lin = [i for i, l in enumerate(m0.mlp) if isinstance(l, nn.Linear)]
        self.mlp = [(st(lambda m, i=i: m.mlp[i].weight), st(lambda m, i=i: m.mlp[i].bias)) for i in lin]
        self.Wo, self.bo = st(lambda m: m.out.weight), st(lambda m: m.out.bias)

    @staticmethod
    def _lin(x, W, b=None):
        y = torch.matmul(x, W.transpose(1, 2))  # (M,N,i) or broadcast (N,i) @ (M,i,o) -> (M,N,o)
        return y if b is None else y + b[:, None, :]

    @torch.inference_mode()
    def forward(self, batch) -> torch.Tensor:
        """(members, queries) predicted log(1 + rows)."""
        M = self.Wo.shape[0]
        parts = [batch["stats"].expand(M, -1, -1)]
        if self.use_tree:
            x, parent, height = batch["x"], batch["parent"], batch["height"]
            N, H = x.shape[0], self.H
            e = torch.relu(self._lin(x, self.We, self.be))
            h = x.new_zeros(M, N, H)
            c = x.new_zeros(M, N, H)
            child = torch.arange(N)
            for lvl in range(int(height.max()) + 1):
                idx = (height == lvl).nonzero(as_tuple=True)[0]
                kids = child[(parent >= 0) & (height[parent.clamp(min=0)] == lvl)]
                kp = parent[kids]
                h_sum = x.new_zeros(M, N, H).index_add(1, kp, h[:, kids])[:, idx]
                iou = self._lin(e[:, idx], self.Wiou, self.biou) + self._lin(h_sum, self.Uiou)
                i, o, u = iou.chunk(3, dim=2)
                i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
                f = torch.sigmoid(self._lin(e[:, kp], self.Wf, self.bf) + self._lin(h[:, kids], self.Uf))
                fc = x.new_zeros(M, N, H).index_add(1, kp, f * c[:, kids])[:, idx]
                c_new = i * u + fc
                c = c.index_copy(1, idx, c_new)
                h = h.index_copy(1, idx, o * torch.tanh(c_new))
            parts.insert(0, h[:, batch["roots"]])
        z = torch.cat(parts, dim=2)
        for W, b in self.mlp:
            z = self.act(self._lin(z, W, b))
        return batch["log_fj"][None, :] + self._lin(z, self.Wo, self.bo).squeeze(2)
