"""What the networks learned: the query-model analogue of CNN filter visualisation.

CNN story: early filters fire on edges, later ones on shapes, the last ones on objects.
Here the "layers" of the TreeLSTM are the levels of the query tree:
  PRED (one condition) -> SCAN (one table after its filters) -> JOIN (tables combined) -> AGG (whole query)
and the MLP's layers sit on top. We measure, per level/layer:
  * neuron atlas: for every hidden unit, which node kinds / columns / operators make it fire
    (the "what does this filter detect" picture);
  * linear probes: how well a straight line on the hidden state recovers the TRUE
    cardinality of that subtree (R^2) - information becoming more task-specific with depth;
  * 2-D PCA maps of hidden states (clusters of similar sub-queries);
  * MLP: first-layer receptive fields, layer-wise probes, integrated-gradient attributions.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from src.engine.executor import filter_rows, pred_mask
from src.engine.sql import Query
from .neural import FusionModel
from .service import Estimators

KINDS = ["PRED", "SCAN", "JOIN", "AGG"]


def _ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0):
    """Closed-form ridge with intercept. Returns (w, b, held-out R^2 on a 30% split)."""
    n = len(y)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    tr, te = perm[: int(0.7 * n)], perm[int(0.7 * n):]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    live = sd > 1e-3 * max(sd.max(), 1e-12)  # (near-)dead units carry no signal; dividing by ~0 explodes the fit
    sd = np.where(live, sd, 1.0)
    Z = np.where(live, (X - mu) / sd, 0.0)
    A = Z[tr].T @ Z[tr] + lam * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z[tr].T @ (y[tr] - y[tr].mean()))
    b = y[tr].mean()
    pred = np.clip(Z[te] @ w + b, y[tr].min(), y[tr].max())  # a probe never extrapolates past seen sizes
    r2 = 1 - ((y[te] - pred) ** 2).sum() / max(((y[te] - y[te].mean()) ** 2).sum(), 1e-9)
    # refit on everything for later per-node readouts
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - y.mean()))
    return {"w": (w / sd).tolist(), "b": float(y.mean() - (mu / sd) @ w), "r2": float(r2)}


def _pca2(X: np.ndarray):
    Xc = X - X.mean(0)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    var = (s ** 2) / max((s ** 2).sum(), 1e-12)
    return Xc @ vt[:2].T, var[:2].tolist()


def _input_groups(S: Estimators, model: FusionModel) -> List[Dict]:
    names = S.fz.stats_names
    off = model.cfg["hidden"] if model.use_tree else 0
    groups = []
    if model.use_tree:
        groups.append({"name": "TreeLSTM query embedding", "idx": list(range(off))})
    nt = len(S.fz.tables)
    groups += [
        {"name": "Tables present", "idx": [off + i for i in range(nt)]},
        {"name": "BN selectivities", "idx": [off + nt + i for i in range(nt)]},
        {"name": "FactorJoin estimate", "idx": [off + names.index("log_fj")]},
        {"name": "PostgreSQL estimate", "idx": [off + names.index("log_pg")]},
        {"name": "Query shape (#tables, #preds)", "idx": [off + names.index("n_tables"), off + names.index("n_preds")]},
    ]
    return groups


def _mlp_input(model: FusionModel, batch, trace=False):
    parts = [batch["stats"]]
    tr = None
    if model.use_tree:
        h, c, tr = model.tree(batch["x"], batch["parent"], batch["height"], trace)
        parts.insert(0, h[batch["roots"]])
        tr = {"h": h, "c": c, "gates": tr}
    return torch.cat(parts, 1), tr


def _head(model: FusionModel, z):
    return model.out(model.mlp(z)).squeeze(1)


def integrated_gradients(model: FusionModel, z: torch.Tensor, steps: int = 32) -> np.ndarray:
    base = torch.zeros_like(z)  # standardised mean query
    alphas = torch.linspace(0, 1, steps).view(-1, 1, 1)
    path = (base + alphas * (z - base)).reshape(-1, z.shape[1]).requires_grad_(True)
    out = _head(model, path).sum()
    (g,) = torch.autograd.grad(out, path)
    g = g.reshape(steps, z.shape[0], z.shape[1]).mean(0)
    return ((z - base) * g).detach().numpy()


def _true_node_targets(S: Estimators, q: Query, meta: List[dict], cache: Dict[str, int]) -> List[float]:
    """log10 of the true row count each node represents (PRED: rows passing that condition)."""
    out = []
    cat = S.engine.catalog
    for m in meta:
        if m["kind"] == "PRED":
            tb = cat.tables[m["table"]]
            out.append(math.log10(max(pred_mask(tb, m["pred"]).sum(), 1)))
        elif m["kind"] == "SCAN":
            out.append(math.log10(max(len(filter_rows(cat.tables[m["table"]], q.preds_of(m["alias"]))), 1)))
        else:
            sub = q.sub(m["aliases"])
            k = sub.key()
            if k not in cache:
                cache[k] = S.engine.true_count(sub)
            out.append(math.log10(max(cache[k], 1)))
    return out


def global_analysis(S: Estimators, queries: List[Query], name: str = "treelstm") -> dict:
    model = S.models[name][0]
    ds = S.dataset(queries, name)
    idx = np.arange(len(queries))
    batch = ds.batch(idx, model.use_tree)
    with torch.no_grad():
        z, tr = _mlp_input(model, batch, trace=True)
        acts, a = [], z
        for m in model.mlp:
            a = m(a)
            if isinstance(m, (nn.ReLU, nn.GELU)):
                acts.append(a.numpy())
        resid = model.out(a).squeeze(1).numpy()
    true_log = np.array([math.log10(max(S.engine.true_count(q), 1)) for q in queries])
    out: dict = {"model": name, "n_queries": len(queries)}

    # ── MLP layers ────────────────────────────────────────────────────────
    layers = [("input", z.numpy())] + [(f"hidden {i + 1}", x) for i, x in enumerate(acts)]
    sample = np.random.default_rng(0).choice(len(queries), size=min(600, len(queries)), replace=False)
    out["mlp_layers"] = []
    for lname, X in layers:
        xy, var = _pca2(X[sample])
        probe = _ridge(X, true_log, lam=10.0)
        out["mlp_layers"].append({"name": lname, "width": int(X.shape[1]), "probe_r2": probe["r2"],
                                  "pca": np.round(xy, 3).tolist(), "pca_var": var,
                                  "dead_frac": float((X.max(0) <= 0).mean()) if lname != "input" else 0.0})
    out["pca_color"] = np.round(true_log[sample], 3).tolist()
    groups = _input_groups(S, model)
    W = model.mlp[0].weight.detach().numpy()  # (units, inputs)
    order = np.argsort(-np.abs(W).sum(1))[:32]
    out["receptive_fields"] = {"groups": [g["name"] for g in groups],
                               "units": [[float(np.abs(W[u, g["idx"]]).mean()) for g in groups] for u in order]}
    ig = integrated_gradients(model, z)
    out["attribution"] = [{"group": g["name"], "mean_abs": float(np.abs(ig[:, g["idx"]].sum(1)).mean())}
                          for g in groups]
    out["residual"] = {"mean_abs": float(np.abs(resid).mean()), "frac_up": float((resid > 0).mean())}

    if not model.use_tree:
        return out

    # ── TreeLSTM levels ───────────────────────────────────────────────────
    H = tr["h"].numpy()
    meta = [m for t in ds.trees for m in t.meta]
    labels = [l for t in ds.trees for l in t.labels]
    kinds = np.array([m["kind"] for m in meta])
    cache: Dict[str, int] = {}
    targets = np.array([v for q, t in zip(queries, ds.trees) for v in _true_node_targets(S, q, t.meta, cache)])
    gates = {k: v.numpy() for k, v in tr["gates"].items()}
    levels = []
    probes = {}
    for k in KINDS:
        m = kinds == k
        if m.sum() < 20:
            continue
        p = _ridge(H[m], targets[m], lam=3.0)
        probes[k] = {"w": p["w"], "b": p["b"]}
        levels.append({"kind": k, "n_nodes": int(m.sum()), "probe_r2": p["r2"],
                       "mean_act": float(np.abs(H[m]).mean()),
                       "gate_input": float(gates["i"][m].mean()), "gate_output": float(gates["o"][m].mean()),
                       "gate_forget": float(gates["f"][m].mean()) if k != "AGG" else None,
                       "active_units": int((np.abs(H[m]).mean(0) > 0.25 * np.abs(H).mean(0).max()).sum())})
    out["tree_levels"] = levels
    out["probes"] = probes

    sample = np.random.default_rng(2).choice(len(H), size=min(1500, len(H)), replace=False)
    xy, var = _pca2(H[sample])
    out["node_map"] = {"xy": np.round(xy, 3).tolist(), "kind": kinds[sample].tolist(),
                       "label": [labels[i] for i in sample], "var": var,
                       "target": np.round(targets[sample], 2).tolist()}

    # neuron atlas: what each hidden unit detects
    units = []
    cols = np.array([f"{m.get('table', '')}.{m.get('col', '')}" if m["kind"] == "PRED" else "" for m in meta])
    for u in range(H.shape[1]):
        a = H[:, u]
        profile = [float(a[kinds == k].mean()) if (kinds == k).any() else 0.0 for k in KINDS]
        top = np.argsort(-a)[:200]
        tk = kinds[top]
        dom_kind = max(KINDS, key=lambda k: (tk == k).sum())
        detail = ""
        if dom_kind == "PRED":
            vc, cc = np.unique(cols[top][tk == "PRED"], return_counts=True)
            detail = f"{vc[cc.argmax()]} ({cc.max() / cc.sum():.0%})"
        elif dom_kind == "SCAN":
            tabs = np.array([meta[i].get("table", "") for i in top])[tk == "SCAN"]
            vt, ct = np.unique(tabs, return_counts=True)
            detail = f"scans of {vt[ct.argmax()]} ({ct.max() / ct.sum():.0%})"
        elif dom_kind == "JOIN":
            depth = np.array([len(meta[i]["aliases"]) for i in top])[tk == "JOIN"]
            detail = f"joins of ~{int(np.median(depth))} tables"
        corr = float(np.corrcoef(a, targets)[0, 1]) if a.std() > 1e-6 else 0.0
        seen, ex = set(), []
        for i in np.argsort(-a):
            if labels[i] not in seen:
                seen.add(labels[i])
                ex.append({"label": labels[i], "act": float(a[i])})
            if len(ex) == 4:
                break
        units.append({"unit": u, "profile": profile, "detects": dom_kind, "detail": detail,
                      "purity": float((tk == dom_kind).mean()), "card_corr": corr, "top": ex})
    out["neurons"] = units
    out["kinds"] = KINDS
    return out


def trace_query(S: Estimators, q: Query, probes: Dict[str, dict], name: str = "treelstm") -> dict:
    """Bottom-up walk of one query through both networks."""
    q = q.sub(q.rels)
    model = S.models[name][0]
    ds = S.dataset([q], name)
    batch = ds.batch(np.arange(1), model.use_tree)
    with torch.no_grad():
        z, tr = _mlp_input(model, batch, trace=True)
        acts, a = [], z
        for m in model.mlp:
            a = m(a)
            if isinstance(m, (nn.ReLU, nn.GELU)):
                acts.append(a[0].numpy())
        resid = float(model.out(a).squeeze(1)[0])
    ig = integrated_gradients(model, z)[0]
    groups = _input_groups(S, model)
    log_fj = float(batch["log_fj"][0])
    pred = math.expm1(log_fj + resid)
    out = {"estimate": max(pred, 1.0), "fj_estimate": math.expm1(log_fj), "residual": resid,
           "mlp": [{"name": f"hidden {i + 1}", "acts": np.round(x, 4).tolist()} for i, x in enumerate(acts)],
           "input": [{"group": g["name"], "values": np.round(z[0, g["idx"]].numpy(), 3).tolist(),
                      "attribution": float(ig[g["idx"]].sum())} for g in groups]}
    if model.use_tree:
        t = ds.trees[0]
        H = tr["h"].numpy()
        g = {k: v.numpy() for k, v in tr["gates"].items()}
        targets = _true_node_targets(S, q, t.meta, {})
        nodes = []
        for i, (lab, m) in enumerate(zip(t.labels, t.meta)):
            p = probes.get(m["kind"])
            readout = float(np.dot(H[i], p["w"]) + p["b"]) if p else None
            nodes.append({"id": i, "label": lab, "kind": m["kind"], "parent": int(t.parent[i]),
                          "height": int(t.height[i]), "h": np.round(H[i], 3).tolist(),
                          "gate_i": float(g["i"][i].mean()), "gate_o": float(g["o"][i].mean()),
                          "gate_f": float(g["f"][i].mean()) if t.parent[i] >= 0 else None,
                          "probe_log10": readout, "true_log10": targets[i]})
        out["nodes"] = nodes
    return out
