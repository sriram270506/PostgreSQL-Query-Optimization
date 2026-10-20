"""Shared training plumbing: featurised workloads, model registry, data fingerprints.

Model files: <models_dir>/<split>/<name>_s<seed>.pt, each carrying its config, feature
normalisation, the data fingerprint it was trained on and a format version, so a model
can never be silently served against different data.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import torch

from src.engine import Engine
from . import mscn
from .factorjoin import FactorJoin
from .neural import Dataset as TreeData, Featurizer, log_qerror
from .neural import predict as tree_predict
from .neural import train as tree_train

FORMAT_VERSION = 3  # v3: weights-only artifacts + node/join statistics features
NEURAL = ("mlp", "treelstm", "mscn")
DEFAULT_CFG = {
    "mlp": {"use_tree": False, "mlp_layers": 3, "mlp_width": 128, "dropout": 0.2, "activation": "gelu",
            "lr": 3e-3, "weight_decay": 1e-4, "batch": 64, "loss": "huber"},
    "treelstm": {"use_tree": True, "mlp_layers": 3, "mlp_width": 256, "dropout": 0.1, "activation": "gelu",
                 "lr": 3e-3, "weight_decay": 1e-4, "batch": 128, "loss": "mse", "embed": 64, "hidden": 64},
    "mscn": {"hidden": 256, "lr": 1e-3, "weight_decay": 1e-5, "batch": 128, "loss": "mse"},
}


def data_fingerprint(engine: Engine) -> str:
    """Content hash of every table (row hashes summed), stable across reloads."""
    h = hashlib.sha256()
    for t, tb in sorted(engine.catalog.tables.items()):
        h.update(t.encode())
        h.update(str(int(pd.util.hash_pandas_object(tb.df, index=False).sum())).encode())
    return h.hexdigest()[:16]


@dataclass
class Workload:
    df: pd.DataFrame
    trees: list
    stats: np.ndarray
    mscn_feats: list
    fz: Featurizer
    mf: mscn.MSCNFeaturizer
    engine: Engine
    fj: FactorJoin

    def split(self, scheme: str = "split") -> Dict[str, np.ndarray]:
        return {s: np.flatnonzero(self.df[scheme].to_numpy() == s) for s in ("train", "val", "test")}

    @property
    def y(self) -> np.ndarray:
        return self.df.true_count.to_numpy(float)


def load_workload(dataset: str) -> Workload:
    eng = Engine(dataset)
    fj = FactorJoin(eng.catalog)
    fz = Featurizer(eng.catalog, fj, eng.planner)
    mf = mscn.MSCNFeaturizer(eng.catalog)
    df = pd.read_csv(eng.ds.workload)
    cache = eng.ds.workload.replace(".csv", f".features.v{FORMAT_VERSION}.joblib")
    stamp = (os.path.getmtime(eng.ds.workload), len(df))
    c = joblib.load(cache) if os.path.exists(cache) else None
    if c is None or c.get("stamp") != stamp:
        qs = [eng.parse(s) for s in df.sql]
        c = {"stamp": stamp, "trees": [fz.tree(q) for q in qs], "stats": np.stack([fz.stats(q) for q in qs]),
             "mscn": [mf.featurize(q) for q in qs]}
        joblib.dump(c, cache)
    return Workload(df, c["trees"], c["stats"], c["mscn"], fz, mf, eng, fj)


def make_tree_data(wl: Workload, mean, std, trees=None, stats=None, y=True) -> TreeData:
    return TreeData(trees if trees is not None else wl.trees, stats if stats is not None else wl.stats,
                    wl.y if y else None, mean, std, wl.fz.stats_names.index("log_fj"))


def train_one(wl: Workload, name: str, cfg: dict, split: str = "split", seed: int = 0,
              max_epochs: int = 150, patience: int = 15, report=None):
    """Train one model; returns (model, bundle) where bundle holds everything to serve it."""
    sp = wl.split(split)
    if name == "mscn":
        data = mscn.MSCNData(wl.mscn_feats, wl.y)
        dims = (wl.mf.t_dim, wl.mf.j_dim, wl.mf.p_dim)
        model, info = mscn.train(cfg, data, sp["train"], sp["val"], dims, max_epochs, patience, seed, report)
        val_pred = mscn.predict(model, data, sp["val"])
        bundle = {"kind": "mscn", "cfg": cfg, "dims": dims}
    else:
        mean = wl.stats[sp["train"]].mean(0)
        std = wl.stats[sp["train"]].std(0) + 1e-6
        data = make_tree_data(wl, mean, std)
        model, info = tree_train(cfg, data, sp["train"], sp["val"], wl.fz.dim, max_epochs, patience, seed, report)
        val_pred = tree_predict(model, data, sp["val"])
        bundle = {"kind": "fusion", "cfg": cfg, "mean": mean, "std": std, "node_dim": wl.fz.dim,
                  "stats_names": wl.fz.stats_names}
    bundle.update(state=model.state_dict(), seed=seed, split=split, version=FORMAT_VERSION,
                  data_hash=data_fingerprint(wl.engine), dataset=wl.engine.ds.name,
                  history=info["history"], best_val=info["best_val"],
                  val_mean_log_q=float(log_qerror(val_pred, wl.y[sp["val"]]).mean()))
    return model, bundle


def model_path(dataset_models_dir: str, split: str, name: str, seed: int) -> str:
    return os.path.join(dataset_models_dir, split, f"{name}_s{seed}.pt")


def _plain(x):
    """Only tensors and plain Python types, so artifacts load with weights_only=True."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(x))
    if isinstance(x, np.generic):
        return x.item()
    return x


def save(bundle: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(_plain(bundle), tmp)
    os.replace(tmp, path)


class InvalidArtifact(ValueError):
    pass


def load_bundle(path: str) -> dict:
    """Load a model artifact without executing pickled code (weights_only=True) and
    validate its schema before anything uses it."""
    try:
        b = torch.load(path, weights_only=True)
    except Exception as e:
        raise InvalidArtifact(f"{path}: not a weights-only artifact ({str(e).splitlines()[0]})") from None
    req = {"kind": str, "cfg": dict, "state": dict, "version": int, "data_hash": str, "dataset": str}
    for k, t in req.items():
        if not isinstance(b.get(k), t):
            raise InvalidArtifact(f"{path}: field {k!r} missing or not {t.__name__}")
    if b["kind"] not in ("fusion", "mscn"):
        raise InvalidArtifact(f"{path}: unknown model kind {b['kind']!r}")
    if not all(isinstance(v, torch.Tensor) for v in b["state"].values()):
        raise InvalidArtifact(f"{path}: state must contain tensors only")
    if b["kind"] == "fusion":
        names = b.get("stats_names")
        if not (isinstance(names, list) and all(isinstance(n, str) for n in names) and isinstance(b.get("node_dim"), int)):
            raise InvalidArtifact(f"{path}: bad stats_names/node_dim")
        for k in ("mean", "std"):
            v = b.get(k)
            if not (isinstance(v, torch.Tensor) and v.dim() == 1 and len(v) == len(names)):
                raise InvalidArtifact(f"{path}: {k} must be a 1-D tensor of len(stats_names)")
            b[k] = v.numpy()
    else:
        dims = b.get("dims")
        if not (isinstance(dims, list) and len(dims) == 3 and all(isinstance(d, int) for d in dims)):
            raise InvalidArtifact(f"{path}: bad dims")
        b["dims"] = tuple(dims)
    return b


def tuned_cfg(models_dir: str, name: str) -> dict:
    """The tail-aware selection (scripts.select_model) if present, else the Optuna best."""
    for f in (f"{name}.selected.json", f"{name}.json"):
        p = os.path.join(models_dir, "tuning", f)
        if os.path.exists(p):
            with open(p) as fh:
                return json.load(fh)["final_cfg"]
    return dict(DEFAULT_CFG[name])


# ── serving-side helpers (shared by scripts.train, scripts.benchmark and the API) ──

def build_model(bundle: dict):
    from .neural import FusionModel
    if bundle["kind"] == "mscn":
        m = mscn.SetConv(*bundle["dims"], bundle["cfg"]["hidden"])
    else:
        m = FusionModel(bundle["node_dim"], len(bundle["stats_names"]), bundle["cfg"])
    m.load_state_dict(bundle["state"])
    m.eval()
    return m


def predict_log(model, bundle: dict, trees, stats, mfeats) -> np.ndarray:
    """log(1+rows) for pre-featurised queries (any of trees/stats or mfeats may be unused)."""
    with torch.inference_mode():
        if bundle["kind"] == "mscn":
            data = mscn.MSCNData(mfeats)
            return mscn.predict(model, data)
        ds = TreeData(trees, stats, None, bundle["mean"], bundle["std"], bundle["stats_names"].index("log_fj"))
        return tree_predict(model, ds)


def load_ensembles(models_dir: str, split: str = "split") -> Dict[str, list]:
    out: Dict[str, list] = {}
    d = os.path.join(models_dir, split)
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if f.endswith(".pt") and "_s" in f:
            name = f.rsplit("_s", 1)[0]
            try:
                b = load_bundle(os.path.join(d, f))
                if b["version"] != FORMAT_VERSION:
                    continue  # incompatible feature format: never serve it
                model = build_model(b)  # strict load: shapes must match the declared config
            except (InvalidArtifact, RuntimeError, KeyError) as e:
                print(f"skipping model artifact {f}: {e}")
                continue
            out.setdefault(name, []).append((model, b))
    return out


_STACKED: Dict[int, object] = {}


def ensemble_log(members: list, trees, stats, mfeats):
    """Mean and std of log(1+rows) across ensemble members. Fusion ensembles run as one
    stacked model (one tree walk for all members); MSCN members run one by one."""
    b0 = members[0][1]
    if b0["kind"] == "fusion" and all(b["cfg"] == b0["cfg"] for _, b in members):
        from .neural import StackedFusion
        key = id(members)
        if key not in _STACKED:
            _STACKED[key] = StackedFusion([m for m, _ in members])
        ds = TreeData(trees, stats, None, b0["mean"], b0["std"], b0["stats_names"].index("log_fj"))
        P = []
        for s in range(0, len(trees), 512):
            P.append(_STACKED[key](ds.batch(np.arange(s, min(s + 512, len(trees))), b0["cfg"].get("use_tree", True))).numpy())
        P = np.concatenate(P, axis=1)
    elif b0["kind"] == "mscn":
        key = id(members)
        if key not in _STACKED:
            _STACKED[key] = mscn.StackedSetConv([m for m, _ in members])
        data = mscn.MSCNData(mfeats)
        P = np.concatenate([_STACKED[key](data.batch(np.arange(s, min(s + 512, len(mfeats))))).numpy()
                            for s in range(0, len(mfeats), 512)], axis=1)
    else:
        P = np.stack([predict_log(m, b, trees, stats, mfeats) for m, b in members])
    return P.mean(0), P.std(0), P
