"""One object that owns the engine and every estimator, for the API and benchmarks."""
from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.engine import Engine, Query
from .factorjoin import FactorJoin
from .neural import Dataset, Featurizer, FusionModel

ESTIMATORS = ["postgres", "factorjoin", "mlp", "treelstm"]
LABELS = {"postgres": "PostgreSQL planner", "factorjoin": "FactorJoin (BN)",
          "mlp": "Fusion MLP", "treelstm": "TreeLSTM + MLP"}


class Estimators:
    def __init__(self, csv_dir: str = "data/out", models_dir: str = "models"):
        self.engine = Engine(csv_dir)
        self.fj = FactorJoin(self.engine.catalog)
        self.fz = Featurizer(self.engine.catalog, self.fj, self.engine.planner)
        self.models: Dict[str, Tuple[FusionModel, dict]] = {}
        for name in ("mlp", "treelstm"):
            p = os.path.join(models_dir, f"fusion_{name}.pt")
            if os.path.exists(p):
                b = torch.load(p, weights_only=False)
                m = FusionModel(b["node_dim"], len(b["stats_names"]), b["cfg"])
                m.load_state_dict(b["state"])
                m.eval()
                self.models[name] = (m, b)

    def dataset(self, qs: List[Query], name: str) -> Dataset:
        b = self.models[name][1]
        trees = [self.fz.tree(q) for q in qs]
        stats = np.stack([self.fz.stats(q) for q in qs])
        return Dataset(trees, stats, None, b["mean"], b["std"], self.fz.stats_names.index("log_fj"))

    def neural(self, qs: List[Query], name: str) -> np.ndarray:
        m = self.models[name][0]
        ds = self.dataset(qs, name)
        with torch.no_grad():
            out = m(ds.batch(np.arange(len(qs)), m.use_tree)).numpy()
        return np.maximum(np.expm1(out.astype(np.float64)), 1.0)

    def estimate(self, q: Query, name: str) -> Tuple[float, float]:
        """(estimate, latency ms) for one COUNT(*) query, including feature extraction."""
        q = q.sub(q.rels)
        t0 = time.perf_counter()
        if name == "postgres":
            est = self.engine.planner.estimate(q)
        elif name == "factorjoin":
            est = self.fj.estimate(q)
        else:
            est = float(self.neural([q], name)[0])
        return float(est), (time.perf_counter() - t0) * 1000

    def estimate_many(self, qs: List[Query], name: str) -> np.ndarray:
        if name in self.models:
            return self.neural(qs, name)
        f = self.engine.planner.estimate if name == "postgres" else self.fj.estimate
        return np.array([f(q) for q in qs])

    def available(self) -> List[str]:
        return [n for n in ESTIMATORS if n in ("postgres", "factorjoin") or n in self.models]


def qerror(est, true) -> np.ndarray:
    est, true = np.maximum(np.asarray(est, float), 1.0), np.maximum(np.asarray(true, float), 1.0)
    return np.maximum(est / true, true / est)
