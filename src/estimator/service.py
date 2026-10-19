"""All estimators of one dataset behind one guarded API.

Serving guarantees, in order:
  1. learned models are multi-seed ensembles (mean of log estimates);
  2. if the ensemble disagrees more than its validation-calibrated threshold (tau), the
     answer falls back to FactorJoin, but only for models where that fallback beat the
     ensemble on the uncertain validation queries (decided per model at training time);
  3. any exception falls back along neural -> FactorJoin -> PostgreSQL emulation;
  4. every answer is clamped to [1, provable upper bound];
  5. a model trained on different data (fingerprint mismatch) is flagged stale.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.engine import Engine, Query
from .bounds import Bounds
from .factorjoin import FactorJoin
from .mscn import MSCNFeaturizer
from .neural import Featurizer
from .training import NEURAL, data_fingerprint, ensemble_log, load_ensembles

# Estimators served to users. PostgreSQL itself is measured on the real server
# (postgres_live / postgres_xstats); the in-process planner emulation ("postgres") is only
# an internal model feature and the last step of the fallback chain, never reported.
ESTIMATORS = ["factorjoin", "mscn", "mlp", "treelstm"]
LABELS = {"postgres_live": "PostgreSQL 16", "postgres_xstats": "PostgreSQL 16 + CREATE STATISTICS",
          "factorjoin": "FactorJoin (BN)", "mscn": "MSCN (baseline)", "mlp": "Fusion MLP",
          "treelstm": "TreeLSTM + MLP"}


class Estimators:
    def __init__(self, dataset: str = "hr", split: str = "split", feedback_path: Optional[str] = None,
                 models_dir: Optional[str] = None):
        """`models_dir` overrides where learned models come from (drift experiment: serve
        models trained on an older snapshot against today's data)."""
        self.engine = Engine(dataset)
        self.ds = self.engine.ds
        self.fj = FactorJoin(self.engine.catalog)
        self.fz = Featurizer(self.engine.catalog, self.fj, self.engine.planner)
        self.mf = MSCNFeaturizer(self.engine.catalog)
        self.bounds = Bounds(self.engine.catalog)
        mdir = models_dir or self.ds.models_dir
        self.ensembles = load_ensembles(mdir, split)
        meta_path = os.path.join(mdir, split, "ensemble.json")
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        self.fingerprint = data_fingerprint(self.engine)
        self.stale = {n: any(b["data_hash"] != self.fingerprint for _, b in m) for n, m in self.ensembles.items()}
        self.feedback_path = feedback_path or os.path.join(self.ds.models_dir, "feedback.jsonl")
        self._lock = threading.Lock()

    # ── raw estimators ───────────────────────────────────────────────────────
    def _neural(self, qs: List[Query], name: str) -> Tuple[np.ndarray, np.ndarray]:
        members = self.ensembles[name]
        if name == "mscn":
            trees, stats, mfeats = None, None, [self.mf.featurize(q) for q in qs]
        else:
            trees = [self.fz.tree(q) for q in qs]
            stats = np.stack([self.fz.stats(q) for q in qs])
            mfeats = None
        mean, std, _ = ensemble_log(members, trees, stats, mfeats)
        return np.expm1(mean.astype(np.float64)), std

    def _raw(self, qs: List[Query], name: str) -> Tuple[np.ndarray, np.ndarray]:
        if name in self.ensembles:
            return self._neural(qs, name)
        f = self.engine.planner.estimate if name == "postgres" else self.fj.estimate
        return np.array([f(q) for q in qs], dtype=float), np.zeros(len(qs))

    # ── guarded API ──────────────────────────────────────────────────────────
    def estimate_many(self, qs: List[Query], name: str, guard: bool = True) -> Dict[str, np.ndarray]:
        """Estimates + per-query flags: fallback (uncertain or failed), clamped, std."""
        qs = [q.sub(q.rels) for q in qs]
        n = len(qs)
        fallback = np.zeros(n, dtype=bool)
        try:
            est, std = self._raw(qs, name)
        except Exception:  # never fail a request because one estimator broke
            chain = [x for x in ("factorjoin", "postgres") if x != name]
            for alt in chain:
                try:
                    est, std = self._raw(qs, alt)
                    break
                except Exception:
                    continue
            else:
                raise
            fallback[:] = True
        if guard and name in self.ensembles and self.meta.get(name, {}).get("fallback"):
            unsure = std > self.meta[name]["tau"]
            if unsure.any():
                fj = np.array([self.fj.estimate(q) for q, u in zip(qs, unsure) if u])
                est = est.copy()
                est[unsure] = fj
                fallback |= unsure
        clamped = np.zeros(n, dtype=bool)
        if guard:
            ub = np.array([self.bounds.upper(q) for q in qs])
            clamped = est > ub
            est = np.clip(est, 1.0, ub)
        else:
            est = np.maximum(est, 1.0)
        return {"est": est, "std": std, "fallback": fallback, "clamped": clamped}

    def estimate(self, q: Query, name: str) -> Tuple[float, float, dict]:
        t0 = time.perf_counter()
        r = self.estimate_many([q], name)
        ms = (time.perf_counter() - t0) * 1000
        return float(r["est"][0]), ms, {"std": float(r["std"][0]), "fallback": bool(r["fallback"][0]),
                                        "clamped": bool(r["clamped"][0])}

    def member(self, name: str):
        """First ensemble member (model, bundle): used by the internals visualisation."""
        return self.ensembles[name][0]

    def tree_data(self, qs: List[Query], bundle: dict):
        from .neural import Dataset as TreeData
        trees = [self.fz.tree(q) for q in qs]
        stats = np.stack([self.fz.stats(q) for q in qs])
        return TreeData(trees, stats, None, bundle["mean"], bundle["std"], bundle["stats_names"].index("log_fj"))

    def available(self) -> List[str]:
        return [n for n in ESTIMATORS if n == "factorjoin" or n in self.ensembles]

    # ── feedback loop (LEO-style): log estimate vs actual, watch for drift ──────
    def log_feedback(self, sql: str, true: int, estimates: Dict[str, float]):
        rec = {"ts": time.time(), "sql": sql, "true": int(true), "est": {k: float(v) for k, v in estimates.items()}}
        with self._lock, open(self.feedback_path, "a", encoding="utf8") as f:
            f.write(json.dumps(rec) + "\n")

    def feedback_summary(self, window: int = 500) -> dict:
        if not os.path.exists(self.feedback_path):
            return {"n": 0}
        with open(self.feedback_path, encoding="utf8") as f:
            recs = [json.loads(l) for l in f.readlines()[-window:]]
        out = {"n": len(recs), "estimators": {}}
        for name in {k for r in recs for k in r["est"]}:
            q = np.array([float(qerror(r["est"][name], r["true"])) for r in recs if name in r["est"]])
            out["estimators"][name] = {"n": int(len(q)), "median": float(np.median(q)),
                                       "p95": float(np.percentile(q, 95)), "max": float(q.max())}
        return out


def qerror(est, true) -> np.ndarray:
    est, true = np.maximum(np.asarray(est, float), 1.0), np.maximum(np.asarray(true, float), 1.0)
    return np.maximum(est / true, true / est)


__all__ = ["Estimators", "ESTIMATORS", "LABELS", "NEURAL", "qerror"]
