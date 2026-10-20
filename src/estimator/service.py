"""All estimators of one dataset behind one guarded API.

Serving guarantees, in order:
  1. learned models are multi-seed ensembles (mean of log estimates);
  2. a model trained on different data (fingerprint mismatch) is never served as if healthy:
     it answers with FactorJoin (marked stale) and readiness fails until it is retrained;
  3. if the ensemble disagrees more than its validation-calibrated threshold (tau), the
     answer falls back to FactorJoin, but only for models where that policy beat the
     ensemble on validation (decided per model at training time);
  4. any exception falls back along neural -> FactorJoin -> internal planner model;
  5. every answer is clamped to [1, provable upper bound].
"""
from __future__ import annotations

import contextlib
import hashlib
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
FEEDBACK_KEEP = 20000             # newest records kept after a trim
FEEDBACK_MAX_BYTES = 12_000_000   # trim once the log grows past this


@contextlib.contextmanager
def _file_lock(path: str, timeout: float = 5.0):
    """Cross-process exclusive lock via an O_EXCL lock file (Windows and POSIX)."""
    t0 = time.monotonic()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() - t0 > timeout:  # left behind by a crashed writer
                with contextlib.suppress(OSError):
                    os.remove(path)
            time.sleep(0.01)
    try:
        yield
    finally:
        os.close(fd)
        with contextlib.suppress(OSError):
            os.remove(path)


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
        # experiments may deliberately serve stale models (the drift study); production never does
        self.serve_stale = models_dir is not None
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
        """Estimates + per-query flags: fallback (uncertain, failed or stale), clamped, std."""
        qs = [q.sub(q.rels) for q in qs]
        n = len(qs)
        fallback = np.zeros(n, dtype=bool)
        stale = bool(self.stale.get(name)) and not self.serve_stale
        try:
            est, std = self._raw(qs, "factorjoin" if stale else name)
            fallback[:] = stale
        except Exception:  # never fail a request because one estimator broke
            for alt in [x for x in ("factorjoin", "postgres") if x != name]:
                try:
                    est, std = self._raw(qs, alt)
                    break
                except Exception:
                    continue
            else:
                raise
            fallback[:] = True
        if guard and not stale and name in self.ensembles and self.meta.get(name, {}).get("fallback"):
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
        return {"est": est, "std": std, "fallback": fallback, "clamped": clamped, "stale": stale}

    def estimate(self, q: Query, name: str) -> Tuple[float, float, dict]:
        t0 = time.perf_counter()
        r = self.estimate_many([q], name)
        ms = (time.perf_counter() - t0) * 1000
        return float(r["est"][0]), ms, {"std": float(r["std"][0]), "fallback": bool(r["fallback"][0]),
                                        "clamped": bool(r["clamped"][0]), "stale": bool(r["stale"])}

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

    def healthy(self) -> bool:
        """Ready to serve: no learned model was trained on different data."""
        return not any(self.stale.values())

    # ── feedback loop (LEO-style): log estimate vs actual, watch for drift ──────
    def log_feedback(self, q: Query, true: int, estimates: Dict[str, float]):
        """Append one record: a hash and literal-free template of the query, not the SQL
        (FEEDBACK_STORE_SQL=1 opts in). Locked across processes; trimmed to the newest
        FEEDBACK_KEEP records once the file passes FEEDBACK_MAX_BYTES."""
        from src.workload.generator import template
        canon = q.canonical().to_sql()
        rec = {"ts": time.time(), "qhash": hashlib.sha256(canon.encode()).hexdigest()[:16],
               "template": template(q), "n_tables": len(q.rels), "true": int(true),
               "est": {k: float(v) for k, v in estimates.items()}}
        if os.environ.get("FEEDBACK_STORE_SQL") == "1":
            rec["sql"] = canon
        with self._lock, _file_lock(self.feedback_path + ".lock"):
            with open(self.feedback_path, "a", encoding="utf8") as f:
                f.write(json.dumps(rec) + "\n")
            if os.path.getsize(self.feedback_path) > FEEDBACK_MAX_BYTES:
                with open(self.feedback_path, encoding="utf8") as f:
                    keep = f.readlines()[-FEEDBACK_KEEP:]
                tmp = self.feedback_path + ".tmp"
                with open(tmp, "w", encoding="utf8") as f:
                    f.writelines(keep)
                os.replace(tmp, self.feedback_path)

    def feedback_summary(self, window: int = 500) -> dict:
        if not os.path.exists(self.feedback_path):
            return {"n": 0, "skipped_malformed": 0, "estimators": {}}
        recs, bad = [], 0
        with open(self.feedback_path, encoding="utf8") as f:
            for line in f.readlines()[-window:]:
                try:
                    r = json.loads(line)
                    if isinstance(r, dict) and isinstance(r.get("est"), dict) and isinstance(r.get("true"), (int, float)):
                        recs.append(r)
                        continue
                except ValueError:
                    pass
                bad += 1  # malformed record: skipped, never breaks the summary
        out = {"n": len(recs), "skipped_malformed": bad, "estimators": {}}
        for name in {k for r in recs for k in r["est"]}:
            q = np.array([float(qerror(r["est"][name], r["true"])) for r in recs
                          if isinstance(r["est"].get(name), (int, float))])
            if len(q):
                out["estimators"][name] = {"n": int(len(q)), "median": float(np.median(q)),
                                           "p95": float(np.percentile(q, 95)), "max": float(q.max())}
        return out


def qerror(est, true) -> np.ndarray:
    est, true = np.maximum(np.asarray(est, float), 1.0), np.maximum(np.asarray(true, float), 1.0)
    return np.maximum(est / true, true / est)


__all__ = ["Estimators", "ESTIMATORS", "LABELS", "NEURAL", "qerror"]
