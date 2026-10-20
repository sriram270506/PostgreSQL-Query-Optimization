"""Train multi-seed ensembles from the tuned configs.

    python -m scripts.train --dataset hr --seeds 5
    python -m scripts.train --dataset hr --split split_shape --seeds 3   # OOD experiment

Writes <models_dir>/<split>/<name>_s<seed>.pt. Seeds differ in initialisation and batch
order; their spread is used both for confidence intervals and as the uncertainty signal
that triggers the FactorJoin fallback.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from src.estimator.neural import log_qerror
from src.estimator.training import (build_model, ensemble_log, load_workload, model_path, save, train_one,
                                    tuned_cfg)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hr")
    ap.add_argument("--models", default="mlp,treelstm,mscn")
    ap.add_argument("--split", default="split")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--calibrate-only", action="store_true", help="recompute fallback calibration only")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", min(8, os.cpu_count() or 1))))
    wl = load_workload(args.dataset)
    mdir = wl.engine.ds.models_dir
    sp = wl.split(args.split)
    meta_path = os.path.join(mdir, args.split, "ensemble.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if args.calibrate_only:  # re-score existing ensembles (no training)
        from src.estimator.training import load_ensembles
        for name, members in load_ensembles(mdir, args.split).items():
            meta[name] = {**meta.get(name, {}), **calibrate(wl, sp["val"], members), "seeds": len(members)}
            print(name, {k: meta[name][k] for k in ("tau", "fallback", "val_uncertain_mean_log_q")}, flush=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
        return
    for name in args.models.split(","):
        cfg = tuned_cfg(mdir, name)
        members = []
        for seed in range(args.seeds):
            t0 = time.time()
            _, bundle = train_one(wl, name, cfg, split=args.split, seed=seed)
            bundle["train_seconds"] = time.time() - t0
            save(bundle, model_path(mdir, args.split, name, seed))
            members.append((build_model(bundle), bundle))
            print(f"[{args.dataset}/{args.split}] {name} seed {seed}: val mean log-q "
                  f"{bundle['val_mean_log_q']:.4f} ({bundle['train_seconds']:.0f}s)", flush=True)
        meta[name] = {**calibrate(wl, sp["val"], members), "seeds": len(members), "cfg": cfg,
                      "train_seconds": [b.get("train_seconds", 0) for _, b in members]}
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)


TAU_GRID = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99)


def calibrate(wl, v, members) -> dict:
    """Uncertainty threshold + whether falling back to FactorJoin actually helps.

    tau is grid-searched over quantiles of the ensemble's log-spread on validation, picking
    the one that minimises validation mean + CVaR95 log Q-error *with the fallback applied*.
    The fallback is switched on only if that beats never falling back: a guard has to earn
    its place on held-out data. Also reports how well the spread ranks errors (Spearman) and
    the error of the most-confident x% (coverage curve)."""
    from scipy.stats import spearmanr
    from src.estimator.neural import val_score
    v = np.asarray(v)
    mfeats = [wl.mscn_feats[i] for i in v]
    trees = [wl.trees[i] for i in v]
    mean, std, _ = ensemble_log(members, trees, wl.stats[v], mfeats)
    y = wl.y[v]
    fj = np.log1p(wl.df.fj_estimate.to_numpy(float)[v])
    e_nn, e_fj = log_qerror(mean, y), log_qerror(fj, y)
    tail = {"select": "tail"}
    base = val_score(e_nn, tail)
    grid = []
    for qt in TAU_GRID:
        tau = float(np.quantile(std, qt))
        e = np.where(std > tau, e_fj, e_nn)
        grid.append({"quantile": qt, "tau": tau, "score": val_score(e, tail), "mean_log_q": float(e.mean())})
    best = min(grid, key=lambda g: g["score"])
    on = best["score"] < base - 1e-4
    tau = best["tau"] if on else float(np.quantile(std, 0.95))
    unsure = std > tau
    order = np.argsort(std)
    coverage = [{"coverage": c, "mean_log_q": float(e_nn[order[:max(1, int(c * len(v)))]].mean())}
                for c in (0.5, 0.7, 0.8, 0.9, 0.95, 1.0)]
    rho = spearmanr(std, e_nn).correlation if len(v) > 2 else float("nan")
    return {"tau": tau, "tau_quantile": best["quantile"] if on else 0.95, "fallback": bool(on),
            "tau_grid": grid, "val_score_no_fallback": base,
            "val_uncertain_n": int(unsure.sum()),
            "val_uncertain_mean_log_q": {"ensemble": float(e_nn[unsure].mean()) if unsure.any() else 0.0,
                                         "factorjoin": float(e_fj[unsure].mean()) if unsure.any() else 0.0},
            "spearman_std_vs_error": float(rho), "coverage_curve": coverage,
            "val_mean_log_q_ensemble": float(e_nn.mean())}


if __name__ == "__main__":
    main()
