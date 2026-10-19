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

FALLBACK_QUANTILE = 0.95  # ensemble spread above the val 95th percentile -> trust FactorJoin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hr")
    ap.add_argument("--models", default="mlp,treelstm,mscn")
    ap.add_argument("--split", default="split")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--calibrate-only", action="store_true", help="recompute fallback calibration only")
    args = ap.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
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


def calibrate(wl, v, members) -> dict:
    """Uncertainty threshold + whether falling back to FactorJoin actually helps.

    tau = 95th percentile of the ensemble's log-spread on validation. The fallback is only
    switched on if, on the validation queries above tau, FactorJoin's estimate is closer to
    the truth than the ensemble's: a guard has to earn its place on held-out data."""
    mfeats = [wl.mscn_feats[i] for i in v]
    trees = [wl.trees[i] for i in v]
    mean, std, _ = ensemble_log(members, trees, wl.stats[v], mfeats)
    tau = float(np.quantile(std, FALLBACK_QUANTILE))
    unsure = std > tau
    fj = np.log1p(np.array([wl.fj.estimate(wl.engine.parse(wl.df.sql.iloc[i])) for i in np.asarray(v)[unsure]]))
    nn_err = float(log_qerror(mean[unsure], wl.y[v][unsure]).mean()) if unsure.any() else 0.0
    fj_err = float(log_qerror(fj, wl.y[v][unsure]).mean()) if unsure.any() else 0.0
    return {"tau": tau, "fallback": bool(unsure.any() and fj_err < nn_err),
            "val_uncertain_n": int(unsure.sum()), "val_uncertain_mean_log_q": {"ensemble": nn_err, "factorjoin": fj_err},
            "val_mean_log_q_ensemble": float(log_qerror(mean, wl.y[v]).mean())}


if __name__ == "__main__":
    main()
