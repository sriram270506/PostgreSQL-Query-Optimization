"""Hyperparameter tuning (Optuna TPE + median pruning) for both learned estimators, then
retrain the best configuration of each and save it.

    python -m scripts.tune --mlp-trials 40 --tree-trials 24

Objective: mean log Q-error on the validation split (log Q = |log est - log true|, so it
punishes over- and under-estimates equally and is dominated by the bad tail).
Outputs: models/fusion_mlp.pt, models/fusion_treelstm.pt, models/tuning/{mlp,treelstm}.json
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import optuna
import pandas as pd
import torch

from src.engine import Engine
from src.estimator.factorjoin import FactorJoin
from src.estimator.neural import Dataset, Featurizer, log_qerror, predict, train

WORKLOAD = "data/workload/workload.csv"
CACHE = "data/workload/features.joblib"


def load_data():
    eng = Engine()
    fj = FactorJoin(eng.catalog)
    fz = Featurizer(eng.catalog, fj, eng.planner)
    df = pd.read_csv(WORKLOAD)
    mtime = os.path.getmtime(WORKLOAD)
    if os.path.exists(CACHE) and joblib.load(CACHE)["mtime"] == mtime:
        c = joblib.load(CACHE)
        trees, stats = c["trees"], c["stats"]
    else:
        qs = [eng.parse(s) for s in df.sql]
        trees, stats = [fz.tree(q) for q in qs], np.stack([fz.stats(q) for q in qs])
        joblib.dump({"mtime": mtime, "trees": trees, "stats": stats}, CACHE)
    split = {s: np.flatnonzero(df.split == s) for s in ("train", "val", "test")}
    mean = stats[split["train"]].mean(0)
    std = stats[split["train"]].std(0) + 1e-6
    ds = Dataset(trees, stats, df.true_count.to_numpy(float), mean, std, fz.stats_names.index("log_fj"))
    return df, fz, ds, split, mean, std


def space(trial: optuna.Trial, use_tree: bool) -> dict:
    cfg = {
        "use_tree": use_tree,
        "mlp_layers": trial.suggest_int("mlp_layers", 1, 4),
        "mlp_width": trial.suggest_categorical("mlp_width", [32, 64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch": trial.suggest_categorical("batch", [64, 128, 256]),
        "loss": trial.suggest_categorical("loss", ["mse", "huber"]),
    }
    if use_tree:
        cfg["embed"] = trial.suggest_categorical("embed", [32, 64, 128])
        cfg["hidden"] = trial.suggest_categorical("hidden", [32, 64, 128])
    return cfg


def metrics(pred_log, true):
    lq = log_qerror(pred_log, true)
    q = np.exp(lq)
    return {"mean_log_q": float(lq.mean()), "median_q": float(np.median(q)),
            "p90_q": float(np.percentile(q, 90)), "p95_q": float(np.percentile(q, 95)),
            "p99_q": float(np.percentile(q, 99)), "max_q": float(q.max()),
            "within_2x": float((q <= 2).mean())}


def run_stage(name, use_tree, n_trials, ds, fz, split, mean, std, max_epochs):
    def objective(trial):
        cfg = space(trial, use_tree)

        def report(ep, val):
            trial.report(val, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        _, info = train(cfg, ds, split["train"], split["val"], fz.dim, max_epochs=max_epochs,
                        patience=6, report=report)
        trial.set_user_attr("epochs", len(info["history"]))
        return info["best_val"]

    study = optuna.create_study(direction="minimize", study_name=name,
                                sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=6))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    tune_s = time.time() - t0
    best = space(optuna.trial.FixedTrial(study.best_params), use_tree)
    print(f"[{name}] best val {study.best_value:.4f} params {study.best_params}")

    t1 = time.time()
    model, info = train(best, ds, split["train"], split["val"], fz.dim, max_epochs=150, patience=15)
    train_s = time.time() - t1
    test = metrics(predict(model, ds, split["test"]), ds.y[split["test"]])
    print(f"[{name}] test {test}")
    n_params = sum(p.numel() for p in model.parameters())
    torch.save({"cfg": best, "state": model.state_dict(), "mean": mean, "std": std,
                "node_dim": fz.dim, "stats_names": fz.stats_names}, f"models/fusion_{name}.pt")
    os.makedirs("models/tuning", exist_ok=True)
    trials = [{"number": t.number, "state": t.state.name, "value": t.value, "params": t.params,
               "curve": [t.intermediate_values[k] for k in sorted(t.intermediate_values)]}
              for t in study.trials]
    importance = {}
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception:  # needs >= 2 completed trials with varied params
        pass
    with open(f"models/tuning/{name}.json", "w") as f:
        json.dump({"model": name, "best_params": study.best_params, "best_val": study.best_value,
                   "final_cfg": best, "final_history": info["history"], "test": test,
                   "n_params": n_params, "tune_seconds": tune_s, "train_seconds": train_s,
                   "importance": importance, "trials": trials}, f, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlp-trials", type=int, default=40)
    ap.add_argument("--tree-trials", type=int, default=24)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--only", choices=["mlp", "treelstm"])
    args = ap.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df, fz, ds, split, mean, std = load_data()
    fjm = metrics(np.log1p(df.fj_estimate.to_numpy()[split["test"]]), ds.y[split["test"]])
    print("FactorJoin test", fjm)
    if args.only != "treelstm":
        run_stage("mlp", False, args.mlp_trials, ds, fz, split, mean, std, args.max_epochs)
    if args.only != "mlp":
        run_stage("treelstm", True, args.tree_trials, ds, fz, split, mean, std, args.max_epochs)


if __name__ == "__main__":
    main()
