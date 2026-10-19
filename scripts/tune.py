"""Hyperparameter tuning (Optuna TPE + median pruning) for the learned estimators.

    python -m scripts.tune --dataset hr --models mlp,treelstm,mscn --trials 40,24,20

Objective: mean log Q-error on the validation split (|log est - log true|, punishes over-
and under-estimates equally, dominated by the tail). Writes
<models_dir>/tuning/<name>.json (trials, importances, best config); scripts.train then
trains the multi-seed ensembles from that config.
"""
import argparse
import json
import os
import time

import optuna
import torch

from src.estimator.training import load_workload, train_one


def space(trial: optuna.Trial, name: str) -> dict:
    if name == "mscn":
        return {"hidden": trial.suggest_categorical("hidden", [64, 128, 256, 512]),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
                "batch": trial.suggest_categorical("batch", [64, 128, 256]),
                "loss": trial.suggest_categorical("loss", ["mse", "huber"])}
    cfg = {
        "use_tree": name == "treelstm",
        "mlp_layers": trial.suggest_int("mlp_layers", 1, 4),
        "mlp_width": trial.suggest_categorical("mlp_width", [32, 64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch": trial.suggest_categorical("batch", [64, 128, 256]),
        "loss": trial.suggest_categorical("loss", ["mse", "huber"]),
    }
    if name == "treelstm":
        cfg["embed"] = trial.suggest_categorical("embed", [32, 64, 128])
        cfg["hidden"] = trial.suggest_categorical("hidden", [32, 64, 128])
    return cfg


def tune(wl, name: str, n_trials: int, max_epochs: int, out_dir: str):
    def objective(trial):
        cfg = space(trial, name)

        def report(ep, val):
            trial.report(val, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        _, bundle = train_one(wl, name, cfg, max_epochs=max_epochs, patience=6, report=report)
        return bundle["best_val"]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=6))
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    best = space(optuna.trial.FixedTrial(study.best_params), name)
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception:  # needs >= 2 completed trials with varied params
        importance = {}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump({"model": name, "best_params": study.best_params, "best_val": study.best_value,
                   "final_cfg": best, "tune_seconds": time.time() - t0, "importance": importance,
                   "trials": [{"number": t.number, "state": t.state.name, "value": t.value, "params": t.params,
                               "curve": [t.intermediate_values[k] for k in sorted(t.intermediate_values)]}
                              for t in study.trials]}, f, indent=1, default=float)
    print(f"[{name}] best val {study.best_value:.4f} {study.best_params}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hr")
    ap.add_argument("--models", default="mlp,treelstm,mscn")
    ap.add_argument("--trials", default="40,24,20")
    ap.add_argument("--max-epochs", type=int, default=40)
    args = ap.parse_args()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    wl = load_workload(args.dataset)
    out = os.path.join(wl.engine.ds.models_dir, "tuning")
    for name, n in zip(args.models.split(","), map(int, args.trials.split(","))):
        tune(wl, name, n, args.max_epochs, out)


if __name__ == "__main__":
    main()
