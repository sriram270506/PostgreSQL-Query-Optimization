"""Tail-aware model selection on validation (after scripts.tune).

    python -m scripts.select_model --dataset hr

Optuna ranks configs by mean log Q-error, which happily trades a worse p99 for a better
median. Here the top-K tuned configs are crossed with loss variants (the tuned loss, and the
tail loss at two strengths), each trained with 2 seeds, and scored on validation by
mean + CVaR95 log Q-error (mean of the worst 5%). The winner is written to
<models_dir>/tuning/<name>.selected.json, which scripts.train then uses. The test split is
never looked at.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from src.estimator.neural import log_qerror, val_score
from src.estimator.training import DEFAULT_CFG, build_model, ensemble_log, load_workload, train_one

LOSSES = ({}, {"loss": "tail", "tail_lambda": 1.0}, {"loss": "tail", "tail_lambda": 3.0})


def candidates(mdir: str, name: str, k: int):
    p = os.path.join(mdir, "tuning", f"{name}.json")
    if not os.path.exists(p):
        return [dict(DEFAULT_CFG[name])]
    d = json.load(open(p))
    trials = sorted((t for t in d.get("trials", []) if t.get("state") == "COMPLETE"), key=lambda t: t["value"])
    base = {k_: v for k_, v in d["final_cfg"].items() if k_ not in d["best_params"]}
    cfgs = [d["final_cfg"]] + [{**base, **t["params"]} for t in trials]
    out = []
    for c in cfgs:
        if c not in out:
            out.append(c)
    return out[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hr")
    ap.add_argument("--models", default="mlp,treelstm,mscn")
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=80)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", min(8, os.cpu_count() or 1))))
    wl = load_workload(args.dataset)
    mdir = wl.engine.ds.models_dir
    v = wl.split("split")["val"]
    for name in args.models.split(","):
        rows = []
        for ci, cfg in enumerate(candidates(mdir, name, args.top_k)):
            for loss in LOSSES:
                c = {**cfg, **loss}  # early stopping stays on mean log-q (tail-based stopping stopped too early)
                t0 = time.time()
                members = []
                for seed in range(args.seeds):
                    _, b = train_one(wl, name, c, seed=seed, max_epochs=args.max_epochs, patience=10)
                    members.append((build_model(b), b))
                mean, _, _ = ensemble_log(members, [wl.trees[i] for i in v], wl.stats[v],
                                          [wl.mscn_feats[i] for i in v])
                e = log_qerror(mean, wl.y[v])
                k = max(1, int(np.ceil(0.05 * len(e))))
                rows.append({"cfg": c, "candidate": ci, "loss": loss.get("loss", cfg.get("loss")),
                             "tail_lambda": loss.get("tail_lambda"), "score": val_score(e, {"select": "tail"}),
                             "val_mean_log_q": float(e.mean()), "val_cvar95": float(np.sort(e)[-k:].mean()),
                             "val_p99_q": float(np.exp(np.quantile(e, 0.99))), "seconds": time.time() - t0})
                r = rows[-1]
                print(f"[{args.dataset}] {name} cand {ci} loss {r['loss']}/{r['tail_lambda']}: score {r['score']:.4f} "
                      f"mean {r['val_mean_log_q']:.4f} cvar95 {r['val_cvar95']:.3f} p99 {r['val_p99_q']:.2f}x "
                      f"({r['seconds']:.0f}s)", flush=True)
        best = min(rows, key=lambda r: r["score"])
        out = os.path.join(mdir, "tuning", f"{name}.selected.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"model": name, "final_cfg": best["cfg"], "criterion": "val mean + CVaR95 log Q-error",
                       "seeds": args.seeds, "candidates": rows}, f, indent=1)
        print(f"[{args.dataset}] {name}: selected cand {best['candidate']} loss {best['loss']}/{best['tail_lambda']}",
              flush=True)


if __name__ == "__main__":
    main()
