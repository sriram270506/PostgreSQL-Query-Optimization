"""Head-to-head benchmark on the held-out test split -> models/benchmark.json

    python -m scripts.benchmark

1. Accuracy: Q-error percentiles overall, per join size, and on the "correlated" slice.
2. Latency: per-query estimation time (features included) vs exact execution time.
3. Plan quality: for every 3+ table query, the DP join-order optimizer is run with each
   estimator's sub-join cardinalities; the chosen plan is scored with TRUE intermediate
   sizes (C_out) and compared with the optimal plan.
"""
import json
import time

import numpy as np
import pandas as pd

from src.engine import optimizer
from src.estimator.service import LABELS, Estimators, qerror

CORRELATED = {"employees.city", "employees.dept_id", "employees.age", "employees.salary",
              "employees.experience_years", "projects.budget_tier", "projects.dept_id",
              "departments.dept_name", "departments.region", "locations.city", "locations.office_tier"}


def summary(q: np.ndarray) -> dict:
    if len(q) == 0:
        return {}
    return {"n": int(len(q)), "median": float(np.median(q)), "mean": float(q.mean()),
            "p90": float(np.percentile(q, 90)), "p95": float(np.percentile(q, 95)),
            "p99": float(np.percentile(q, 99)), "max": float(q.max()),
            "within_2x": float((q <= 2).mean()), "within_10x": float((q <= 10).mean())}


def main():
    S = Estimators()
    names = S.available()
    df = pd.read_csv("data/workload/workload.csv")
    test = df[df.split == "test"].reset_index(drop=True)
    qs = [S.engine.parse(s) for s in test.sql]
    true = test.true_count.to_numpy(float)
    corr = np.array([sum(f"{q.rels[p.alias]}.{p.col}" in CORRELATED for p in q.preds) >= 2 for q in qs])

    est = {n: S.estimate_many(qs, n) for n in names}
    acc = {}
    for n in names:
        qe = qerror(est[n], true)
        acc[n] = {"overall": summary(qe), "correlated": summary(qe[corr]),
                  "by_category": {c: summary(qe[(test.category == c).to_numpy()])
                                  for c in sorted(test.category.unique())},
                  "under": float((np.maximum(est[n], 1) < np.maximum(true, 1)).mean())}
        print(f"{LABELS[n]:20s} median {acc[n]['overall']['median']:.3f}  p95 {acc[n]['overall']['p95']:.2f}  "
              f"p99 {acc[n]['overall']['p99']:.2f}  max {acc[n]['overall']['max']:.0f}")

    # latency: single-query calls, as the optimizer would issue them
    lat = {n: [] for n in names}
    exec_ms = []
    sample = list(range(0, len(qs), max(1, len(qs) // 300)))
    for i in sample:
        for n in names:
            lat[n].append(S.estimate(qs[i], n)[1])
        t0 = time.perf_counter()
        S.engine.true_count(qs[i])
        exec_ms.append((time.perf_counter() - t0) * 1000)
    latency = {n: {"mean_ms": float(np.mean(v)), "p50_ms": float(np.median(v)),
                   "p95_ms": float(np.percentile(v, 95))} for n, v in lat.items()}
    latency["exact_execution"] = {"mean_ms": float(np.mean(exec_ms)), "p50_ms": float(np.median(exec_ms)),
                                  "p95_ms": float(np.percentile(exec_ms, 95))}

    # plan quality
    multi = [i for i, q in enumerate(qs) if len(q.rels) >= 3]
    subsets_all, owners = [], []
    for i in multi:
        for s in optimizer.connected_subsets(qs[i]):
            subsets_all.append(qs[i].sub(s))
            owners.append((i, s))
    true_sub = np.array([S.engine.true_count(q) for q in subsets_all], float)
    est_sub = {n: S.estimate_many(subsets_all, n) for n in names}
    ratios = {n: [] for n in names}
    for i in multi:
        idx = [k for k, (o, _) in enumerate(owners) if o == i]
        tcard = {owners[k][1]: true_sub[k] for k in idx}
        opt_cost = optimizer.best_plan(qs[i], tcard)[0]
        for n in names:
            ecard = {owners[k][1]: est_sub[n][k] for k in idx}
            plan = optimizer.best_plan(qs[i], ecard)[1]
            ratios[n].append(optimizer.plan_true_cost(plan, tcard) / max(opt_cost, 1.0))
    plans = {n: {"n": len(v), "optimal_rate": float(np.mean(np.isclose(v, 1.0))),
                 "median_ratio": float(np.median(v)), "mean_ratio": float(np.mean(v)),
                 "p95_ratio": float(np.percentile(v, 95)), "max_ratio": float(np.max(v))} for n, v in ratios.items()}
    for n in names:
        print(f"{LABELS[n]:20s} optimal plans {plans[n]['optimal_rate']:.1%}  mean C_out ratio {plans[n]['mean_ratio']:.3f}  "
              f"max {plans[n]['max_ratio']:.2f}")

    examples = []
    for i in np.argsort(-qerror(est.get("postgres"), true))[:12]:
        examples.append({"sql": test.sql[i], "category": test.category[i], "true": int(true[i]),
                         **{n: float(est[n][i]) for n in names}})

    out = {"n_test": len(test), "n_train": int((df.split == "train").sum()), "n_val": int((df.split == "val").sum()),
           "estimators": names, "labels": {n: LABELS[n] for n in names},
           "accuracy": acc, "latency": latency, "plans": plans, "worst_for_postgres": examples,
           "scatter": {"true": true.tolist(), "category": test.category.tolist(),
                       **{n: est[n].tolist() for n in names}}}
    tun = {}
    for n in ("mlp", "treelstm"):
        try:
            with open(f"models/tuning/{n}.json") as f:
                t = json.load(f)
            tun[n] = {k: t[k] for k in ("best_params", "best_val", "n_params", "tune_seconds", "train_seconds")}
            tun[n]["n_trials"] = len(t["trials"])
        except FileNotFoundError:
            pass
    out["tuning"] = tun
    with open("models/benchmark.json", "w") as f:
        json.dump(out, f)
    print("saved models/benchmark.json")


if __name__ == "__main__":
    main()
