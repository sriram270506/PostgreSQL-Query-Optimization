"""Full evaluation of one dataset -> <models_dir>/benchmark.json

    python -m scripts.benchmark --dataset hr
    python -m scripts.benchmark --dataset stats      # also STATS-CEB + data drift

Sections
  accuracy     Q-error summary per estimator on the i.i.d. test split (and STATS-CEB),
               with bootstrap 95% CIs; PostgreSQL measured on a real server, with and
               without CREATE STATISTICS
  seeds        per-seed spread of each learned model (mean +- std over seeds)
  significance paired Wilcoxon signed-rank tests on per-query log Q-error
  ood          template hold-out and shape hold-out models vs the baselines
  drift        (stats) models built on the 2013 snapshot, served on today's data
  plans        DP join order from each estimator's sub-plan estimates, scored with true C_out
  latency      single-query estimate latency (guarded) vs exact execution
  properties   order invariance, monotonicity under an added filter, bound violations
  guards       how often the uncertainty fallback and the upper-bound clamp fire
"""
import argparse
import json
import os
import random
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.engine import Join, Pred, Query, optimizer
from src.engine.live_pg import LivePG, dsn_for
from src.estimator.service import LABELS, Estimators, qerror

PG_DSN = os.environ.get("PG_DSN", "postgresql://postgres:postgres@127.0.0.1:55432/carddb")
CORRELATED = {"hr": {"employees.city", "employees.dept_id", "employees.age", "employees.salary",
                     "employees.experience_years", "projects.budget_tier", "projects.dept_id",
                     "departments.dept_name", "departments.region", "locations.city", "locations.office_tier"}}


def summary(q: np.ndarray) -> dict:
    if len(q) == 0:
        return {}
    return {"n": int(len(q)), "median": float(np.median(q)), "mean": float(np.mean(q)),
            "p90": float(np.percentile(q, 90)), "p95": float(np.percentile(q, 95)),
            "p99": float(np.percentile(q, 99)), "max": float(q.max()),
            "within_2x": float((q <= 2).mean()), "within_10x": float((q <= 10).mean())}


def bootstrap_ci(q: np.ndarray, B: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(q), size=(B, len(q)))
    s = np.sort(q[idx], axis=1)
    out = {}
    for name, p in (("median", 50), ("p95", 95), ("p99", 99)):
        vals = np.percentile(s, p, axis=1)
        out[name] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return out


class Bench:
    def __init__(self, ds_name: str, use_pg: bool = True):
        self.ds_name = ds_name
        self.S = Estimators(ds_name)
        self.names = self.S.available()
        self.pg = None
        if use_pg:
            try:
                self.pg = LivePG(dsn_for(PG_DSN, self.S.ds))
            except Exception as e:
                print("live PostgreSQL unavailable:", str(e).splitlines()[0])
        self.true_cache: Dict[str, int] = {}

    def true(self, q: Query) -> int:
        k = q.key() + "|" + ";".join(sorted(j.to_sql() for j in q.joins))
        if k not in self.true_cache:
            self.true_cache[k] = self.S.engine.true_count(q)
        return self.true_cache[k]

    # ── estimates of every estimator for a list of queries ───────────────────
    def estimates(self, qs: List[Query], S: Estimators = None, live: bool = True) -> Dict[str, dict]:
        S = S or self.S
        out = {n: S.estimate_many(qs, n) for n in S.available()}
        if live and self.pg:
            out["postgres_live"] = {"est": np.array([max(self.pg.explain_rows(q.to_sql()), 1) for q in qs])}
            self.pg.create_extended_stats(S.ds)
            out["postgres_xstats"] = {"est": np.array([max(self.pg.explain_rows(q.to_sql()), 1) for q in qs])}
            self.pg.drop_extended_stats(S.ds)
        return out

    def accuracy(self, qs, true, est: Dict[str, dict], extra_masks: Dict[str, np.ndarray] = None) -> dict:
        res = {}
        for n, r in est.items():
            qe = qerror(r["est"], true)
            res[n] = {"overall": summary(qe), "ci95": bootstrap_ci(qe),
                      "under": float((np.maximum(r["est"], 1) < np.maximum(true, 1)).mean())}
            if "fallback" in r:
                res[n]["guards"] = {"fallback_rate": float(r["fallback"].mean()),
                                    "clamp_rate": float(r["clamped"].mean())}
            for k, m in (extra_masks or {}).items():
                res[n][k] = summary(qe[m])
        return res

    def seeds(self, qs, true) -> dict:
        out = {}
        for n, members in self.S.ensembles.items():
            per = []
            for m, b in members:
                one = Estimators.__new__(Estimators)
                one.__dict__.update(self.S.__dict__)
                one.ensembles = {n: [(m, b)]}
                r = one.estimate_many(qs, n, guard=False)
                per.append(summary(qerror(r["est"], true)))
            out[n] = {k: {"mean": float(np.mean([p[k] for p in per])), "std": float(np.std([p[k] for p in per]))}
                      for k in ("median", "p95", "p99", "max", "within_2x")}
            out[n]["n_seeds"] = len(per)
        return out

    @staticmethod
    def significance(true, est: Dict[str, dict]) -> List[dict]:
        lq = {n: np.abs(np.log(np.maximum(r["est"], 1)) - np.log(np.maximum(true, 1))) for n, r in est.items()}
        pairs = [("treelstm", "factorjoin"), ("mlp", "factorjoin"), ("treelstm", "mlp"), ("treelstm", "mscn"),
                 ("mlp", "mscn"), ("factorjoin", "postgres_live"), ("factorjoin", "postgres_xstats")]
        out = []
        for a, b in pairs:
            if a in lq and b in lq:
                d = lq[a] - lq[b]
                p = float(wilcoxon(lq[a], lq[b]).pvalue) if np.any(d != 0) else 1.0
                out.append({"a": a, "b": b, "mean_log_q_a": float(lq[a].mean()), "mean_log_q_b": float(lq[b].mean()),
                            "a_better_frac": float((d < 0).mean()), "b_better_frac": float((d > 0).mean()),
                            "p_value": p})
        return out

    def plans(self, qs: List[Query], max_rels: int = 6) -> dict:
        multi = [q for q in qs if 3 <= len(q.rels) <= max_rels]
        subq, owner = [], []
        for i, q in enumerate(multi):
            for s in optimizer.connected_subsets(q):
                subq.append(q.sub(s))
                owner.append((i, s))
        t0 = time.time()
        true_sub = np.array([self.true(q) for q in subq], float)
        print(f"  plans: {len(multi)} queries, {len(subq)} sub-plans ({time.time() - t0:.0f}s true counts)", flush=True)
        est_sub = {n: self.S.estimate_many(subq, n)["est"] for n in self.names}
        if self.pg:
            est_sub["postgres_live"] = np.array([max(self.pg.explain_rows(q.to_sql()), 1) for q in subq])
        by_q: Dict[int, List[int]] = {}
        for k, (i, _) in enumerate(owner):
            by_q.setdefault(i, []).append(k)
        ratios = {n: [] for n in est_sub}
        for i, ks in by_q.items():
            tcard = {owner[k][1]: true_sub[k] for k in ks}
            opt = optimizer.best_plan(multi[i], tcard)[0]
            for n in est_sub:
                plan = optimizer.best_plan(multi[i], {owner[k][1]: est_sub[n][k] for k in ks})[1]
                ratios[n].append(optimizer.plan_true_cost(plan, tcard) / max(opt, 1.0))
        return {n: {"n": len(v), "optimal_rate": float(np.mean(np.isclose(v, 1.0))),
                    "median_ratio": float(np.median(v)), "mean_ratio": float(np.mean(v)),
                    "p95_ratio": float(np.percentile(v, 95)), "max_ratio": float(np.max(v))}
                for n, v in ratios.items()}

    def latency(self, qs: List[Query]) -> dict:
        lat = {n: [] for n in self.names}
        ex = []
        for q in qs:
            for n in self.names:
                lat[n].append(self.S.estimate(q, n)[1])
            t0 = time.perf_counter()
            self.S.engine.true_count(q)
            ex.append((time.perf_counter() - t0) * 1000)
        if self.pg:
            lat["postgres_live"] = []
            for q in qs:
                t0 = time.perf_counter()
                self.pg.explain_rows(q.to_sql())
                lat["postgres_live"].append((time.perf_counter() - t0) * 1000)
        out = {n: {"p50_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95)),
                   "p99_ms": float(np.percentile(v, 99)), "mean_ms": float(np.mean(v))} for n, v in lat.items()}
        out["exact_execution"] = {"p50_ms": float(np.median(ex)), "p95_ms": float(np.percentile(ex, 95)),
                                  "p99_ms": float(np.percentile(ex, 99)), "mean_ms": float(np.mean(ex))}
        return out

    def properties(self, qs: List[Query], seed: int = 0) -> dict:
        rng = random.Random(seed)
        ds = self.S.ds
        shuffled = []
        for q in qs:  # same query, written differently
            rels = list(q.rels.items())
            rng.shuffle(rels)
            joins = [Join(j.b, j.bcol, j.a, j.acol) for j in reversed(q.joins)]
            preds = list(q.preds)
            rng.shuffle(preds)
            shuffled.append(Query(rels=dict(rels), joins=joins, preds=preds))
        tighter = []
        for q in qs:  # one extra filter can only remove rows
            a = rng.choice(list(q.rels))
            t = q.rels[a]
            cols = [c for c in ds.filters.get(t, []) if self.S.engine.catalog.tables[t].numeric[c]]
            if not cols:
                tighter.append(None)
                continue
            c = rng.choice(cols)
            v = self.S.engine.catalog.tables[t].cols[c]
            med = float(np.nanmedian(v))
            tighter.append(Query(rels=q.rels, joins=q.joins,
                                 preds=q.preds + [Pred(a, c, "ge", int(med), c in ds.timestamps.get(t, []))]))
        keep = [i for i, x in enumerate(tighter) if x is not None]
        out = {}
        ub = np.array([self.S.bounds.upper(q) for q in qs])
        pg = (lambda xs: np.array([max(self.pg.explain_rows(x.to_sql()), 1.0) for x in xs])) if self.pg else None
        for n in self.names + (["postgres_live"] if pg else []):
            if n == "postgres_live":
                base = raw = pg(qs)
                perm, t_est = pg(shuffled), pg([tighter[i] for i in keep])
            else:
                base = self.S.estimate_many(qs, n)["est"]
                perm = self.S.estimate_many(shuffled, n)["est"]
                raw = self.S.estimate_many(qs, n, guard=False)["est"]
                t_est = self.S.estimate_many([tighter[i] for i in keep], n)["est"]
            mono = t_est > base[keep] * (1 + 1e-6)
            out[n] = {"order_invariant_rate": float(np.mean(np.isclose(base, perm, rtol=1e-6))),
                      "monotone_violation_rate": float(mono.mean()),
                      "monotone_violation_median_ratio": float(np.median((t_est / base[keep])[mono])) if mono.any() else 1.0,
                      "raw_bound_violation_rate": float(np.mean(raw > ub))}
        return out


def run(ds_name: str, quick: bool = False) -> dict:
    B = Bench(ds_name)
    S = B.S
    df = pd.read_csv(S.ds.workload)
    out = {"dataset": ds_name, "estimators": B.names + (["postgres_live", "postgres_xstats"] if B.pg else []),
           "labels": LABELS,
           "n_train": int((df.split == "train").sum()), "n_val": int((df.split == "val").sum()),
           "data_hash": S.fingerprint, "stale_models": S.stale}
    test = df[df.split == "test"].reset_index(drop=True)
    if quick:
        test = test.sample(n=min(300, len(test)), random_state=0).reset_index(drop=True)
    qs = [S.engine.parse(s) for s in test.sql]
    true = test.true_count.to_numpy(float)
    out["n_test"] = len(test)
    print(f"[{ds_name}] accuracy on {len(qs)} test queries", flush=True)
    est = B.estimates(qs)
    masks = {f"cat:{c}": (test.category == c).to_numpy() for c in sorted(test.category.unique())}
    if ds_name in CORRELATED:
        masks["correlated"] = np.array([sum(f"{q.rels[p.alias]}.{p.col}" in CORRELATED[ds_name] for p in q.preds) >= 2
                                        for q in qs])
    out["accuracy"] = B.accuracy(qs, true, est, masks)
    if not quick:  # same 300-query subsample CI re-runs, so the regression gate compares like with like
        sub = test.sample(n=min(300, len(test)), random_state=0).index.to_numpy()
        out["quick_accuracy"] = B.accuracy([qs[i] for i in sub], true[sub],
                                           {n: {k: v[sub] for k, v in r.items()} for n, r in est.items()})
    out["significance"] = B.significance(true, est)
    out["seeds"] = B.seeds(qs, true)
    out["scatter"] = {"true": true.tolist(), "category": test.category.tolist(),
                      **{n: r["est"].tolist() for n, r in est.items()}}
    for n, r in out["accuracy"].items():
        print(f"  {LABELS.get(n, n):32s} median {r['overall']['median']:.2f} p95 {r['overall']['p95']:.1f} "
              f"p99 {r['overall']['p99']:.1f} max {r['overall']['max']:.0f}", flush=True)

    if ds_name.startswith("stats"):
        ceb = [l.strip().split("||", 1) for l in open(os.path.join(S.ds.csv_dir, "stats_CEB.sql"))]
        cq = [S.engine.parse(s) for _, s in ceb]
        ct = np.array([int(c) for c, _ in ceb], float)
        print("[stats] STATS-CEB", flush=True)
        cest = B.estimates(cq)
        out["ceb"] = {"accuracy": B.accuracy(cq, ct, cest), "significance": B.significance(ct, cest),
                      "scatter": {"true": ct.tolist(), **{n: r["est"].tolist() for n, r in cest.items()}}}
        if not quick:
            out["ceb"]["plans"] = B.plans(cq, max_rels=7)

    out["ood"] = {}
    for split in ("split_tmpl", "split_shape"):
        So = Estimators(ds_name, split=split)
        if not So.ensembles:
            continue
        t = df[df[split] == "test"].reset_index(drop=True)
        if quick:
            t = t.sample(n=min(300, len(t)), random_state=0).reset_index(drop=True)
        oq = [S.engine.parse(s) for s in t.sql]
        print(f"[{ds_name}] OOD {split}: {len(oq)} queries", flush=True)
        oe = B.estimates(oq, So)
        out["ood"][split] = {"n": len(oq), "accuracy": B.accuracy(oq, t.true_count.to_numpy(float), oe)}

    if ds_name == "stats" and os.path.isdir("models/stats_old/split"):
        print("[stats] drift", flush=True)
        variants = {"stale (BN + NN from 2013)": Estimators("stats_old"),
                    "BN refreshed, NN stale": Estimators("stats", models_dir="models/stats_old"),
                    "fully retrained": S}
        out["drift"] = {}
        for label, Sv in variants.items():
            ve = {n: Sv.estimate_many(qs, n) for n in Sv.available()}
            if Sv is S and "postgres_live" in est:  # PostgreSQL re-ANALYZEs on its own: reference row
                ve["postgres_live"] = est["postgres_live"]
            out["drift"][label] = {n: summary(qerror(r["est"], true)) for n, r in ve.items()}

    if not quick:
        print(f"[{ds_name}] plans", flush=True)
        out["plans"] = B.plans(qs)
    sample = qs[:: max(1, len(qs) // 200)]
    print(f"[{ds_name}] latency + properties on {len(sample)} queries", flush=True)
    out["latency"] = B.latency(sample)
    out["properties"] = B.properties(sample)
    tun = {}
    for n in ("mlp", "treelstm", "mscn"):
        p = os.path.join(S.ds.models_dir, "tuning", f"{n}.json")
        if os.path.exists(p):
            t = json.load(open(p))
            tun[n] = {"best_params": t["best_params"], "best_val": t["best_val"], "n_trials": len(t["trials"]),
                      "tune_seconds": t.get("tune_seconds"), "importance": t.get("importance", {})}
    out["tuning"] = tun
    out["ensemble"] = S.meta
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hr")
    ap.add_argument("--quick", action="store_true", help="300-query subsample, no plan study (CI smoke run)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(args.dataset, args.quick)
    from src.engine import datasets
    path = args.out or os.path.join(datasets.get(args.dataset).models_dir, "benchmark.json")
    with open(path, "w") as f:
        json.dump(res, f)
    print("saved", path)


if __name__ == "__main__":
    main()
