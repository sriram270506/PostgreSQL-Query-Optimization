"""Generate a labelled workload with the embedded engine (no PostgreSQL needed).

    python -m scripts.build_workload --n 8000

Writes data/workload/workload.csv: sql, category, n_tables, true_count, pg_estimate,
fj_estimate, split (train/val/test = 70/15/15, stratified by category).
"""
import argparse
import os
import random
import time

import pandas as pd

from src.engine import Engine
from src.estimator.factorjoin import FactorJoin
from src.workload.generator import WorkloadGenerator, category


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/workload/workload.csv")
    args = ap.parse_args()

    eng = Engine()
    fj = FactorJoin(eng.catalog)
    gen = WorkloadGenerator(eng.catalog, args.seed)
    rng = random.Random(args.seed)
    seen, rows = set(), []
    t0 = time.time()
    while len(rows) < args.n:
        q = gen.query()
        sql = q.to_sql()
        if sql in seen:
            continue
        true = eng.true_count(q)
        if true == 0 and rng.random() < 0.9:  # keep a few empty results, like real workloads
            continue
        seen.add(sql)
        rows.append({"sql": sql, "category": category(len(q.rels)), "n_tables": len(q.rels),
                     "true_count": true, "pg_estimate": eng.planner.estimate(q),
                     "fj_estimate": fj.estimate(q)})
        if len(rows) % 1000 == 0:
            print(f"{len(rows)} queries ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df["split"] = "train"
    for _, g in df.groupby("category"):
        idx = g.sample(frac=1, random_state=args.seed).index
        n = len(idx)
        df.loc[idx[int(0.70 * n):int(0.85 * n)], "split"] = "val"
        df.loc[idx[int(0.85 * n):], "split"] = "test"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby(["category", "split"]).size().unstack())
    print("saved", args.out)


if __name__ == "__main__":
    main()
