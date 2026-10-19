"""Generate a labelled workload with the embedded engine (no PostgreSQL needed).

    python -m scripts.build_workload --dataset hr --n 8000
    python -m scripts.build_workload --dataset stats --n 8000

Columns: sql, category, n_tables, template, true_count, pg_estimate, fj_estimate and three
split schemes, each train/val/test:
  split        i.i.d. 70/15/15, stratified by join size
  split_tmpl   template hold-out: 20% of query templates (relations + filtered columns)
               never appear in train/val
  split_shape  shape hold-out: train/val on 1-3 table queries, test on 4-5 table queries
"""
import argparse
import os
import random
import time

import numpy as np
import pandas as pd

from src.engine import Engine
from src.estimator.factorjoin import FactorJoin
from src.workload.generator import WorkloadGenerator, category, template


def add_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df["split"] = "train"
    for _, g in df.groupby("category"):
        idx = g.sample(frac=1, random_state=seed).index
        n = len(idx)
        df.loc[idx[int(0.70 * n):int(0.85 * n)], "split"] = "val"
        df.loc[idx[int(0.85 * n):], "split"] = "test"
    tmpls = df.template.unique()
    held = set(rng.choice(tmpls, size=max(1, int(0.2 * len(tmpls))), replace=False))
    df["split_tmpl"] = np.where(df.template.isin(held), "test",
                                np.where(rng.random(len(df)) < 0.15, "val", "train"))
    df["split_shape"] = np.where(df.n_tables >= 4, "test",
                                 np.where(rng.random(len(df)) < 0.15, "val", "train"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hr")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    eng = Engine(args.dataset)
    out = args.out or eng.ds.workload
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
                     "template": template(q), "true_count": true, "pg_estimate": eng.planner.estimate(q),
                     "fj_estimate": fj.estimate(q)})
        if len(rows) % 1000 == 0:
            print(f"{len(rows)} queries ({time.time() - t0:.0f}s)", flush=True)

    df = add_splits(pd.DataFrame(rows), args.seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    for s in ("split", "split_tmpl", "split_shape"):
        print(s, df[s].value_counts().to_dict())
    print("templates:", df.template.nunique(), "saved", out)


if __name__ == "__main__":
    main()
