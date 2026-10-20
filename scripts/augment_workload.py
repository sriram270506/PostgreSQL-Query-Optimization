"""Extend a workload with targeted examples and two more hold-out split schemes.

    python -m scripts.augment_workload --dataset hr
    python -m scripts.augment_workload --dataset stats

Adds rows (column `aug`):
  tiny       training-only queries (kept to ~5% of training rows) whose true result is empty or tiny (0..10 rows)
  rare       training-only queries filtering on the rarest or absent literal values
  eval_tiny  / eval_rare   the same kinds, never trained on: split = "special"
Adds split schemes (train / val / test):
  split_join  hold out 20% of join graphs (relation set + join edges)
  split_sel   selectivity regime: train/val on the middle of the cardinality range, test on
              both tails (bottom and top 7.5% of true counts), i.e. extrapolation
Base rows and their existing splits are unchanged; augmented training rows only ever enter
`train` of schemes where their own template / graph / shape / regime is not held out.
"""
import argparse
import random

import numpy as np
import pandas as pd

from src.engine import Engine, Pred, Query
from src.estimator.factorjoin import FactorJoin
from src.workload.generator import WorkloadGenerator, category, template

# ablation (HR TreeLSTM, STATS MLP): ~27% augmented rows hurt the main distribution
# (val mean log-q 0.129 -> 0.227); ~5% is neutral on val and halves rare-value error
N_TINY, N_RARE, N_EVAL = 210, 90, 300


def join_graph(q: Query) -> str:
    edges = sorted("=".join(sorted([f"{q.rels[j.a]}.{j.acol}", f"{q.rels[j.b]}.{j.bcol}"])) for j in q.joins)
    return "|".join(sorted(q.rels.values())) + "#" + ";".join(edges)


def rare_query(gen: WorkloadGenerator, eng: Engine, rng: random.Random) -> Query:
    """A generated query whose first filter is replaced by the rarest or an absent value."""
    q = gen.query()
    cands = [(a, t, c) for a, t in q.rels.items() for c in eng.ds.filters.get(t, [])]
    if not cands:
        return q
    a, t, c = rng.choice(cands)
    tb = eng.catalog.tables[t]
    v = tb.cols[c][np.asarray(pd.notna(tb.cols[c]))]
    uniq, cnt = np.unique(v.astype(str) if v.dtype == object else v, return_counts=True)
    if rng.random() < 0.5 and len(uniq):  # rarest existing value
        val = uniq[np.argmin(cnt)]
        val = str(val) if v.dtype == object else (int(val) if float(val).is_integer() else float(val))
    elif v.dtype == object:  # a category the column has never seen
        val = "__unseen_" + str(rng.randint(0, 999))
    else:  # a number absent from the column (just outside its range)
        val = int(np.nanmax(v.astype(float))) + rng.randint(1, 1000)
    preds = [p for p in q.preds if not (p.alias == a and p.col == c)]
    preds.append(Pred(a, c, "eq", val, c in eng.ds.timestamps.get(t, [])))
    return Query(rels=q.rels, joins=q.joins, preds=preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hr")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    eng = Engine(args.dataset)
    fj = FactorJoin(eng.catalog)
    df = pd.read_csv(eng.ds.workload)
    df = df[~df.get("aug", pd.Series("", index=df.index)).fillna("").astype(bool)].copy()  # idempotent
    df["aug"] = ""
    rng = random.Random(args.seed)
    gen = WorkloadGenerator(eng.catalog, args.seed)
    seen = set(df.sql)
    rows = []

    def add(q, kind):
        sql = q.to_sql()
        if sql in seen:
            return False
        try:
            true = eng.true_count(q)
        except Exception:
            return False
        if kind.endswith("tiny") and true > 10:
            return False
        seen.add(sql)
        rows.append({"sql": sql, "category": category(len(q.rels)), "n_tables": len(q.rels), "template": template(q),
                     "true_count": true, "pg_estimate": eng.planner.estimate(q), "fj_estimate": fj.estimate(q),
                     "aug": kind})
        return True

    targets = {"tiny": N_TINY, "rare": N_RARE, "eval_tiny": N_EVAL, "eval_rare": N_EVAL}
    for kind, n in targets.items():
        got, tries = 0, 0
        while got < n and tries < n * 400:
            tries += 1
            q = rare_query(gen, eng, rng) if kind.endswith("rare") else gen.query()
            got += add(q, kind)
        print(f"{kind}: {got} queries ({tries} tries)", flush=True)

    aug = pd.DataFrame(rows)
    full = pd.concat([df, aug], ignore_index=True)
    full["aug"] = full["aug"].fillna("")
    graphs = [join_graph(eng.parse(s)) for s in full.sql]
    full["join_graph"] = graphs
    srng = np.random.default_rng(args.seed)
    base = full.aug == ""
    held_graphs = set(srng.choice(sorted(set(full.join_graph[base])), size=max(1, int(0.2 * full.join_graph[base].nunique())),
                                  replace=False))
    lo, hi = np.quantile(full.true_count[base], [0.075, 0.925])
    r = srng.random(len(full))
    full["split_join"] = np.where(full.join_graph.isin(held_graphs), "test", np.where(r < 0.15, "val", "train"))
    regime_test = (full.true_count <= lo) | (full.true_count >= hi)
    full["split_sel"] = np.where(regime_test, "test", np.where(r < 0.15, "val", "train"))
    # augmented rows: training-only, and never inside a held-out part of any scheme
    special = full.aug.str.startswith("eval")
    train_aug = (full.aug != "") & ~special
    for sch in ("split", "split_tmpl", "split_shape", "split_join", "split_sel"):
        held = (full[sch] == "test") if sch != "split" else pd.Series(False, index=full.index)
        if sch == "split_tmpl":
            held = full.template.isin(set(full.template[base & (full[sch] == "test")]))
        if sch == "split_shape":
            held = full.n_tables >= 4
        full.loc[train_aug, sch] = np.where(held[train_aug], "none", "train")
        full.loc[special, sch] = "special"
    full.to_csv(eng.ds.workload, index=False)
    for sch in ("split", "split_tmpl", "split_shape", "split_join", "split_sel"):
        print(sch, full[sch].value_counts().to_dict())
    print("saved", eng.ds.workload, len(full), "rows")


if __name__ == "__main__":
    main()
