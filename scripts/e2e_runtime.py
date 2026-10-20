"""End-to-end query runtime on real PostgreSQL 16 (CardEst methodology, Han et al. VLDB 2022).

    python -m scripts.e2e_runtime --timeout 30

For every STATS-CEB query and every estimator, the estimator's cardinality for every
connected multi-table sub-plan is injected into PostgreSQL with pg_hint_plan Rows()
hints; PostgreSQL then plans and *executes* the query. Configurations:
  postgres         no hints (PostgreSQL's own estimates)
  postgres_xstats  no hints, after CREATE STATISTICS on every table's filter columns
  oracle           true sub-plan cardinalities (best this approach can do)
  factorjoin, mscn, mlp, treelstm   injected estimates (guarded ensembles)
Base-table estimates cannot be overridden by Rows(), so single-table selectivities stay
PostgreSQL's in every configuration; only join cardinalities differ. Queries hitting the
timeout are charged the timeout. Writes models/stats/e2e.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from src.engine import optimizer
from src.engine.live_pg import LivePG, dsn_for
from scripts.benchmark import provenance
from src.estimator.service import Estimators

from src.env import load_dotenv

load_dotenv()
PG_DSN = os.environ.get("PG_DSN")  # no default credentials: set it in .env


def estimation_cost(path: str):
    """Add the time each estimator needs to produce all sub-plan estimates of every query
    (one batched call per query, caches cold) and the resulting net runtime."""
    e = json.load(open(path))
    S = Estimators("stats")
    lines = [l.strip().split("||", 1) for l in open(os.path.join(S.ds.csv_dir, "stats_CEB.sql"))][: e["n_queries"]]
    subs = [[q.sub(x) for x in optimizer.connected_subsets(q)] for q in (S.engine.parse(sql) for _, sql in lines)]
    for c in e["summary"]:
        if c not in S.available():
            continue  # PostgreSQL configs: estimation is part of PostgreSQL itself; oracle: not obtainable
        S.estimate_many(subs[0], c)  # warm-up (model load), not counted
        ms = []
        for sq in subs:
            S.fj._cache.clear()
            S.fj._lcache.clear()
            t0 = time.perf_counter()
            S.estimate_many(sq, c)
            ms.append((time.perf_counter() - t0) * 1000)
        s = e["summary"][c]
        s["estimation_s"] = float(sum(ms) / 1000)
        s["estimation_median_ms"] = float(np.median(ms))
        s["net_total_s"] = s["total_s"] + s["estimation_s"]
        print(f"{c:12s} estimation {s['estimation_s']:.2f}s (median {s['estimation_median_ms']:.1f} ms/query), "
              f"net {s['net_total_s']:.1f}s vs PostgreSQL {e['summary']['postgres']['total_s']:.1f}s", flush=True)
    e["n_subplans"] = int(sum(len(x) for x in subs))
    json.dump(e, open(path, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--configs", default="postgres,postgres_xstats,oracle,factorjoin,mscn,mlp,treelstm")
    ap.add_argument("--limit", type=int, default=0, help="first N queries only (smoke test)")
    ap.add_argument("--out", default="models/stats/e2e.json")
    ap.add_argument("--estimation-cost", action="store_true",
                    help="only (re)measure estimation time per query and add net totals to --out")
    args = ap.parse_args()
    if args.estimation_cost:
        estimation_cost(args.out)
        return

    if not PG_DSN:
        raise SystemExit("PG_DSN is not set (see .env.example): the end-to-end test needs a real PostgreSQL")
    S = Estimators("stats")
    pg = LivePG(dsn_for(PG_DSN, S.ds))
    lines = [l.strip().split("||", 1) for l in open(os.path.join(S.ds.csv_dir, "stats_CEB.sql"))]
    if args.limit:
        lines = lines[: args.limit]
    queries = [(int(c), sql.rstrip(";"), S.engine.parse(sql)) for c, sql in lines]
    configs = args.configs.split(",")

    # sub-plan cardinalities per configuration (computed once, before any timing)
    hints = {c: [] for c in configs}
    t0 = time.time()
    for _, _, q in queries:
        subsets = optimizer.connected_subsets(q)
        subq = [q.sub(s) for s in subsets]
        keys = [tuple(sorted(s)) for s in subsets]
        for c in configs:
            if c in ("postgres", "postgres_xstats"):
                hints[c].append(None)
            elif c == "oracle":
                hints[c].append(dict(zip(keys, (float(S.engine.true_count(x)) for x in subq))))
            else:
                hints[c].append(dict(zip(keys, S.estimate_many(subq, c)["est"].tolist())))
    print(f"sub-plan estimates ready ({time.time() - t0:.0f}s)", flush=True)

    res = {c: {"ms": [], "timeouts": 0} for c in configs}
    for c in configs:
        if c == "postgres_xstats":
            pg.create_extended_stats(S.ds)
        t1 = time.time()
        for i, (_, sql, _) in enumerate(queries):
            ms, _ = pg.run_hinted(sql, hints[c][i], args.timeout)
            if ms is None:
                res[c]["timeouts"] += 1
                ms = args.timeout * 1000
            res[c]["ms"].append(ms)
        if c == "postgres_xstats":
            pg.drop_extended_stats(S.ds)
        tot = sum(res[c]["ms"]) / 1000
        print(f"{c:16s} total {tot:8.1f}s  timeouts {res[c]['timeouts']}  ({time.time() - t1:.0f}s wall)", flush=True)

    base = np.array(res["postgres"]["ms"]) if "postgres" in res else None
    summary = {}
    for c, r in res.items():
        ms = np.array(r["ms"])
        summary[c] = {"total_s": float(ms.sum() / 1000), "timeouts": r["timeouts"], "median_ms": float(np.median(ms)),
                      "p95_ms": float(np.percentile(ms, 95))}
        if base is not None:
            ratio = ms / np.maximum(base, 1e-3)
            summary[c].update(speedup_total=float(base.sum() / ms.sum()),
                              faster_queries=int((ratio < 0.9).sum()), slower_queries=int((ratio > 1.1).sum()))
    out = {"timeout_s": args.timeout, "n_queries": len(queries), "summary": summary,
           "per_query_ms": {c: r["ms"] for c, r in res.items()}, "true": [c for c, _, _ in queries],
           "provenance": provenance(S, pg, sys.argv)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print("saved", args.out)


if __name__ == "__main__":
    main()
