"""FastAPI backend: embedded SQL engine + live cardinality estimation shoot-out.

    python server.py           # http://127.0.0.1:8000  (frontend dev server proxies /api)

No PostgreSQL needed: queries execute on the in-process engine (src/engine) and the
"PostgreSQL" estimate comes from its planner emulation. If PG_DSN is set and psycopg2
is installed, /api/run additionally reports a live EXPLAIN estimate from a real server.
"""
import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.engine import SQLError, optimizer
from src.engine.executor import JOIN_EDGES
from src.estimator.service import LABELS, Estimators, qerror
from src.estimator import viz

MODELS_DIR = "models"
WORKLOAD = "data/workload/workload.csv"
MAX_PLAN_RELS = 6

app = FastAPI(title="Cardinality Lab")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                                                  "http://localhost:5174", "http://127.0.0.1:5174"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])

S = Estimators(models_dir=MODELS_DIR)
_workload = pd.read_csv(WORKLOAD) if os.path.exists(WORKLOAD) else None
_viz_cache: Dict[str, dict] = {}


def _json_file(name: str) -> dict:
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(404, f"{name} not found - run the training/benchmark scripts first")
    with open(path) as f:
        return json.load(f)


def _py(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else float(v)
    return v


# ── schema / data ───────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "estimators": S.available(),
            "tables": {t: tb.n for t, tb in S.engine.catalog.tables.items()},
            "workload": 0 if _workload is None else len(_workload),
            "postgres_live": bool(os.environ.get("PG_DSN"))}


@app.get("/api/schema")
def schema():
    out = []
    for t, tb in S.engine.catalog.tables.items():
        cols = []
        for c in tb.cols:
            v = tb.df[c]
            info = {"name": c, "type": "number" if tb.numeric[c] else "text", "distinct": int(v.nunique())}
            if tb.numeric[c]:
                info.update(min=_py(v.min()), max=_py(v.max()))
            elif info["distinct"] <= 12:
                info["values"] = sorted(map(str, v.unique()))
            cols.append(info)
        out.append({"name": t, "rows": tb.n, "columns": cols})
    return {"tables": out, "joins": [{"left": f"{a}.{b}", "right": f"{c}.{d}"} for a, b, c, d in JOIN_EDGES]}


@app.get("/api/tables/{name}")
def table_rows(name: str, page: int = 1, page_size: int = 25):
    tb = S.engine.catalog.tables.get(name)
    if tb is None:
        raise HTTPException(404, f"unknown table {name}")
    page_size = max(1, min(page_size, 200))
    start = (max(page, 1) - 1) * page_size
    chunk = tb.df.iloc[start:start + page_size]
    return {"columns": list(tb.df.columns), "rows": chunk.astype(object).values.tolist(),
            "total": tb.n, "page": page, "page_size": page_size}


@app.get("/api/bn/{table}")
def bayes_net(table: str):
    bn = S.fj.bns.get(table)
    if bn is None:
        raise HTTPException(404, f"unknown table {table}")
    nodes = [{"id": c, "bins": bn.disc[c].labels(), "marginal": np.round(bn.marg[c], 4).tolist()} for c in bn.cols]
    edges = [{"parent": p, "child": c, "mi": round(mi, 4),
              "cpt": np.round(bn.cpt(c, p), 4).tolist()} for p, c, mi in bn.edges]
    return {"table": table, "rows": bn.n, "nodes": nodes, "edges": edges}


@app.get("/api/workload/sample")
def workload_sample(category: Optional[str] = None, n: int = 6, seed: Optional[int] = None):
    if _workload is None:
        raise HTTPException(404, "workload not generated - run scripts.build_workload")
    df = _workload[_workload.split == "test"]
    if category:
        df = df[df.category == category]
    df = df.sample(n=min(n, len(df)), random_state=seed)
    return [{"sql": r.sql, "category": r.category, "true_count": int(r.true_count)} for r in df.itertuples()]


# ── the shoot-out ───────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    sql: str = Field(..., max_length=5000)
    row_limit: int = Field(100, ge=1, le=1000)


def _postgres_live(sql: str) -> Optional[dict]:
    dsn = os.environ.get("PG_DSN")
    if not dsn:
        return None
    try:
        import psycopg2
        with psycopg2.connect(dsn, connect_timeout=3) as conn, conn.cursor() as cur:
            t0 = time.perf_counter()
            cur.execute("EXPLAIN (FORMAT JSON) " + sql)  # sql is the engine's normalised SELECT
            plan = cur.fetchone()[0]
            ms = (time.perf_counter() - t0) * 1000
        return {"estimate": float(plan[0]["Plan"]["Plan Rows"]), "latency_ms": ms}
    except Exception as e:  # live PG is optional; report, never fail the request
        return {"error": str(e).splitlines()[0]}


def _plans(q, names: List[str]) -> Optional[dict]:
    if len(q.rels) < 3 or len(q.rels) > MAX_PLAN_RELS:
        return None
    subsets = optimizer.connected_subsets(q)
    subq = [q.sub(s) for s in subsets]
    true = {s: float(S.engine.true_count(sq)) for s, sq in zip(subsets, subq)}
    opt_cost, opt_plan = optimizer.best_plan(q, true)
    out = {"optimal_cost": opt_cost, "optimal": optimizer.annotate(opt_plan, true, true), "estimators": {}}
    for n in names:
        ests = dict(zip(subsets, S.estimate_many(subq, n)))
        _, plan = optimizer.best_plan(q, ests)
        cost = optimizer.plan_true_cost(plan, true)
        out["estimators"][n] = {"true_cost": cost, "ratio": cost / max(opt_cost, 1.0),
                                "plan": optimizer.annotate(plan, ests, true)}
    return out


@app.post("/api/run")
def run(req: RunRequest):
    try:
        res = S.engine.run(req.sql, req.row_limit)
    except SQLError as e:
        raise HTTPException(400, str(e))
    q = res["query"]
    count_q = q.sub(q.rels)
    names = S.available()
    ests = []
    for n in names:
        est, ms = S.estimate(count_q, n)
        ests.append({"name": n, "label": LABELS[n], "estimate": est, "latency_ms": ms,
                     "qerror": float(qerror(est, res["count"])), "under": est < max(res["count"], 1)})
    live = _postgres_live(count_q.to_sql())
    if live and "estimate" in live:
        ests.append({"name": "postgres_live", "label": "PostgreSQL (live server)", "estimate": live["estimate"],
                     "latency_ms": live["latency_ms"], "qerror": float(qerror(live["estimate"], res["count"])),
                     "under": live["estimate"] < max(res["count"], 1)})
    rows = res["rows"]
    scan = []
    for a, s in res["scan"].items():
        scan.append({**s, "alias": a, "factorjoin": S.fj.table_estimate(count_q, a),
                     "preds": [p.to_sql() for p in q.preds_of(a)]})
    return {
        "sql": q.to_sql(), "count": res["count"],
        "exec_ms": res["count_ms"] + res["fetch_ms"], "count_ms": res["count_ms"], "fetch_ms": res["fetch_ms"],
        "rows": None if rows is None else {"columns": list(rows.columns),
                                           "data": rows.astype(object).where(rows.notna(), None).values.tolist(),
                                           "shown": len(rows), "truncated": len(rows) < res["count"]},
        "estimators": ests,
        "best": min(ests, key=lambda e: e["qerror"])["name"],
        "scan": scan,
        "plans": _plans(count_q, names),
        "postgres_live_error": live.get("error") if live else None,
    }


# ── results / model internals ───────────────────────────────────────────────

@app.get("/api/benchmark")
def benchmark():
    return _json_file("benchmark.json")


@app.get("/api/tuning/{name}")
def tuning(name: str):
    if name not in ("mlp", "treelstm"):
        raise HTTPException(404, "unknown model")
    return _json_file(f"tuning/{name}.json")


def _viz_global(name: str) -> dict:
    if name not in S.models:
        raise HTTPException(404, f"model {name} not trained")
    if name not in _viz_cache:
        path = os.path.join(MODELS_DIR, f"viz_{name}.json")
        model_path = os.path.join(MODELS_DIR, f"fusion_{name}.pt")
        if os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(model_path):
            with open(path) as f:
                _viz_cache[name] = json.load(f)
        else:
            df = _workload[_workload.split == "test"].sample(n=800, random_state=0)
            data = viz.global_analysis(S, [S.engine.parse(s) for s in df.sql], name)
            with open(path, "w") as f:
                json.dump(data, f)
            _viz_cache[name] = data
    return _viz_cache[name]


@app.get("/api/viz/global")
def viz_global(model: str = "treelstm"):
    return _viz_global(model)


class TraceRequest(BaseModel):
    sql: str = Field(..., max_length=5000)
    model: str = "treelstm"


@app.post("/api/viz/trace")
def viz_trace(req: TraceRequest):
    g = _viz_global(req.model)
    try:
        q = S.engine.parse(req.sql)
    except SQLError as e:
        raise HTTPException(400, str(e))
    out = viz.trace_query(S, q, g.get("probes", {}), req.model)
    out["true_count"] = S.engine.true_count(q.sub(q.rels))
    return out


DIST = os.path.join("frontend", "dist")
if os.path.isdir(DIST):  # serve the built UI from the same origin
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
