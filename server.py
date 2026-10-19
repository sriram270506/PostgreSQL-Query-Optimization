"""FastAPI backend: embedded SQL engine + live cardinality estimation shoot-out.

    python server.py           # http://127.0.0.1:8000  (frontend dev server proxies /api)

Every endpoint takes ?dataset=hr|stats (default hr). Queries execute on the in-process
engine; learned estimators are guarded ensembles (src/estimator/service.py). If PG_DSN is
set, /api/run also reports the live estimate of a real PostgreSQL server.
"""
import json
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query as Q
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.engine import SQLError, datasets, optimizer
from src.engine.live_pg import LivePG, dsn_for
from src.estimator import viz
from src.estimator.service import LABELS, Estimators, qerror

MAX_PLAN_RELS = 6
SERVED = [d for d in ("hr", "stats") if os.path.exists(os.path.join(datasets.get(d).csv_dir,
                                                                    next(iter(datasets.get(d).files.values()))))]

app = FastAPI(title="Cardinality Lab")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173",
                                                  "http://localhost:5174", "http://127.0.0.1:5174"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])

_services: Dict[str, Estimators] = {}
_workloads: Dict[str, Optional[pd.DataFrame]] = {}
_viz_cache: Dict[str, dict] = {}
_load_lock = threading.Lock()


def svc(dataset: str) -> Estimators:
    if dataset not in SERVED:
        raise HTTPException(404, f"unknown or unavailable dataset {dataset!r}; available: {SERVED}")
    with _load_lock:
        if dataset not in _services:
            _services[dataset] = Estimators(dataset)
            wl = datasets.get(dataset).workload
            _workloads[dataset] = pd.read_csv(wl) if os.path.exists(wl) else None
    return _services[dataset]


def _json_file(path: str) -> dict:
    if not os.path.exists(path):
        raise HTTPException(404, f"{path} not found - run the training/benchmark scripts first")
    with open(path) as f:
        return json.load(f)


def _py(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else float(v)
    return v


# ── datasets / schema / data ─────────────────────────────────────────────────

@app.get("/api/datasets")
def list_datasets():
    out = []
    for d in SERVED:
        ds = datasets.get(d)
        out.append({"name": d, "tables": len(ds.files), "trained": os.path.isdir(os.path.join(ds.models_dir, "split")),
                    "benchmark": os.path.exists(os.path.join(ds.models_dir, "benchmark.json"))})
    return out


@app.get("/api/health")
def health(dataset: str = "hr"):
    S = svc(dataset)
    return {"status": "ok", "dataset": dataset, "estimators": S.available(),
            "tables": {t: tb.n for t, tb in S.engine.catalog.tables.items()},
            "workload": 0 if _workloads[dataset] is None else len(_workloads[dataset]),
            "postgres_live": bool(os.environ.get("PG_DSN")), "data_hash": S.fingerprint, "stale_models": S.stale}


@app.get("/api/schema")
def schema(dataset: str = "hr"):
    S = svc(dataset)
    out = []
    for t, tb in S.engine.catalog.tables.items():
        cols = []
        for c in tb.cols:
            v = tb.df[c]
            info = {"name": c, "type": "number" if tb.numeric[c] else "text", "distinct": int(v.nunique()),
                    "nulls": int(v.isna().sum())}
            if c in S.ds.timestamps.get(t, []):
                info["type"] = "timestamp"
            if tb.numeric[c]:
                info.update(min=_py(v.min()), max=_py(v.max()))
            elif info["distinct"] <= 12:
                info["values"] = sorted(map(str, v.dropna().unique()))
            cols.append(info)
        out.append({"name": t, "rows": tb.n, "columns": cols})
    return {"tables": out, "joins": [{"left": f"{a}.{b}", "right": f"{c}.{d}"} for a, b, c, d in S.ds.join_edges]}


@app.get("/api/tables/{name}")
def table_rows(name: str, dataset: str = "hr", page: int = 1, page_size: int = 25):
    tb = svc(dataset).engine.catalog.tables.get(name)
    if tb is None:
        raise HTTPException(404, f"unknown table {name}")
    page_size = max(1, min(page_size, 200))
    start = (max(page, 1) - 1) * page_size
    chunk = tb.df.iloc[start:start + page_size]
    return {"columns": list(tb.df.columns), "rows": chunk.astype(object).where(chunk.notna(), None).values.tolist(),
            "total": tb.n, "page": page, "page_size": page_size}


@app.get("/api/bn/{table}")
def bayes_net(table: str, dataset: str = "hr"):
    bn = svc(dataset).fj.bns.get(table)
    if bn is None:
        raise HTTPException(404, f"unknown table {table}")
    nodes = [{"id": c, "bins": bn.disc[c].labels(), "marginal": np.round(bn.marg[c], 4).tolist()} for c in bn.cols]
    edges = [{"parent": a, "child": b, "mi": round(w, 4), "cpt": np.round(bn.cpt(b, a), 4).tolist()}
             for a, b, w in bn.edges]
    pruned = [{"a": a, "b": b, "mi": round(v[0], 4), "mi_corrected": round(v[1], 4)}
              for (a, b), v in bn.pair_mi.items() if not v[2]]
    return {"table": table, "rows": bn.n, "nodes": nodes, "edges": edges, "pruned": pruned}


@app.get("/api/workload/sample")
def workload_sample(dataset: str = "hr", category: Optional[str] = None, n: int = 6, seed: Optional[int] = None):
    svc(dataset)
    wl = _workloads[dataset]
    if wl is None:
        raise HTTPException(404, "workload not generated - run scripts.build_workload")
    df = wl[wl.split == "test"]
    if category:
        df = df[df.category == category]
    df = df.sample(n=min(n, len(df)), random_state=seed)
    return [{"sql": r.sql, "category": r.category, "true_count": int(r.true_count)} for r in df.itertuples()]


# ── the shoot-out ───────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    sql: str = Field(..., max_length=5000)
    row_limit: int = Field(100, ge=1, le=1000)
    dataset: str = "hr"


def _live_pg(dataset: str) -> Optional[LivePG]:
    base = os.environ.get("PG_DSN")
    return LivePG(dsn_for(base, datasets.get(dataset))) if base else None


def _postgres_live(pg: Optional[LivePG], sql: str) -> Optional[dict]:
    if pg is None:
        return {"error": "PostgreSQL not connected (set PG_DSN and run docker compose up -d)"}
    try:
        t0 = time.perf_counter()
        est = pg.explain_rows(sql)  # sql is the engine's normalised SELECT
        return {"estimate": est, "latency_ms": (time.perf_counter() - t0) * 1000}
    except Exception as e:  # live PG is optional; report, never fail the request
        return {"error": str(e).splitlines()[0]}


def _plans(S: Estimators, q, names: List[str], pg: Optional[LivePG] = None) -> Optional[dict]:
    if len(q.rels) < 3 or len(q.rels) > MAX_PLAN_RELS:
        return None
    subsets = optimizer.connected_subsets(q)
    subq = [q.sub(s) for s in subsets]
    true = {s: float(S.engine.true_count(sq)) for s, sq in zip(subsets, subq)}
    opt_cost, opt_plan = optimizer.best_plan(q, true)
    out = {"optimal_cost": opt_cost, "optimal": optimizer.annotate(opt_plan, true, true), "estimators": {}}
    est_by = {n: S.estimate_many(subq, n)["est"].tolist() for n in names}
    if pg is not None:
        try:
            est_by = {"postgres_live": [max(pg.explain_rows(x.to_sql()), 1.0) for x in subq], **est_by}
        except Exception:
            pass  # live PG optional
    for n, vals in est_by.items():
        ests = dict(zip(subsets, vals))
        _, plan = optimizer.best_plan(q, ests)
        cost = optimizer.plan_true_cost(plan, true)
        out["estimators"][n] = {"true_cost": cost, "ratio": cost / max(opt_cost, 1.0),
                                "plan": optimizer.annotate(plan, ests, true)}
    return out


@app.post("/api/run")
def run(req: RunRequest):
    S = svc(req.dataset)
    try:
        res = S.engine.run(req.sql, req.row_limit)
    except SQLError as e:
        raise HTTPException(400, str(e))
    q = res["query"]
    count_q = q.sub(q.rels)
    names = S.available()
    ests = []
    for n in names:
        try:
            est, ms, info = S.estimate(count_q, n)
        except Exception as e:  # pragma: no cover - service already falls back internally
            raise HTTPException(500, f"{n} failed: {e}")
        ests.append({"name": n, "label": LABELS[n], "estimate": est, "latency_ms": ms,
                     "qerror": float(qerror(est, res["count"])), "under": est < max(res["count"], 1), **info})
    try:
        pg = _live_pg(req.dataset)
    except Exception as e:
        pg, pg_err = None, str(e).splitlines()[0]
    else:
        pg_err = None
    live = _postgres_live(pg, count_q.to_sql())
    if "estimate" in live:
        ests.insert(0, {"name": "postgres_live", "label": LABELS["postgres_live"], "estimate": live["estimate"],
                        "latency_ms": live["latency_ms"], "qerror": float(qerror(live["estimate"], res["count"])),
                        "under": live["estimate"] < max(res["count"], 1)})
    S.log_feedback(count_q.to_sql(), res["count"], {e["name"]: e["estimate"] for e in ests})
    rows = res["rows"]
    scan = []
    for a, s in res["scan"].items():
        one = count_q.sub([a])
        pg_rows = None
        if pg is not None:
            try:
                pg_rows = pg.explain_rows(one.to_sql())
            except Exception:
                pass
        scan.append({"table": s["table"], "actual": s["actual"], "total": s["total"], "pg": pg_rows, "alias": a,
                     "factorjoin": S.fj.table_estimate(count_q, a), "preds": [p.to_sql() for p in q.preds_of(a)]})
    plans = _plans(S, count_q, names, pg)
    if pg is not None:
        pg.close()
    return {
        "sql": q.to_sql(), "count": res["count"], "dataset": req.dataset,
        "exec_ms": res["count_ms"] + res["fetch_ms"], "count_ms": res["count_ms"], "fetch_ms": res["fetch_ms"],
        "rows": None if rows is None else {"columns": list(rows.columns),
                                           "data": rows.astype(object).where(rows.notna(), None).values.tolist(),
                                           "shown": len(rows), "truncated": len(rows) < res["count"]},
        "estimators": ests,
        "best": min(ests, key=lambda e: e["qerror"])["name"],
        "upper_bound": S.bounds.upper(count_q),
        "scan": scan,
        "plans": plans,
        "postgres_live_error": pg_err or live.get("error"),
    }


@app.get("/api/feedback")
def feedback(dataset: str = "hr", window: int = 500):
    return svc(dataset).feedback_summary(max(10, min(window, 10000)))


# ── results / model internals ───────────────────────────────────────────────

@app.get("/api/benchmark")
def benchmark(dataset: str = "hr"):
    return _json_file(os.path.join(datasets.get(dataset).models_dir, "benchmark.json"))


@app.get("/api/tuning/{name}")
def tuning(name: str, dataset: str = "hr"):
    if name not in ("mlp", "treelstm", "mscn"):
        raise HTTPException(404, "unknown model")
    ds = datasets.get(dataset)
    out = _json_file(os.path.join(ds.models_dir, "tuning", f"{name}.json"))
    meta = os.path.join(ds.models_dir, "split", "ensemble.json")
    if os.path.exists(meta):
        out["ensemble"] = json.load(open(meta)).get(name)
    return out


def _viz_global(dataset: str, name: str) -> dict:
    S = svc(dataset)
    if name not in S.ensembles or name == "mscn":
        raise HTTPException(404, f"model {name} not trained for {dataset}")
    key = f"{dataset}/{name}"
    if key not in _viz_cache:
        path = os.path.join(S.ds.models_dir, f"viz_{name}.json")
        model_path = os.path.join(S.ds.models_dir, "split", f"{name}_s0.pt")
        if os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(model_path):
            with open(path) as f:
                _viz_cache[key] = json.load(f)
        else:
            wl = _workloads[dataset]
            df = wl[wl.split == "test"].sample(n=min(800, (wl.split == "test").sum()), random_state=0)
            data = viz.global_analysis(S, [S.engine.parse(s) for s in df.sql], name)
            with open(path, "w") as f:
                json.dump(data, f)
            _viz_cache[key] = data
    return _viz_cache[key]


@app.get("/api/viz/global")
def viz_global(model: str = "treelstm", dataset: str = "hr"):
    return _viz_global(dataset, model)


class TraceRequest(BaseModel):
    sql: str = Field(..., max_length=5000)
    model: str = "treelstm"
    dataset: str = "hr"


@app.post("/api/viz/trace")
def viz_trace(req: TraceRequest):
    g = _viz_global(req.dataset, req.model)
    S = svc(req.dataset)
    try:
        q = S.engine.parse(req.sql)
    except SQLError as e:
        raise HTTPException(400, str(e))
    out = viz.trace_query(S, q, g.get("probes", {}), req.model)
    out["true_count"] = S.engine.true_count(q.sub(q.rels))
    return out


@app.middleware("http")
async def no_stale_ui(request, call_next):
    """Hashed asset files may be cached forever; index.html must not, or browsers keep
    loading an old bundle after a rebuild."""
    resp = await call_next(request)
    if not request.url.path.startswith(("/api", "/assets")):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


DIST = os.path.join("frontend", "dist")
if os.path.isdir(DIST):  # serve the built UI from the same origin
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
