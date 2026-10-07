"""
FastAPI Backend: PostgreSQL Cardinality Estimation Research Platform
Serves 6-table synthetic dataset, BN models, FactorJoin, ML Fusion, and benchmark data.
"""
import os
import re
import time
import json
import math
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Query as QueryParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.estimator.factorjoin import FactorJoin

app = FastAPI(title="PostgreSQL Cardinality Estimation Research Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_DIR = "data/out"
MODELS_DIR = "models"
QUERIES_CSV = "data/queries/test_queries_with_counts.csv"

ALL_TABLES = ["departments", "employees", "projects", "assignments", "locations", "budgets"]

# Globals
factor_join: Optional[FactorJoin] = None
fusion_payload: Optional[Dict] = None
bn_models: Dict[str, Any] = {}


def load_artifacts():
    global factor_join, fusion_payload, bn_models
    try:
        factor_join = FactorJoin(csv_dir=CSV_DIR)
        print("FactorJoin initialized.")
    except Exception as e:
        print(f"Warning: FactorJoin init failed: {e}")

    fusion_path = os.path.join(MODELS_DIR, "fusion_model.joblib")
    if os.path.exists(fusion_path):
        try:
            fusion_payload = joblib.load(fusion_path)
            print("Fusion model loaded.")
        except Exception as e:
            print(f"Warning: Fusion model load failed: {e}")

    for tbl in ALL_TABLES:
        bn_path = os.path.join(MODELS_DIR, f"bn_{tbl}.joblib")
        if os.path.exists(bn_path):
            try:
                bn_models[tbl] = joblib.load(bn_path)
                print(f"Loaded BN for {tbl}")
            except Exception as e:
                print(f"Warning: BN load failed for {tbl}: {e}")


load_artifacts()


def to_py(v):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, dict):
        return {str(k): to_py(val) for k, val in v.items()}
    if isinstance(v, list):
        return [to_py(x) for x in v]
    return v


class EstimateRequest(BaseModel):
    sql: str


# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────
@app.get("/api/health")
def health():
    stats = fusion_payload.get("stats", {}) if fusion_payload else {}
    return {
        "status": "online",
        "factor_join": factor_join is not None,
        "fusion_model": fusion_payload is not None,
        "bn_tables": list(bn_models.keys()),
        "fusion_stats": to_py(stats),
    }


# ─────────────────────────────────────────────
# TABLE LISTING
# ─────────────────────────────────────────────
@app.get("/api/tables")
def list_tables():
    result = []
    for tbl in ALL_TABLES:
        fpath = os.path.join(CSV_DIR, f"{tbl}.csv")
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            result.append({
                "name": tbl,
                "row_count": len(df),
                "columns": [{"name": c, "type": str(df[c].dtype)} for c in df.columns],
                "head": df.head(5).fillna("").to_dict(orient="records"),
            })
    return result


# ─────────────────────────────────────────────
# TABLE DATA (paginated)
# ─────────────────────────────────────────────
@app.get("/api/tables/{table_name}")
def get_table(table_name: str, page: int = 1, page_size: int = 25, search: Optional[str] = None):
    fpath = os.path.join(CSV_DIR, f"{table_name}.csv")
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")

    df = pd.read_csv(fpath)
    if search:
        mask = df.astype(str).apply(lambda row: row.str.lower().str.contains(search.lower()).any(), axis=1)
        df = df[mask]

    total_rows = len(df)
    start = (page - 1) * page_size
    page_df = df.iloc[start: start + page_size]

    stats = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats[col] = {
                "min": to_py(df[col].min()),
                "max": to_py(df[col].max()),
                "mean": round(to_py(df[col].mean()), 2),
                "std": round(to_py(df[col].std()), 2),
                "unique": int(df[col].nunique()),
                "type": "numeric",
            }
        else:
            vc = df[col].value_counts().head(5).to_dict()
            stats[col] = {
                "unique": int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in vc.items()},
                "type": "categorical",
            }

    return {
        "table_name": table_name,
        "total_rows": total_rows,
        "page": page,
        "page_size": page_size,
        "total_pages": math.ceil(total_rows / page_size),
        "columns": [{"name": c, "type": str(df[c].dtype)} for c in df.columns],
        "rows": page_df.fillna("").to_dict(orient="records"),
        "stats": stats,
    }


# ─────────────────────────────────────────────
# BAYESIAN NETWORKS
# ─────────────────────────────────────────────
@app.get("/api/bayesian-networks")
def get_all_bns():
    result = {}
    for tbl, model in bn_models.items():
        cols = model.get("cols", [])
        edges = model.get("edges", [])
        disc = model.get("discretization", {})
        cpts_raw = model.get("cpts", {})

        nodes = []
        for col in cols:
            d = disc.get(col, {})
            bins = d.get("bins")
            nodes.append({
                "id": col,
                "label": col.replace("_", " ").title(),
                "bins_count": len(bins) - 1 if bins else None,
            })

        # Compute mutual information strengths from CPT shape
        edge_list = []
        for par, child in edges:
            edge_list.append({"source": par, "target": child, "label": f"{par} → {child}"})

        # CPT summary
        cpt_summary = {}
        for col, info in cpts_raw.items():
            if "_prior" in info:
                cpt_summary[col] = {
                    "type": "prior",
                    "parent": None,
                    "states": list(info["_prior"].keys())[:5],
                    "probs": {k: round(v, 4) for k, v in list(info["_prior"].items())[:5]},
                }
            elif "table" in info:
                par = info.get("parent", "?")
                # Sample one parent value
                sample_par = list(info["table"].keys())[0] if info["table"] else None
                sample_row = info["table"].get(sample_par, {}) if sample_par else {}
                cpt_summary[col] = {
                    "type": "conditional",
                    "parent": par,
                    "n_parent_states": len(info["table"]),
                    "sample_parent_val": sample_par,
                    "sample_probs": {k: round(v, 4) for k, v in list(sample_row.items())[:5]},
                }

        result[tbl] = {
            "table_name": tbl,
            "cols": cols,
            "nodes": nodes,
            "edges": edge_list,
            "cpt_summary": cpt_summary,
        }
    return result


@app.get("/api/cpts/{table_name}")
def get_table_cpts(table_name: str):
    if table_name not in bn_models:
        raise HTTPException(status_code=404, detail=f"No BN model for table '{table_name}'")

    model = bn_models[table_name]
    cpts_raw = model.get("cpts", {})

    # Serialize safely
    cpts_out = {}
    for col, info in cpts_raw.items():
        if "_prior" in info:
            cpts_out[col] = {
                "type": "prior",
                "parent": None,
                "data": {str(k): round(float(v), 5) for k, v in info["_prior"].items()},
            }
        elif "table" in info:
            par = info.get("parent", "?")
            table_data = {}
            for pv, row in info["table"].items():
                table_data[str(pv)] = {str(cv): round(float(p), 5) for cv, p in row.items()}
            cpts_out[col] = {
                "type": "conditional",
                "parent": par,
                "data": table_data,
            }

    return {
        "table_name": table_name,
        "cols": model.get("cols", []),
        "edges": model.get("edges", []),
        "cpts": cpts_out,
    }


# ─────────────────────────────────────────────
# ESTIMATION (live query)
# ─────────────────────────────────────────────
@app.post("/api/estimate")
def estimate(req: EstimateRequest):
    sql = req.sql.strip()
    if not sql:
        raise HTTPException(status_code=400, detail="SQL required")
    if factor_join is None:
        raise HTTPException(status_code=503, detail="FactorJoin not initialized")

    # FactorJoin inference
    feats = factor_join.extract_features(sql)
    fj_est = feats.get("factorjoin_est", 0.0)
    fj_latency = feats.get("latency_ms", 0.0)

    # ML Fusion inference
    ml_est = None
    ml_latency = None
    if fusion_payload:
        model = fusion_payload["model"]
        X = np.array([[
            feats.get("s_e", 1.0),
            feats.get("s_d", 1.0),
            feats.get("s_p", 1.0),
            feats.get("s_a", 1.0),
            feats.get("s_l", 1.0),
            feats.get("s_b", 1.0),
            feats.get("p_dept_given_emp", 1.0),
            feats.get("p_proj_given_emp", 1.0),
            fj_est,
        ]])
        t0 = time.perf_counter()
        pred_log = model.predict(X)[0]
        ml_latency = round((time.perf_counter() - t0) * 1000.0, 4)
        ml_est = max(0.0, float(np.expm1(pred_log)))

    # Ground truth from workload CSV if available
    true_count = None
    if os.path.exists(QUERIES_CSV):
        df_q = pd.read_csv(QUERIES_CSV)
        match = df_q[df_q["sql"] == sql]
        if not match.empty:
            true_count = int(match.iloc[0]["true_count"])

    # Simulated Postgres latency
    pg_latency = None
    if os.path.exists(QUERIES_CSV):
        df_q = pd.read_csv(QUERIES_CSV)
        match = df_q[df_q["sql"] == sql]
        if not match.empty and "postgres_latency_ms" in df_q.columns:
            pg_latency = float(match.iloc[0]["postgres_latency_ms"])

    def q_err(est, tc):
        if est is None or tc is None or tc <= 0:
            return None
        e, t = max(float(est), 1e-6), max(float(tc), 1e-6)
        return round(max(e / t, t / e), 3)

    return {
        "sql": sql,
        "factorjoin_estimate": round(fj_est, 2),
        "ml_fusion_estimate": round(ml_est, 2) if ml_est is not None else None,
        "true_count": true_count,
        "latency": {
            "postgres_ms": pg_latency,
            "factorjoin_ms": round(fj_latency, 4),
            "ml_fusion_ms": ml_latency,
        },
        "q_errors": {
            "factorjoin": q_err(fj_est, true_count),
            "ml_fusion": q_err(ml_est, true_count),
        },
        "features": {k: round(float(v), 5) if isinstance(v, float) else v for k, v in feats.items() if k != "latency_ms"},
    }


# ─────────────────────────────────────────────
# QUERY WORKLOAD
# ─────────────────────────────────────────────
@app.get("/api/queries")
def get_queries(category: Optional[str] = None, limit: int = 50, offset: int = 0):
    if not os.path.exists(QUERIES_CSV):
        raise HTTPException(status_code=404, detail="Query workload not generated yet")

    df = pd.read_csv(QUERIES_CSV)
    if category:
        df = df[df["category"].str.contains(category, case=False, na=False)]

    total = len(df)
    df = df.iloc[offset: offset + limit]

    records = []
    for _, row in df.iterrows():
        tc = max(1, int(row.get("true_count", 1)))
        fj_est = float(row.get("factorjoin_estimate", 0))
        pg_est = float(row.get("postgres_estimate", 0))

        def qe(est):
            e, t = max(est, 1e-6), max(tc, 1e-6)
            return round(max(e / t, t / e), 2)

        records.append({
            "id": int(row.get("id", 0)),
            "category": row.get("category", ""),
            "sql": row.get("sql", ""),
            "true_count": tc,
            "postgres_estimate": int(pg_est),
            "factorjoin_estimate": round(fj_est, 2),
            "postgres_latency_ms": float(row.get("postgres_latency_ms", 0)),
            "factorjoin_latency_ms": float(row.get("factorjoin_latency_ms", 0)),
            "postgres_q_error": qe(pg_est),
            "factorjoin_q_error": qe(fj_est),
        })

    return {"total": total, "queries": records}


# ─────────────────────────────────────────────
# BENCHMARK DASHBOARD
# ─────────────────────────────────────────────
@app.get("/api/benchmark")
def get_benchmark():
    if not os.path.exists(QUERIES_CSV):
        raise HTTPException(status_code=404, detail="Query workload not generated yet")

    df = pd.read_csv(QUERIES_CSV)

    stats = fusion_payload.get("stats", {}) if fusion_payload else {}
    ml_median_q = stats.get("median_q", None)
    ml_latency = stats.get("ml_latency_ms", None)
    fj_latency_mean = stats.get("fj_latency_ms", None)
    pg_latency_mean = stats.get("pg_latency_ms", None)

    # Compute FJ Q-errors from workload
    def qe(est, tc):
        e, t = max(float(est), 1e-6), max(float(tc), 1e-6)
        return max(e / t, t / e)

    fj_qs, pg_qs = [], []
    for _, row in df.iterrows():
        tc = max(1, int(row.get("true_count", 1)))
        fj_qs.append(qe(row.get("factorjoin_estimate", 1), tc))
        pg_qs.append(qe(row.get("postgres_estimate", 1), tc))

    # Per-category breakdown
    cat_breakdown = []
    for cat, grp in df.groupby("category"):
        tc_arr = grp["true_count"].values.astype(float)
        fj_arr = grp["factorjoin_estimate"].values.astype(float)
        pg_arr = grp["postgres_estimate"].values.astype(float)
        fj_q_arr = [qe(fj_arr[i], tc_arr[i]) for i in range(len(tc_arr))]
        pg_q_arr = [qe(pg_arr[i], tc_arr[i]) for i in range(len(tc_arr))]
        cat_breakdown.append({
            "category": cat,
            "count": len(grp),
            "fj_median_q": round(float(np.median(fj_q_arr)), 2),
            "pg_median_q": round(float(np.median(pg_q_arr)), 2),
            "fj_latency_ms": round(float(grp["factorjoin_latency_ms"].mean()), 3) if "factorjoin_latency_ms" in grp else None,
            "pg_latency_ms": round(float(grp["postgres_latency_ms"].mean()), 2) if "postgres_latency_ms" in grp else None,
        })

    def dist(qs):
        return {
            "under_2": int(sum(q < 2.0 for q in qs)),
            "2_to_10": int(sum(2.0 <= q < 10.0 for q in qs)),
            "10_to_100": int(sum(10.0 <= q < 100.0 for q in qs)),
            "over_100": int(sum(q >= 100.0 for q in qs)),
        }

    return {
        "total_queries": len(df),
        "summary": {
            "postgres_median_q": round(float(np.median(pg_qs)), 2),
            "factorjoin_median_q": round(float(np.median(fj_qs)), 2),
            "ml_fusion_median_q": round(ml_median_q, 3) if ml_median_q else None,
        },
        "latency_ms": {
            "postgres_mean": round(pg_latency_mean, 2) if pg_latency_mean else round(float(df["postgres_latency_ms"].mean()), 2) if "postgres_latency_ms" in df else None,
            "factorjoin_mean": round(fj_latency_mean, 4) if fj_latency_mean else round(float(df["factorjoin_latency_ms"].mean()), 4) if "factorjoin_latency_ms" in df else None,
            "ml_fusion_mean": round(ml_latency, 4) if ml_latency else None,
        },
        "q_error_distribution": {
            "postgres": dist(pg_qs),
            "factorjoin": dist(fj_qs),
        },
        "category_breakdown": cat_breakdown,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
