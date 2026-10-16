# Cardinality Lab: learned cardinality estimation without a database server

A query optimizer picks join orders from *estimated* row counts. This project compares
four estimators on the same data and the same queries:

| Estimator | Idea |
|---|---|
| **PostgreSQL planner** | Faithful emulation of PostgreSQL's `ANALYZE` statistics and `selfuncs.c` rules (MCVs, histograms, `eqjoinsel`). Assumes columns are independent. |
| **FactorJoin** | Per-table Chow-Liu Bayesian networks (MLE), joined by passing per-key-bin messages across tables (Wu et al., SIGMOD 2023). |
| **Fusion MLP** | MLP on BN selectivities + FactorJoin + PostgreSQL estimates; predicts a residual on top of log FactorJoin. |
| **TreeLSTM + MLP** | Child-Sum TreeLSTM over the query's plan tree, whose root state feeds the same MLP head. |

Everything runs on an **embedded SQL engine** (`src/engine`): no PostgreSQL needed.
Queries execute exactly (true counts, real result rows), and PostgreSQL's estimates
come from its planner emulation. If `PG_DSN` is set, the API also reports a live
`EXPLAIN` estimate from a real server.

## Results (held-out test split)

| Estimator | Median | p95 | p99 | Max | ≤2× | p99 (correlated) | Optimal join order | Latency p50 |
|---|---|---|---|---|---|---|---|---|
| PostgreSQL planner | 1.04× | 4.29× | 22.76× | 59,392× | 86.7% | 987× | 94.3% | 0.10 ms |
| FactorJoin (BN) | 1.01× | 1.87× | 5.29× | 66.42× | 95.4% | 6.80× | 96.9% | 0.39 ms |
| Fusion MLP | 1.01× | 1.86× | 4.76× | 45.45× | 95.3% | 6.07× | 96.7% | 2.82 ms |
| TreeLSTM + MLP | 1.02× | 1.79× | 4.88× | 41.83× | 95.5% | 5.60× | 95.9% | 5.11 ms |

1,202 test queries (5,598 train / 1,200 val); correlated slice = 535 queries with ≥2 predicates on correlated columns. Exact execution p50: 1.70 ms. Join orders: 609 queries with 3+ tables, DP optimizer, scored by true C_out.

Reading it honestly: both learned models cut the tail versus FactorJoin (max 66× → 42–45×, p99 5.3× → 4.8×); the TreeLSTM is best on p95, max and ≤2×, the plain MLP on p99. FactorJoin alone already gives the best join orders, so on this data the neural correction improves accuracy, not plan choice.

Full numbers, per-category breakdowns and scatter data: `models/benchmark.json`
(also shown in the web app's **Benchmark** page).

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install -r requirements.txt

python -m scripts.build_workload --n 8000                # labelled workload (≈30 s)
python -m scripts.tune --mlp-trials 40 --tree-trials 24  # Optuna tuning + final models
python -m scripts.benchmark                              # accuracy, latency, plan quality

python server.py                                         # API on http://127.0.0.1:8000
cd frontend && npm install && npm run dev                # UI on http://localhost:5173
```

`npm run build` puts the UI in `frontend/dist`. `server.py` then serves it at
http://127.0.0.1:8000 directly.

To regenerate the synthetic data: `python -m src.schema.data_generator` (reads `config.yaml`).

## Web app

* **Query Lab**: type any supported SQL. It runs exactly, every estimator is scored
  against the true count, per-table filter estimates are shown, and the join order each
  estimator would choose is drawn and costed with true intermediate sizes.
* **Benchmark**: Q-error percentiles (all / correlated slice / per join size),
  estimate-vs-truth scatters, plan-quality and latency comparison, Optuna trial history and
  parameter importances, and auto-generated résumé lines from the measured numbers.
* **Inside the Models**: the query-model analogue of CNN filter visualisation. Tree
  levels (condition → table → join → query) with linear-probe R², gate statistics, a neuron
  atlas showing what each hidden unit fires on, a PCA map of hidden states, a live bottom-up
  trace of any query, MLP layer-wise PCA/probes, first-layer receptive fields and
  integrated-gradient attributions.
* **Data & Engine**: schema, table browser, and each table's Chow-Liu network with
  MI-weighted edges and CPT heatmaps.

## Supported SQL

`SELECT COUNT(*) | * | t.col, … FROM t a JOIN u b ON a.x = b.y … WHERE` a conjunction of
`col = | <> | < | <= | > | >= literal`, `col BETWEEN lo AND hi`, `col IN (…)`, `a.x = b.y`,
plus `LIMIT n`. `OR`, subqueries, `GROUP BY` and `ORDER BY` return a clear 400 error. Cyclic
join graphs execute through a hash-join fallback.

## Layout

```
server.py                 FastAPI: /api/run, /api/benchmark, /api/tuning, /api/viz/*, /api/bn, /api/schema
src/engine/               sql.py (parser) · executor.py (exact counts, row fetch)
                          pg_planner.py (PostgreSQL estimate emulation) · optimizer.py (DP join order)
src/estimator/            bn.py (Chow-Liu + MLE) · factorjoin.py · neural.py (TreeLSTM, MLP, training)
                          service.py (all estimators behind one API) · viz.py (probes, atlas, IG)
src/workload/generator.py random acyclic join queries over the schema join graph
src/schema/               synthetic correlated data generator (config.yaml)
scripts/                  build_workload · tune · benchmark · init_postgres (optional live PG)
frontend/                 React + TypeScript + Vite UI
models/                   fusion_{mlp,treelstm}.pt, tuning/*.json, benchmark.json, viz_*.json
tests/                    engine-vs-pandas oracle, planner, optimizer, API contract
```

## Notes and limits

* Data is synthetic (6 tables, 5k–10k rows) with planted correlations. Most random
  predicates hit independent columns, so every estimator's *median* is close to 1×. The
  differences are in the tail (p95 and above), which is what drives bad plans.
* FactorJoin's remaining errors come from correlations that cross tables through
  high-cardinality keys (e.g. `projects.budget_tier` ↔ `budgets.approved_amount` via
  `proj_id`). Per-table BNs cannot see these; the learned models partially can.
* Latencies are single-query Python calls on CPU and include feature extraction.
