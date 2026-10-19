# Cardinality Lab: learned cardinality estimation, evaluated like a production component

A query optimizer chooses join orders from *estimated* row counts. This project builds a
probabilistic + learned estimator, then evaluates it the way an industrial team would:

* **on real data**: the STATS benchmark (Stack Exchange, 8 tables, ~1 M rows) and its
  official STATS-CEB workload, plus a synthetic schema with planted correlations;
* **against real PostgreSQL 16**: its own estimates, with and without `CREATE STATISTICS`,
  plus the standard learned baseline MSCN;
* **on queries and data the models never saw**: held-out templates, held-out join sizes,
  and a database that doubled in size after training;
* **end to end**: estimates injected into PostgreSQL with `pg_hint_plan`, and the queries
  actually executed and timed;
* **with guarantees**: ensembles with uncertainty fallback, provable upper bounds,
  order-invariant featurisation, property-based and fuzz tests, CI regression gating.

| Estimator | Idea |
|---|---|
| **PostgreSQL 16** | Real server, `EXPLAIN` estimates. Every PostgreSQL number in this README and the app comes from it. |
| **PG + CREATE STATISTICS** | PostgreSQL's own fix for correlated columns (ndistinct, dependencies, MCV on each table's filter columns). |
| **FactorJoin** | Per-table Chow-Liu Bayesian-network forests (MLE, G-test pruned), joined by per-key-bin messages (Wu et al., SIGMOD 2023). |
| **MSCN** | Multi-set convolutional network with sample bitmaps (Kipf et al., CIDR 2019): the standard learned baseline. |
| **Fusion MLP** | MLP on BN selectivities + FactorJoin + PostgreSQL estimates, predicting a log-space correction to FactorJoin. |
| **TreeLSTM + MLP** | Child-Sum TreeLSTM over the canonical plan tree feeding the same MLP head. |

## Results

All numbers below are generated from `models/*/benchmark.json` and `models/stats/e2e.json`
by `python -m scripts.report`. Q-error = max(est/true, true/est), 1× = exact.

<!-- RESULTS:BEGIN -->

### STATS (real Stack Exchange data)

1,201 held-out test queries (5,598 train / 1,201 val).

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 1.84× | 357× | **38,109×** (12,234×–151,059×) | 49,192,339× | 51.5% |
| PostgreSQL 16 + CREATE STATISTICS | 1.85× | 356× | **38,109×** (12,234×–150,724×) | 52,836,216× | 51.5% |
| FactorJoin (Chow-Liu BN) | 1.41× | 30.20× | **955×** (238×–4,532×) | 10,137,790× | 65.4% |
| MSCN (baseline) | 1.77× | 106× | **14,940×** (3,709×–86,088×) | 3,374,839× | 55.2% |
| Fusion MLP | 1.20× | 13.90× | **836×** (154×–3,610×) | 207,000× | 75.5% |
| TreeLSTM + MLP | 1.11× | 5.40× | **92.54×** (35.26×–417×) | 6,924× | 83.8% |

**Official STATS-CEB workload (146 queries):**

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 18.79× | 15,533× | **503,794×** (36,985×–904,501×) | 904,501× | 19.9% |
| PostgreSQL 16 + CREATE STATISTICS | 18.88× | 16,016× | **463,060×** (37,218×–961,930×) | 961,930× | 19.9% |
| FactorJoin (Chow-Liu BN) | 6.56× | 5,573× | **66,450×** (7,555×–10,105,126×) | 10,105,126× | 26.0% |
| MSCN (baseline) | 7.00× | 722× | **13,417×** (883×–26,658×) | 26,658× | 24.0% |
| Fusion MLP | 3.22× | 1,722× | **20,560×** (2,304×–4,783,604×) | 4,783,604× | 35.6% |
| TreeLSTM + MLP | 2.16× | 193× | **1,580×** (398×–7,213×) | 7,213× | 47.3% |

Paired Wilcoxon signed-rank tests on per-query log Q-error:

| A vs B | mean log-Q A / B | A closer / B closer | p-value |
|---|---|---|---|
| TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | 0.405 / 0.883 | 61% / 38% | 2.4e-63 |
| Fusion MLP vs FactorJoin (Chow-Liu BN) | 0.645 / 0.883 | 53% / 45% | 2.6e-24 |
| TreeLSTM + MLP vs Fusion MLP | 0.405 / 0.645 | 63% / 36% | 2e-32 |
| FactorJoin (Chow-Liu BN) vs PostgreSQL 16 | 0.883 / 1.589 | 66% / 33% | 1.9e-26 |
| FactorJoin (Chow-Liu BN) vs PostgreSQL 16 + CREATE STATISTICS | 0.883 / 1.589 | 66% / 33% | 8.8e-27 |

Seed spread (each learned model trained with independent seeds):

| Model | seeds | median | p95 | p99 |
|---|---|---|---|---|
| MSCN (baseline) | 5 | 2.206 ± 0.132 | 315.52 ± 133.65 | 41045.3 ± 27281.6 |
| Fusion MLP | 5 | 1.207 ± 0.011 | 15.69 ± 0.66 | 892.1 ± 267.5 |
| TreeLSTM + MLP | 5 | 1.132 ± 0.010 | 6.25 ± 0.45 | 118.7 ± 21.2 |

**Out of distribution: Unseen query templates** (1,485 queries)

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 1.89× | 421× | **162,217×** | 107,902,772× | 51.9% |
| PostgreSQL 16 + CREATE STATISTICS | 1.85× | 406× | **182,094×** | 130,734,044× | 52.5% |
| FactorJoin (Chow-Liu BN) | 1.53× | 28.01× | **617×** | 25,323,511× | 62.5% |
| MSCN (baseline) | 2.08× | 164× | **10,428×** | 23,521,082× | 48.4% |
| Fusion MLP | 1.22× | 14.84× | **350×** | 25,712,028× | 73.6% |
| TreeLSTM + MLP | 1.14× | 6.22× | **52.20×** | 1,006,490× | 83.0% |

**Out of distribution: Unseen join sizes (train 1–3 tables, test 4–5)** (1,712 queries)

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 13.21× | 41,091× | **11,879,518×** | 245,863,658× | 18.3% |
| PostgreSQL 16 + CREATE STATISTICS | 12.77× | 39,622× | **13,447,962×** | 288,885,153× | 19.0% |
| FactorJoin (Chow-Liu BN) | 4.04× | 274× | **81,652×** | 25,323,511× | 31.9% |
| MSCN (baseline) | 9.53× | 10,821× | **264,090×** | 11,246,309× | 20.7% |
| Fusion MLP | 2.43× | 197× | **94,819×** | 59,451,928× | 41.8% |
| TreeLSTM + MLP | 1.80× | 64.63× | **6,788×** | 3,295,149× | 55.0% |

**Data drift** (models built on the 2013-06 snapshot, about half of today's rows):

| Variant | Estimator | median | p95 | p99 |
|---|---|---|---|---|
| stale (BN + NN from 2013) | FactorJoin (Chow-Liu BN) | 9.42× | 30,502,414× | 22,506,455,809× |
| stale (BN + NN from 2013) | MSCN (baseline) | 8.20× | 881,381× | 1,337,560,151× |
| stale (BN + NN from 2013) | Fusion MLP | 5.29× | 30,502,414× | 22,506,455,809× |
| stale (BN + NN from 2013) | TreeLSTM + MLP | 4.19× | 30,502,414× | 22,506,455,809× |
| BN refreshed, NN stale | FactorJoin (Chow-Liu BN) | 1.41× | 30.20× | 955× |
| BN refreshed, NN stale | MSCN (baseline) | 2.82× | 337× | 82,965× |
| BN refreshed, NN stale | Fusion MLP | 1.23× | 15.39× | 596× |
| BN refreshed, NN stale | TreeLSTM + MLP | 1.14× | 6.30× | 92.62× |
| fully retrained | PostgreSQL 16 | 1.84× | 357× | 38,109× |
| fully retrained | FactorJoin (Chow-Liu BN) | 1.41× | 30.20× | 955× |
| fully retrained | MSCN (baseline) | 1.77× | 106× | 14,940× |
| fully retrained | Fusion MLP | 1.20× | 13.90× | 836× |
| fully retrained | TreeLSTM + MLP | 1.11× | 5.40× | 92.54× |

**Join-order quality** (DP optimizer on each estimator's sub-plan estimates, scored with true C_out):

| Estimator | optimal plan | mean cost / optimal | worst |
|---|---|---|---|
| PostgreSQL 16 | 80.7% | 2.968× | 596× |
| FactorJoin (Chow-Liu BN) | 88.1% | 2.284× | 596× |
| MSCN (baseline) | 75.4% | 2.909× | 596× |
| Fusion MLP | 89.4% | 2.277× | 596× |
| TreeLSTM + MLP | 91.3% | 2.047× | 596× |

**Robustness** (guards and property checks):

| Estimator | uncertainty fallback | clamped | order-invariant | monotone | latency p50 / p99 |
|---|---|---|---|---|---|
| PostgreSQL 16 | 0.0% | 0.0% | 62% | 100% | 2.7 / 10.4 ms |
| FactorJoin (Chow-Liu BN) | 0.0% | 0.1% | 100% | 100% | 0.2 / 0.6 ms |
| MSCN (baseline) | 4.9% | 1.8% | 100% | 74% | 6.0 / 32.1 ms |
| Fusion MLP | 0.0% | 1.4% | 100% | 87% | 3.8 / 11.7 ms |
| TreeLSTM + MLP | 0.0% | 0.7% | 100% | 72% | 7.3 / 26.8 ms |
| exact execution | | | | | 21.6 / 125.2 ms |


### End-to-end runtime on PostgreSQL 16 (STATS-CEB)

Each estimator's sub-plan cardinalities injected with pg_hint_plan, all 146 STATS-CEB queries executed, timeout 30 s (timeouts charged in full). Note: postgres baseline re-run on an idle machine; the first run overlapped Windows standby (649.5 s, 13 timeouts).

| Configuration | execution | + estimation | net vs PostgreSQL | timeouts | faster / slower queries (>10%) |
|---|---|---|---|---|---|
| PostgreSQL 16 (own estimates) | 595 s | – | 1.000× | 11 | 0 / 0 |
| PostgreSQL 16 + CREATE STATISTICS | 639 s | – | 0.932× | 11 | 4 / 74 |
| True cardinalities (oracle) | 585 s | n/a | 1.017× | 11 | 44 / 55 |
| FactorJoin (Chow-Liu BN) | 580 s | 1.7 s | 1.024× | 11 | 35 / 43 |
| MSCN (baseline) | 601 s | 16.0 s | 0.965× | 12 | 36 / 44 |
| Fusion MLP | 583 s | 3.8 s | 1.016× | 12 | 41 / 29 |
| TreeLSTM + MLP | 573 s | 4.7 s | 1.030× | 12 | 45 / 26 |

Estimation = time to estimate all 1566 sub-plans (one batched call per query, cold caches) after the latency optimisations; PostgreSQL's own planning time is excluded from every row alike. The oracle's true counts cannot be produced at planning time, so it is an upper bound.


### HR (synthetic, planted correlations)

1,202 held-out test queries (5,599 train / 1,199 val).

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 1.04× | 5.22× | **1,540×** (79.07×–13,978×) | 2,086,786× | 89.0% |
| PostgreSQL 16 + CREATE STATISTICS | 1.04× | 4.83× | **1,540×** (81.49×–14,502×) | 3,431,292× | 90.3% |
| FactorJoin (Chow-Liu BN) | 1.01× | 1.92× | **14.09×** (7.36×–87.65×) | 3,446× | 95.0% |
| MSCN (baseline) | 1.16× | 4.21× | **306×** (12.74×–3,446×) | 405,192× | 88.9% |
| Fusion MLP | 1.01× | 1.84× | **7.12×** (5.29×–10.15×) | 576× | 95.7% |
| TreeLSTM + MLP | 1.02× | 1.84× | **7.08×** (5.33×–10.04×) | 563× | 95.6% |

Paired Wilcoxon signed-rank tests on per-query log Q-error:

| A vs B | mean log-Q A / B | A closer / B closer | p-value |
|---|---|---|---|
| TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | 0.127 / 0.152 | 26% / 73% | 7.8e-57 |
| Fusion MLP vs FactorJoin (Chow-Liu BN) | 0.120 / 0.152 | 31% / 68% | 2.1e-27 |
| TreeLSTM + MLP vs Fusion MLP | 0.127 / 0.120 | 27% / 71% | 3.2e-47 |
| FactorJoin (Chow-Liu BN) vs PostgreSQL 16 | 0.152 / 0.374 | 62% / 37% | 3e-60 |
| FactorJoin (Chow-Liu BN) vs PostgreSQL 16 + CREATE STATISTICS | 0.152 / 0.353 | 61% / 38% | 2.1e-54 |

Seed spread (each learned model trained with independent seeds):

| Model | seeds | median | p95 | p99 |
|---|---|---|---|---|
| MSCN (baseline) | 5 | 1.255 ± 0.025 | 5.14 ± 0.43 | 786.3 ± 317.2 |
| Fusion MLP | 5 | 1.012 ± 0.003 | 1.85 ± 0.03 | 7.3 ± 0.1 |
| TreeLSTM + MLP | 5 | 1.020 ± 0.011 | 1.87 ± 0.07 | 7.4 ± 0.6 |

**Out of distribution: Unseen query templates** (1,251 queries)

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 1.05× | 4.10× | **22,566×** | 4,269,361× | 87.2% |
| PostgreSQL 16 + CREATE STATISTICS | 1.04× | 3.82× | **18,284×** | 4,387,676× | 88.9% |
| FactorJoin (Chow-Liu BN) | 1.01× | 1.91× | **50.92×** | 11,432× | 95.2% |
| MSCN (baseline) | 1.17× | 3.61× | **2,040×** | 2,567,224× | 87.2% |
| Fusion MLP | 1.01× | 1.79× | **6.42×** | 10,148× | 95.6% |
| TreeLSTM + MLP | 1.02× | 1.76× | **6.69×** | 8,100× | 95.5% |

**Out of distribution: Unseen join sizes (train 1–3 tables, test 4–5)** (1,950 queries)

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 1.17× | 8.97× | **175,300×** | 4,100,576× | 80.5% |
| PostgreSQL 16 + CREATE STATISTICS | 1.14× | 8.09× | **175,668×** | 4,180,544× | 82.1% |
| FactorJoin (Chow-Liu BN) | 1.04× | 3.25× | **201×** | 24,690× | 90.8% |
| MSCN (baseline) | 1.80× | 87.39× | **1,537×** | 171,927× | 52.5% |
| Fusion MLP | 1.04× | 3.07× | **86.19×** | 23,932× | 91.2% |
| TreeLSTM + MLP | 1.05× | 3.02× | **30.72×** | 15,481× | 91.1% |

**Join-order quality** (DP optimizer on each estimator's sub-plan estimates, scored with true C_out):

| Estimator | optimal plan | mean cost / optimal | worst |
|---|---|---|---|
| PostgreSQL 16 | 94.1% | 7.551× | 2,506× |
| FactorJoin (Chow-Liu BN) | 95.6% | 0.980× | 3.54× |
| MSCN (baseline) | 88.7% | 5.719× | 2,506× |
| Fusion MLP | 95.4% | 0.980× | 3.54× |
| TreeLSTM + MLP | 95.3% | 0.980× | 3.54× |

**Robustness** (guards and property checks):

| Estimator | uncertainty fallback | clamped | order-invariant | monotone | latency p50 / p99 |
|---|---|---|---|---|---|
| PostgreSQL 16 | 0.0% | 0.0% | 79% | 100% | 2.4 / 5.0 ms |
| FactorJoin (Chow-Liu BN) | 0.0% | 0.0% | 100% | 100% | 0.1 / 0.4 ms |
| MSCN (baseline) | 3.7% | 0.3% | 100% | 93% | 5.1 / 11.4 ms |
| Fusion MLP | 0.0% | 0.1% | 100% | 94% | 2.0 / 4.9 ms |
| TreeLSTM + MLP | 0.0% | 0.0% | 100% | 86% | 5.2 / 11.1 ms |
| exact execution | | | | | 2.0 / 7.6 ms |

<!-- RESULTS:END -->

## How it works

**Engine (`src/engine`).** sqlglot parser into a logical query. Exact execution counts
acyclic joins by message passing over the join tree (no join materialisation) and fetches
rows through a semi-join reducer. It matches all 146 STATS-CEB ground-truth counts (up to
1.8 × 10¹⁰ rows). NULL and timestamp semantics follow SQL. Predicates are answered from
per-column sorted indexes (binary search, not full scans).

*Internal only:* `src/engine/pg_planner.py` re-implements PostgreSQL's ANALYZE statistics and
`selfuncs.c` rules (matching the real planner on 100% of single-table estimates). It is
never reported as a result. It supplies one input feature of the fusion models (so they
need no database connection at inference) and is the last step of the fallback chain.

**Bayesian networks (`src/estimator/bn.py`).** One network per table over the columns
queries can touch (filters + join keys). Edges are maximum-MI (Chow-Liu), weighted by
Miller-Madow-corrected MI and kept only if a G-test rejects independence (p < 0.001), so
the structure is a forest and noise never becomes an edge. CPTs by MLE with α = 10⁻³. NULL
is its own state. Join keys use shared bins: heavy hitters alone, the rest equal-row-mass.

**FactorJoin (`src/estimator/factorjoin.py`).** Each relation sends its parent
N·P(key bin, filters, child evidence); children enter as soft evidence m(b)/ndv(b).

**Learned models (`src/estimator/neural.py`, `mscn.py`).** Queries are canonicalised
(relations, joins and predicates in a fixed order) before featurisation, so writing a
query differently can never change an estimate. Fusion models predict a residual on
log FactorJoin.

**Serving guards (`src/estimator/service.py`).**
1. Each learned model is a multi-seed ensemble (mean of log estimates).
2. If the seeds disagree beyond τ (95th percentile of validation spread), FactorJoin answers.
3. Errors fall back along neural → FactorJoin → internal planner model.
4. Every estimate is clamped to a provable upper bound: |T_root| × Π max key degree, minimised over roots.
5. Models carry a data fingerprint and format version and are flagged stale if the data changed.
6. Each executed query logs estimate vs actual (`/api/feedback`) for drift monitoring.

## Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.lock            # exact versions used for the results

python -m scripts.fetch_stats               # STATS + STATS-CEB (~40 MB)
python -m scripts.make_drift                # 2013-06 snapshot for the drift experiment
python -m scripts.build_workload --dataset hr
python -m scripts.build_workload --dataset stats
python -m scripts.build_workload --dataset stats_old --n 6000

python -m scripts.tune  --dataset stats     # Optuna (TPE + median pruning) per model
python -m scripts.train --dataset stats --seeds 5
python -m scripts.train --dataset stats --split split_tmpl --seeds 3
python -m scripts.train --dataset stats --split split_shape --seeds 3
#   (same for --dataset hr; stats_old trains with the STATS configs for the drift study)

docker compose up -d                        # PostgreSQL 16 + pg_hint_plan on port 55432
python -m scripts.init_postgres --dataset hr
python -m scripts.init_postgres --dataset stats
python -m scripts.benchmark --dataset hr
python -m scripts.benchmark --dataset stats
python -m scripts.e2e_runtime               # executes STATS-CEB under every estimator
python -m scripts.report                    # refresh the tables above
pytest -q                                   # unit, property, fuzz, API tests
```

App: `python server.py`, then `cd frontend && npm install && npm run dev`, or open
http://127.0.0.1:8000 after `npm run build`. Set
`PG_DSN=postgresql://postgres:postgres@127.0.0.1:55432/carddb` for live PostgreSQL rows.

## Limitations (stated, not hidden)

* `Rows()` hints set join cardinalities only; base-table estimates stay PostgreSQL's in
  every end-to-end configuration.
* Learned models are not monotone by construction: adding a filter raised the TreeLSTM's
  estimate on a sizeable share of probe queries (see the robustness table). The upper-bound
  clamp caps the damage but does not fix it; a monotone architecture is future work.
* Latency: ensembles run as one stacked model (all seeds in a single tree walk, numerically
  identical to running them one by one) and features use sorted indexes instead of column
  scans, which cut a guarded TreeLSTM estimate on STATS from ~12 ms to ~5–7 ms. It is still
  Python on CPU. A production version would live inside the database in C++. The
  end-to-end table above charges every estimator its full estimation time.
* The "stale" drift variant also uses upper bounds computed on the old snapshot, which
  are invalid for the grown database. Refreshing statistics (BNs and bounds, seconds of
  counting) recovers full accuracy without retraining the networks.
* The uncertainty fallback is enabled per model only when it beat the model on uncertain
  validation queries; it is on for MSCN only. For the fusion models, FactorJoin was worse
  exactly where they were unsure.
* Workloads are generated from the schema's join graph. STATS-CEB is the only
  human-written workload.
