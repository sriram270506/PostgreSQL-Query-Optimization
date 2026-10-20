# Cardinality Lab: learned cardinality estimation, evaluated like a production component

A query optimizer chooses join orders from *estimated* row counts. This project builds a
probabilistic + learned estimator, then evaluates it the way an industrial team would:

* **on real data**: the STATS benchmark (Stack Exchange, 8 tables, ~1 M rows) and its
  official STATS-CEB workload, plus a synthetic schema with planted correlations;
* **against real PostgreSQL 16**: its own estimates, with and without `CREATE STATISTICS`,
  plus the standard learned baseline MSCN;
* **on queries and data the models never saw**: held-out templates, join graphs, join
  sizes and cardinality tails, never-trained sets of empty/tiny results and rare/absent
  values, and a database that doubled in size after training;
* **with honest comparisons**: paired bootstrap CIs on every model-vs-model difference
  (overall, per slice, per held-out split), per-seed agreement, and "inconclusive" when the
  CI contains zero; errors broken down by table count, join shape, predicate type and
  true-count bucket;
* **end to end**: estimates injected into PostgreSQL with `pg_hint_plan`, and the queries
  actually executed and timed;
* **with guarantees**: ensembles with uncertainty fallback, provable upper bounds,
  order-invariant featurisation, property-based and fuzz tests, CI regression gating, and
  a nightly job against real PostgreSQL in which any skipped test fails the run;
* **reproducibly**: every result file records the commit, data/workload/model hashes,
  split definitions, seeds, hardware and PostgreSQL version/settings; STATS is pinned to
  one upstream commit and checked against SHA-256 checksums.

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

1,201 held-out test queries (5,898 train / 1,201 val). Produced by commit `55d7ea2fcc` (dirty tree), data `298197cd790988b4`, workload `9b35fb22fd4a0b72`.

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 1.85× | 354× | **28,582×** (12,234×–159,540×) | 56,312,283× | 51.9% |
| PostgreSQL 16 + CREATE STATISTICS | 1.86× | 375× | **38,109×** (12,234×–153,050×) | 51,563,054× | 51.5% |
| FactorJoin (Chow-Liu BN) | 1.41× | 30.20× | **955×** (238×–4,532×) | 10,137,790× | 65.4% |
| MSCN (baseline) | 1.62× | 33.74× | **955×** (238×–5,077×) | 10,137,790× | 62.3% |
| Fusion MLP | 1.23× | 14.88× | **238×** (102×–762×) | 254,993× | 72.8% |
| TreeLSTM + MLP | 1.11× | 6.67× | **111×** (45.95×–1,595×) | 20,931× | 83.4% |

**Official STATS-CEB workload (146 queries):**

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 18.78× | 15,409× | **492,346×** (37,362×–993,468×) | 993,468× | 19.9% |
| PostgreSQL 16 + CREATE STATISTICS | 19.19× | 15,531× | **452,231×** (33,912×–865,737×) | 865,737× | 19.9% |
| FactorJoin (Chow-Liu BN) | 6.56× | 5,573× | **66,450×** (7,555×–10,105,126×) | 10,105,126× | 26.0% |
| MSCN (baseline) | 7.41× | 6,269× | **66,450×** (10,001×–10,105,126×) | 10,105,126× | 18.5% |
| Fusion MLP | 3.68× | 1,071× | **9,264×** (1,334×–28,235×) | 28,235× | 27.4% |
| TreeLSTM + MLP | 2.04× | 1,153× | **45,589×** (1,745×–10,105,126×) | 10,105,126× | 49.3% |

Paired Wilcoxon signed-rank tests on per-query log Q-error:

| A vs B | mean log-Q A / B | A closer / B closer | p-value |
|---|---|---|---|
| TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | 0.429 / 0.883 | 60% / 37% | 1.1e-60 |
| Fusion MLP vs FactorJoin (Chow-Liu BN) | 0.662 / 0.883 | 52% / 47% | 2.1e-17 |
| TreeLSTM + MLP vs Fusion MLP | 0.429 / 0.662 | 70% / 29% | 1.1e-46 |
| FactorJoin (Chow-Liu BN) vs PostgreSQL 16 | 0.883 / 1.585 | 66% / 33% | 4.7e-26 |
| FactorJoin (Chow-Liu BN) vs PostgreSQL 16 + CREATE STATISTICS | 0.883 / 1.588 | 66% / 33% | 7.2e-26 |

Seed spread (each learned model trained with independent seeds):

| Model | seeds | median | p95 | p99 |
|---|---|---|---|---|
| MSCN (baseline) | 5 | 2.593 ± 0.132 | 192.80 ± 35.15 | 6200.2 ± 2901.6 |
| Fusion MLP | 5 | 1.254 ± 0.025 | 16.48 ± 1.55 | 299.3 ± 111.0 |
| TreeLSTM + MLP | 5 | 1.126 ± 0.005 | 6.58 ± 0.44 | 90.9 ± 11.7 |

**Out of distribution: Unseen query templates** (1,485 queries, models retrained without them)

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 1.87× | 436× | **172,335×** | 127,702,879× | 52.3% |
| PostgreSQL 16 + CREATE STATISTICS | 1.90× | 426× | **166,842×** | 96,187,484× | 51.3% |
| FactorJoin (Chow-Liu BN) | 1.53× | 28.01× | **617×** | 25,323,511× | 62.5% |
| MSCN (baseline) | 1.97× | 43.55× | **939×** | 25,323,511× | 50.6% |
| Fusion MLP | 1.28× | 13.28× | **276×** | 13,130,840× | 70.0% |
| TreeLSTM + MLP | 1.13× | 6.07× | **58.43×** | 506,925× | 83.6% |

Matched: TreeLSTM + MLP vs Fusion MLP: -0.277 [-0.317, -0.239] (TreeLSTM + MLP better); TreeLSTM + MLP vs FactorJoin (Chow-Liu BN): -0.518 [-0.578, -0.464] (TreeLSTM + MLP better); Fusion MLP vs FactorJoin (Chow-Liu BN): -0.240 [-0.283, -0.198] (Fusion MLP better); TreeLSTM + MLP vs MSCN (baseline): -0.720 [-0.787, -0.658] (TreeLSTM + MLP better); TreeLSTM + MLP vs PostgreSQL 16: -1.205 [-1.324, -1.095] (TreeLSTM + MLP better)

**Out of distribution: Unseen join sizes (train 1–3 tables, test 4–5)** (1,712 queries, models retrained without them)

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 13.22× | 39,189× | **11,467,673×** | 285,853,253× | 19.0% |
| PostgreSQL 16 + CREATE STATISTICS | 11.62× | 39,231× | **13,649,444×** | 365,372,469× | 19.2% |
| FactorJoin (Chow-Liu BN) | 4.04× | 274× | **81,652×** | 25,323,511× | 31.9% |
| MSCN (baseline) | 4.47× | 865× | **68,566×** | 25,323,511× | 30.1% |
| Fusion MLP | 3.04× | 154× | **23,217×** | 14,638,680× | 33.4% |
| TreeLSTM + MLP | 1.73× | 59.96× | **8,125×** | 4,340,978× | 56.5% |

Matched: TreeLSTM + MLP vs Fusion MLP: -0.523 [-0.578, -0.469] (TreeLSTM + MLP better); TreeLSTM + MLP vs FactorJoin (Chow-Liu BN): -0.778 [-0.839, -0.715] (TreeLSTM + MLP better); Fusion MLP vs FactorJoin (Chow-Liu BN): -0.255 [-0.307, -0.201] (Fusion MLP better); TreeLSTM + MLP vs MSCN (baseline): -1.031 [-1.115, -0.941] (TreeLSTM + MLP better); TreeLSTM + MLP vs PostgreSQL 16: -2.457 [-2.624, -2.296] (TreeLSTM + MLP better)

**Out of distribution: Unseen join graphs** (963 queries, models retrained without them)

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 4.83× | 2,546× | **84,233×** | 303,751,071× | 31.9% |
| PostgreSQL 16 + CREATE STATISTICS | 4.85× | 2,520× | **86,102×** | 289,283,937× | 32.3% |
| FactorJoin (Chow-Liu BN) | 2.20× | 45.33× | **5,663×** | 21,306,067× | 46.2% |
| MSCN (baseline) | 2.39× | 92.23× | **5,478×** | 21,306,067× | 43.5% |
| Fusion MLP | 1.72× | 23.10× | **6,034×** | 21,440,671× | 56.2% |
| TreeLSTM + MLP | 1.34× | 12.20× | **393×** | 1,027,752× | 72.8% |

Matched: TreeLSTM + MLP vs Fusion MLP: -0.376 [-0.448, -0.311] (TreeLSTM + MLP better); TreeLSTM + MLP vs FactorJoin (Chow-Liu BN): -0.665 [-0.754, -0.580] (TreeLSTM + MLP better); Fusion MLP vs FactorJoin (Chow-Liu BN): -0.289 [-0.359, -0.221] (Fusion MLP better); TreeLSTM + MLP vs MSCN (baseline): -0.778 [-0.877, -0.683] (TreeLSTM + MLP better); TreeLSTM + MLP vs PostgreSQL 16: -1.684 [-1.846, -1.535] (TreeLSTM + MLP better)

**Out of distribution: Unseen cardinality range (both tails held out)** (1,201 queries, models retrained without them)

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 16.52× | 133,643× | **51,899,093×** | 2,900,042,284× | 19.7% |
| PostgreSQL 16 + CREATE STATISTICS | 16.75× | 130,841× | **49,050,214×** | 2,494,739,574× | 19.2% |
| FactorJoin (Chow-Liu BN) | 1.47× | 357× | **133,785×** | 23,710,665× | 59.6% |
| MSCN (baseline) | 158× | 213,970× | **2,231,481×** | 117,532,316× | 5.7% |
| Fusion MLP | 6.27× | 8,411× | **670,023×** | 124,391,052× | 28.8% |
| TreeLSTM + MLP | 2.54× | 79,851× | **2,806,298,872×** | 414,309,687,620,160× | 44.5% |

Matched: TreeLSTM + MLP vs Fusion MLP: -0.315 [-0.536, -0.080] (TreeLSTM + MLP better); TreeLSTM + MLP vs FactorJoin (Chow-Liu BN): +1.185 [+0.970, +1.408] (FactorJoin (Chow-Liu BN) better); Fusion MLP vs FactorJoin (Chow-Liu BN): +1.500 [+1.361, +1.624] (FactorJoin (Chow-Liu BN) better); TreeLSTM + MLP vs MSCN (baseline): -2.802 [-3.034, -2.568] (TreeLSTM + MLP better); TreeLSTM + MLP vs PostgreSQL 16: -1.289 [-1.566, -0.989] (TreeLSTM + MLP better)

**Never-trained set: empty / tiny results (0-10 rows)** (300 queries)

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 85.00× | 5,212,965× | **425,772,803×** | 470,404,767,714× | 22.3% |
| PostgreSQL 16 + CREATE STATISTICS | 82.50× | 4,556,634× | **362,763,720×** | 411,804,143,814× | 23.0% |
| FactorJoin (Chow-Liu BN) | 2.03× | 33,232× | **37,498,742×** | 141,788,172× | 50.0% |
| MSCN (baseline) | 2.72× | 39,976× | **37,498,742×** | 141,788,172× | 47.3% |
| Fusion MLP | 1.00× | 4,944× | **243,133×** | 80,805,359× | 53.7% |
| TreeLSTM + MLP | 1.00× | 14,598× | **2,213,869×** | 141,788,172× | 66.0% |

**Never-trained set: rare or absent literal values** (300 queries)

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 1.00× | 1,192× | **66,307×** | 5,759,099× | 59.7% |
| PostgreSQL 16 + CREATE STATISTICS | 1.00× | 1,336× | **60,697×** | 6,290,304× | 60.7% |
| FactorJoin (Chow-Liu BN) | 1.00× | 26.69× | **3,477×** | 74,096× | 85.7% |
| MSCN (baseline) | 1.00× | 26.60× | **6,696×** | 74,096× | 77.0% |
| Fusion MLP | 1.00× | 17.77× | **264×** | 8,026× | 87.7% |
| TreeLSTM + MLP | 1.00× | 15.88× | **3,336×** | 74,096× | 89.3% |

**Matched comparisons** (paired bootstrap 95% CI of the difference in mean log Q-error; negative = first is better; CI containing 0 = inconclusive):

| Queries | Pair | Δ mean log-q [95% CI] | verdict |
|---|---|---|---|
| overall | TreeLSTM + MLP vs Fusion MLP | -0.233 [-0.275, -0.192] | TreeLSTM + MLP better |
| overall | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.454 [-0.510, -0.398] | TreeLSTM + MLP better |
| overall | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.221 [-0.274, -0.170] | Fusion MLP better |
| overall | TreeLSTM + MLP vs MSCN (baseline) | -0.532 [-0.592, -0.475] | TreeLSTM + MLP better |
| overall | TreeLSTM + MLP vs PostgreSQL 16 | -1.157 [-1.275, -1.033] | TreeLSTM + MLP better |
| tables: 1 table | TreeLSTM + MLP vs Fusion MLP | -0.014 [-0.018, -0.009] | TreeLSTM + MLP better |
| tables: 1 table | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | +0.015 [+0.008, +0.023] | FactorJoin (Chow-Liu BN) better |
| tables: 1 table | Fusion MLP vs FactorJoin (Chow-Liu BN) | +0.029 [+0.023, +0.035] | FactorJoin (Chow-Liu BN) better |
| tables: 2 tables | TreeLSTM + MLP vs Fusion MLP | -0.148 [-0.189, -0.110] | TreeLSTM + MLP better |
| tables: 2 tables | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.334 [-0.404, -0.275] | TreeLSTM + MLP better |
| tables: 2 tables | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.187 [-0.239, -0.138] | Fusion MLP better |
| tables: 3 tables | TreeLSTM + MLP vs Fusion MLP | -0.309 [-0.395, -0.226] | TreeLSTM + MLP better |
| tables: 3 tables | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.601 [-0.719, -0.491] | TreeLSTM + MLP better |
| tables: 3 tables | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.293 [-0.393, -0.199] | Fusion MLP better |
| tables: 4 tables | TreeLSTM + MLP vs Fusion MLP | -0.380 [-0.539, -0.236] | TreeLSTM + MLP better |
| tables: 4 tables | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.759 [-0.948, -0.595] | TreeLSTM + MLP better |
| tables: 4 tables | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.379 [-0.544, -0.211] | Fusion MLP better |
| tables: 5 tables | TreeLSTM + MLP vs Fusion MLP | -0.666 [-1.007, -0.331] | TreeLSTM + MLP better |
| tables: 5 tables | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -1.171 [-1.612, -0.791] | TreeLSTM + MLP better |
| tables: 5 tables | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.505 [-0.996, -0.037] | Fusion MLP better |
| join shape: branching | TreeLSTM + MLP vs Fusion MLP | -0.700 [-1.120, -0.268] | TreeLSTM + MLP better |
| join shape: branching | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -1.144 [-1.732, -0.616] | TreeLSTM + MLP better |
| join shape: branching | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.444 [-1.130, +0.195] | inconclusive |
| join shape: chain | TreeLSTM + MLP vs Fusion MLP | -0.255 [-0.307, -0.202] | TreeLSTM + MLP better |
| join shape: chain | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.509 [-0.574, -0.449] | TreeLSTM + MLP better |
| join shape: chain | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.254 [-0.311, -0.199] | Fusion MLP better |
| join shape: single table | TreeLSTM + MLP vs Fusion MLP | -0.014 [-0.018, -0.009] | TreeLSTM + MLP better |
| join shape: single table | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | +0.015 [+0.008, +0.023] | FactorJoin (Chow-Liu BN) better |
| join shape: single table | Fusion MLP vs FactorJoin (Chow-Liu BN) | +0.029 [+0.023, +0.035] | FactorJoin (Chow-Liu BN) better |
| join shape: star | TreeLSTM + MLP vs Fusion MLP | -0.348 [-0.582, -0.142] | TreeLSTM + MLP better |
| join shape: star | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.827 [-1.154, -0.518] | TreeLSTM + MLP better |
| join shape: star | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.479 [-0.736, -0.233] | Fusion MLP better |
| predicates: mixed | TreeLSTM + MLP vs Fusion MLP | -0.175 [-0.287, -0.070] | TreeLSTM + MLP better |
| predicates: mixed | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.510 [-0.647, -0.369] | TreeLSTM + MLP better |
| predicates: mixed | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.335 [-0.487, -0.186] | Fusion MLP better |
| predicates: range only | TreeLSTM + MLP vs Fusion MLP | -0.239 [-0.283, -0.194] | TreeLSTM + MLP better |
| predicates: range only | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.446 [-0.507, -0.391] | TreeLSTM + MLP better |
| predicates: range only | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.207 [-0.265, -0.153] | Fusion MLP better |
| true rows: 0 rows | TreeLSTM + MLP vs Fusion MLP | -0.692 [-1.260, -0.119] | TreeLSTM + MLP better |
| true rows: 0 rows | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -1.923 [-2.766, -1.098] | TreeLSTM + MLP better |
| true rows: 0 rows | Fusion MLP vs FactorJoin (Chow-Liu BN) | -1.231 [-1.972, -0.616] | Fusion MLP better |
| true rows: 11-1k | TreeLSTM + MLP vs Fusion MLP | -0.292 [-0.423, -0.189] | TreeLSTM + MLP better |
| true rows: 11-1k | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.434 [-0.570, -0.309] | TreeLSTM + MLP better |
| true rows: 11-1k | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.142 [-0.276, +0.008] | inconclusive |
| true rows: 1k-1M | TreeLSTM + MLP vs Fusion MLP | -0.172 [-0.218, -0.130] | TreeLSTM + MLP better |
| true rows: 1k-1M | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.405 [-0.463, -0.349] | TreeLSTM + MLP better |
| true rows: 1k-1M | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.232 [-0.289, -0.182] | Fusion MLP better |
| true rows: >1M | TreeLSTM + MLP vs Fusion MLP | -0.271 [-0.371, -0.188] | TreeLSTM + MLP better |
| true rows: >1M | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.397 [-0.490, -0.311] | TreeLSTM + MLP better |
| true rows: >1M | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.125 [-0.229, -0.015] | Fusion MLP better |

Per-seed agreement (share of seed pairs in which the first model wins): TreeLSTM + MLP vs Fusion MLP 100%, TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) 100%, Fusion MLP vs FactorJoin (Chow-Liu BN) 100%, TreeLSTM + MLP vs MSCN (baseline) 100%, TreeLSTM + MLP vs PostgreSQL 16 100%

**Where errors come from** (p95 Q-error per slice of the test set):

| tables | n | PostgreSQL 16 | PostgreSQL 16 + CREATE STATISTICS | FactorJoin (Chow-Liu BN) | MSCN (baseline) | Fusion MLP | TreeLSTM + MLP |
|---|---|---|---|---|---|---|---|
| 1 table | 238 | 1.55× | 1.46× | 1.21× | 2.46× | 1.20× | 1.20× |
| 2 tables | 399 | 23.91× | 29.52× | 5.59× | 8.47× | 5.61× | 2.32× |
| 3 tables | 307 | 165× | 169× | 24.22× | 25.00× | 13.99× | 6.00× |
| 4 tables | 169 | 6,096× | 5,906× | 61.45× | 104× | 45.88× | 14.47× |
| 5 tables | 88 | 138,012× | 135,999× | 1,324× | 3,505× | 153× | 103× |

| join shape | n | PostgreSQL 16 | PostgreSQL 16 + CREATE STATISTICS | FactorJoin (Chow-Liu BN) | MSCN (baseline) | Fusion MLP | TreeLSTM + MLP |
|---|---|---|---|---|---|---|---|
| branching | 52 | 127,381× | 127,631× | 627× | 946× | 62.64× | 46.54× |
| chain | 828 | 243× | 229× | 29.67× | 31.57× | 13.97× | 6.68× |
| single table | 238 | 1.55× | 1.46× | 1.21× | 2.46× | 1.20× | 1.20× |
| star | 83 | 17,184× | 15,516× | 146× | 146× | 35.97× | 10.38× |

| predicates | n | PostgreSQL 16 | PostgreSQL 16 + CREATE STATISTICS | FactorJoin (Chow-Liu BN) | MSCN (baseline) | Fusion MLP | TreeLSTM + MLP |
|---|---|---|---|---|---|---|---|
| eq only | 12 | 772× | 772× | 742× | 709× | 1,040× | 398× |
| mixed | 157 | 150× | 153× | 21.95× | 30.43× | 14.48× | 7.08× |
| no filter | 1 | 3.44× | 3.44× | 3.17× | 3.17× | 4.40× | 2.12× |
| range only | 1,031 | 438× | 459× | 31.37× | 33.46× | 14.47× | 5.88× |

| true rows | n | PostgreSQL 16 | PostgreSQL 16 + CREATE STATISTICS | FactorJoin (Chow-Liu BN) | MSCN (baseline) | Fusion MLP | TreeLSTM + MLP |
|---|---|---|---|---|---|---|---|
| 0 rows | 41 | 158,794× | 151,986× | 6,213× | 6,213× | 549× | 3,763× |
| 1-10 | 12 | 407× | 401× | 585× | 585× | 737× | 105× |
| 11-1k | 153 | 60.41× | 56.60× | 24.85× | 36.19× | 10.67× | 7.25× |
| 1k-1M | 694 | 61.79× | 60.81× | 18.19× | 15.17× | 8.75× | 3.98× |
| >1M | 301 | 4,057× | 3,896× | 25.24× | 36.73× | 12.92× | 7.08× |

**Uncertainty calibration on test** (Spearman of seed spread vs error; p95 Q-error on the most-confident 80% / all):

- MSCN (baseline): Spearman 0.07, p95 24.90× / 33.74×, fallback fired on 51.1%
- Fusion MLP: Spearman 0.66, p95 4.71× / 14.88×, fallback fired on 0.0%
- TreeLSTM + MLP: Spearman 0.68, p95 2.33× / 6.67×, fallback fired on 1.5%

**Data drift** (models built on the 2013-06 snapshot, about half of today's rows):

| Variant | Estimator | median | p95 | p99 |
|---|---|---|---|---|
| stale (BN + NN from 2013) | FactorJoin (Chow-Liu BN) | 9.42× | 30,502,414× | 22,506,455,809× |
| stale (BN + NN from 2013) | MSCN (baseline) | 9.18× | 23,838,224× | 15,563,350,142× |
| stale (BN + NN from 2013) | Fusion MLP | 5.91× | 30,502,414× | 22,506,455,809× |
| stale (BN + NN from 2013) | TreeLSTM + MLP | 4.10× | 30,502,414× | 22,506,455,809× |
| BN refreshed, NN stale | FactorJoin (Chow-Liu BN) | 1.41× | 30.20× | 955× |
| BN refreshed, NN stale | MSCN (baseline) | 2.35× | 118× | 2,914× |
| BN refreshed, NN stale | Fusion MLP | 1.28× | 16.06× | 350× |
| BN refreshed, NN stale | TreeLSTM + MLP | 1.15× | 6.49× | 71.13× |
| fully retrained | PostgreSQL 16 | 1.85× | 354× | 28,582× |
| fully retrained | FactorJoin (Chow-Liu BN) | 1.41× | 30.20× | 955× |
| fully retrained | MSCN (baseline) | 1.62× | 33.74× | 955× |
| fully retrained | Fusion MLP | 1.23× | 14.88× | 238× |
| fully retrained | TreeLSTM + MLP | 1.11× | 6.67× | 111× |

**Join-order quality** (DP optimizer on each estimator's sub-plan estimates, scored with true C_out):

| Estimator | optimal plan | mean cost / optimal | worst |
|---|---|---|---|
| PostgreSQL 16 | 80.7% | 2.968× | 596× |
| FactorJoin (Chow-Liu BN) | 88.1% | 2.284× | 596× |
| MSCN (baseline) | 82.3% | 2.662× | 596× |
| Fusion MLP | 89.4% | 2.278× | 596× |
| TreeLSTM + MLP | 91.7% | 2.026× | 596× |

**Robustness** (guards and property checks):

| Estimator | uncertainty fallback | clamped | order-invariant | monotone | latency p50 / p99 |
|---|---|---|---|---|---|
| PostgreSQL 16 | 0.0% | 0.0% | 61% | 100% | 2.5 / 5.8 ms |
| FactorJoin (Chow-Liu BN) | 0.0% | 0.1% | 100% | 100% | 0.2 / 0.4 ms |
| MSCN (baseline) | 51.1% | 2.5% | 100% | 85% | 5.1 / 8.3 ms |
| Fusion MLP | 0.0% | 1.0% | 100% | 86% | 2.8 / 8.6 ms |
| TreeLSTM + MLP | 1.5% | 0.1% | 100% | 79% | 5.1 / 13.1 ms |
| exact execution | | | | | 17.6 / 87.5 ms |


### End-to-end runtime on PostgreSQL 16 (STATS-CEB)

Each estimator's sub-plan cardinalities injected with pg_hint_plan, all 146 STATS-CEB queries executed, timeout 30 s (timeouts charged in full).

| Configuration | execution | + estimation | net vs PostgreSQL | timeouts | faster / slower queries (>10%) |
|---|---|---|---|---|---|
| PostgreSQL 16 (own estimates) | 593 s | – | 1.000× | 11 | 0 / 0 |
| PostgreSQL 16 + CREATE STATISTICS | 614 s | – | 0.965× | 11 | 24 / 34 |
| True cardinalities (oracle) | 569 s | n/a | 1.041× | 11 | 72 / 21 |
| FactorJoin (Chow-Liu BN) | 601 s | 2.9 s | 0.981× | 11 | 42 / 41 |
| MSCN (baseline) | 600 s | 31.3 s | 0.938× | 10 | 46 / 45 |
| Fusion MLP | 647 s | 8.6 s | 0.904× | 13 | 42 / 66 |
| TreeLSTM + MLP | 621 s | 10.0 s | 0.939× | 12 | 37 / 80 |

Estimation = time to estimate all 1566 sub-plans (one batched call per query, cold caches) after the latency optimisations; PostgreSQL's own planning time is excluded from every row alike. The oracle's true counts cannot be produced at planning time, so it is an upper bound.


### HR (synthetic, planted correlations)

1,202 held-out test queries (5,899 train / 1,199 val). Produced by commit `0466a0b5a6` (dirty tree), data `9ef922b0f261a21c`, workload `66a3329465a834d8`.

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 1.04× | 5.22× | **1,540×** (79.07×–13,978×) | 2,086,786× | 89.0% |
| PostgreSQL 16 + CREATE STATISTICS | 1.04× | 4.83× | **1,540×** (81.49×–14,502×) | 3,431,292× | 90.3% |
| FactorJoin (Chow-Liu BN) | 1.01× | 1.92× | **14.09×** (7.36×–87.65×) | 3,446× | 95.0% |
| MSCN (baseline) | 1.06× | 2.67× | **28.67×** (10.14×–361×) | 490,925× | 92.5% |
| Fusion MLP | 1.01× | 1.85× | **7.13×** (5.87×–14.40×) | 1,291× | 95.5% |
| TreeLSTM + MLP | 1.02× | 1.79× | **7.22×** (5.67×–23.83×) | 977× | 95.7% |

Paired Wilcoxon signed-rank tests on per-query log Q-error:

| A vs B | mean log-Q A / B | A closer / B closer | p-value |
|---|---|---|---|
| TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | 0.140 / 0.152 | 27% / 72% | 1.5e-32 |
| Fusion MLP vs FactorJoin (Chow-Liu BN) | 0.133 / 0.152 | 29% / 69% | 3.4e-21 |
| TreeLSTM + MLP vs Fusion MLP | 0.140 / 0.133 | 36% / 62% | 5.2e-20 |
| FactorJoin (Chow-Liu BN) vs PostgreSQL 16 | 0.152 / 0.374 | 62% / 37% | 3e-60 |
| FactorJoin (Chow-Liu BN) vs PostgreSQL 16 + CREATE STATISTICS | 0.152 / 0.353 | 61% / 38% | 2.1e-54 |

Seed spread (each learned model trained with independent seeds):

| Model | seeds | median | p95 | p99 |
|---|---|---|---|---|
| MSCN (baseline) | 5 | 1.291 ± 0.051 | 5.51 ± 0.67 | 519.5 ± 178.7 |
| Fusion MLP | 5 | 1.011 ± 0.005 | 1.86 ± 0.05 | 8.0 ± 1.0 |
| TreeLSTM + MLP | 5 | 1.029 ± 0.003 | 1.82 ± 0.07 | 8.0 ± 1.2 |

**Out of distribution: Unseen query templates** (1,251 queries, models retrained without them)

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 1.05× | 4.10× | **22,566×** | 4,269,361× | 87.2% |
| PostgreSQL 16 + CREATE STATISTICS | 1.04× | 3.82× | **18,284×** | 4,387,676× | 88.9% |
| FactorJoin (Chow-Liu BN) | 1.01× | 1.91× | **50.92×** | 11,432× | 95.2% |
| MSCN (baseline) | 1.07× | 2.50× | **82.28×** | 53,606× | 92.4% |
| Fusion MLP | 1.01× | 1.66× | **4.05×** | 1,104× | 96.0% |
| TreeLSTM + MLP | 1.02× | 1.66× | **3.86×** | 918× | 96.2% |

Matched: TreeLSTM + MLP vs Fusion MLP: +0.005 [-0.002, +0.012] (inconclusive); TreeLSTM + MLP vs FactorJoin (Chow-Liu BN): -0.041 [-0.072, -0.016] (TreeLSTM + MLP better); Fusion MLP vs FactorJoin (Chow-Liu BN): -0.046 [-0.077, -0.021] (Fusion MLP better); TreeLSTM + MLP vs MSCN (baseline): -0.155 [-0.206, -0.114] (TreeLSTM + MLP better); TreeLSTM + MLP vs PostgreSQL 16: -0.328 [-0.407, -0.257] (TreeLSTM + MLP better)

**Out of distribution: Unseen join sizes (train 1–3 tables, test 4–5)** (1,950 queries, models retrained without them)

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 1.17× | 8.97× | **175,300×** | 4,100,576× | 80.5% |
| PostgreSQL 16 + CREATE STATISTICS | 1.14× | 8.09× | **175,668×** | 4,180,544× | 82.1% |
| FactorJoin (Chow-Liu BN) | 1.04× | 3.25× | **201×** | 24,690× | 90.8% |
| MSCN (baseline) | 1.06× | 8.55× | **446×** | 212,832× | 85.4% |
| Fusion MLP | 1.04× | 3.00× | **10.12×** | 15,443× | 91.3% |
| TreeLSTM + MLP | 1.05× | 2.92× | **9.37×** | 14,973× | 91.5% |

Matched: TreeLSTM + MLP vs Fusion MLP: -0.005 [-0.011, +0.001] (inconclusive); TreeLSTM + MLP vs FactorJoin (Chow-Liu BN): -0.049 [-0.072, -0.030] (TreeLSTM + MLP better); Fusion MLP vs FactorJoin (Chow-Liu BN): -0.044 [-0.065, -0.028] (Fusion MLP better); TreeLSTM + MLP vs MSCN (baseline): -0.183 [-0.220, -0.149] (TreeLSTM + MLP better); TreeLSTM + MLP vs PostgreSQL 16: -0.466 [-0.541, -0.394] (TreeLSTM + MLP better)

**Out of distribution: Unseen join graphs** (2,233 queries, models retrained without them)

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 1.02× | 3.26× | **49.08×** | 456,503× | 90.4% |
| PostgreSQL 16 + CREATE STATISTICS | 1.02× | 3.00× | **49.08×** | 336,751× | 91.3% |
| FactorJoin (Chow-Liu BN) | 1.01× | 2.23× | **8.43×** | 1,368× | 94.0% |
| MSCN (baseline) | 1.03× | 2.88× | **18.05×** | 29,893× | 85.4% |
| Fusion MLP | 1.01× | 2.16× | **7.82×** | 361× | 94.2% |
| TreeLSTM + MLP | 1.03× | 2.16× | **7.54×** | 347× | 94.4% |

Matched: TreeLSTM + MLP vs Fusion MLP: +0.012 [+0.009, +0.014] (Fusion MLP better); TreeLSTM + MLP vs FactorJoin (Chow-Liu BN): +0.004 [-0.005, +0.011] (inconclusive); Fusion MLP vs FactorJoin (Chow-Liu BN): -0.008 [-0.017, -0.001] (Fusion MLP better); TreeLSTM + MLP vs MSCN (baseline): -0.121 [-0.141, -0.103] (TreeLSTM + MLP better); TreeLSTM + MLP vs PostgreSQL 16: -0.111 [-0.143, -0.082] (TreeLSTM + MLP better)

**Out of distribution: Unseen cardinality range (both tails held out)** (1,202 queries, models retrained without them)

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 1.13× | 2,933× | **456,215×** | 4,269,361× | 75.0% |
| PostgreSQL 16 + CREATE STATISTICS | 1.07× | 3,112× | **396,553×** | 4,387,676× | 76.6% |
| FactorJoin (Chow-Liu BN) | 1.01× | 13.11× | **623×** | 24,690× | 87.6% |
| MSCN (baseline) | 1.01× | 63.23× | **15,217×** | 329,044× | 80.2% |
| Fusion MLP | 1.01× | 13.68× | **621×** | 24,729× | 87.8% |
| TreeLSTM + MLP | 1.01× | 32.73× | **5,000×** | 78,300,000× | 86.5% |

Matched: TreeLSTM + MLP vs Fusion MLP: +0.115 [+0.051, +0.184] (Fusion MLP better); TreeLSTM + MLP vs FactorJoin (Chow-Liu BN): +0.120 [+0.056, +0.190] (FactorJoin (Chow-Liu BN) better); Fusion MLP vs FactorJoin (Chow-Liu BN): +0.005 [+0.003, +0.007] (FactorJoin (Chow-Liu BN) better); TreeLSTM + MLP vs MSCN (baseline): -0.112 [-0.201, -0.025] (TreeLSTM + MLP better); TreeLSTM + MLP vs PostgreSQL 16: -0.727 [-0.872, -0.599] (TreeLSTM + MLP better)

**Never-trained set: empty / tiny results (0-10 rows)** (300 queries)

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 537× | 615,391× | **2,155,856×** | 12,199,430× | 21.7% |
| PostgreSQL 16 + CREATE STATISTICS | 537× | 584,381× | **2,154,025×** | 12,199,430× | 21.7% |
| FactorJoin (Chow-Liu BN) | 1.00× | 3,973× | **66,989×** | 1,278,881× | 57.3% |
| MSCN (baseline) | 1.00× | 13,082× | **687,644×** | 1,521,618× | 68.0% |
| Fusion MLP | 1.00× | 695× | **66,607×** | 1,271,586× | 80.0% |
| TreeLSTM + MLP | 1.00× | 660× | **67,772×** | 1,235,134× | 82.0% |

**Never-trained set: rare or absent literal values** (300 queries)

| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |
|---|---|---|---|---|---|
| PostgreSQL 16 | 1.16× | 27,066× | **280,237×** | 11,419,380× | 61.3% |
| PostgreSQL 16 + CREATE STATISTICS | 1.15× | 26,208× | **183,041×** | 11,419,380× | 63.0% |
| FactorJoin (Chow-Liu BN) | 1.00× | 6.31× | **984×** | 11,444× | 93.0% |
| MSCN (baseline) | 1.00× | 12.80× | **1,859×** | 35,190× | 92.7% |
| Fusion MLP | 1.00× | 1.64× | **53.76×** | 1,322× | 96.0% |
| TreeLSTM + MLP | 1.00× | 1.80× | **39.30×** | 1,122× | 96.0% |

**Matched comparisons** (paired bootstrap 95% CI of the difference in mean log Q-error; negative = first is better; CI containing 0 = inconclusive):

| Queries | Pair | Δ mean log-q [95% CI] | verdict |
|---|---|---|---|
| overall | TreeLSTM + MLP vs Fusion MLP | +0.007 [+0.001, +0.015] | Fusion MLP better |
| overall | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.012 [-0.042, +0.015] | inconclusive |
| overall | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.019 [-0.048, +0.008] | inconclusive |
| overall | TreeLSTM + MLP vs MSCN (baseline) | -0.109 [-0.147, -0.076] | TreeLSTM + MLP better |
| overall | TreeLSTM + MLP vs PostgreSQL 16 | -0.234 [-0.301, -0.173] | TreeLSTM + MLP better |
| tables: 1 table | TreeLSTM + MLP vs Fusion MLP | +0.006 [+0.003, +0.008] | Fusion MLP better |
| tables: 1 table | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | +0.010 [+0.007, +0.013] | FactorJoin (Chow-Liu BN) better |
| tables: 1 table | Fusion MLP vs FactorJoin (Chow-Liu BN) | +0.005 [+0.004, +0.005] | FactorJoin (Chow-Liu BN) better |
| tables: 2 tables | TreeLSTM + MLP vs Fusion MLP | +0.009 [+0.004, +0.014] | Fusion MLP better |
| tables: 2 tables | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | +0.014 [+0.006, +0.024] | FactorJoin (Chow-Liu BN) better |
| tables: 2 tables | Fusion MLP vs FactorJoin (Chow-Liu BN) | +0.006 [+0.000, +0.012] | FactorJoin (Chow-Liu BN) better |
| tables: 3 tables | TreeLSTM + MLP vs Fusion MLP | -0.001 [-0.010, +0.007] | inconclusive |
| tables: 3 tables | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.017 [-0.044, +0.007] | inconclusive |
| tables: 3 tables | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.015 [-0.040, +0.004] | inconclusive |
| tables: 4 tables | TreeLSTM + MLP vs Fusion MLP | -0.001 [-0.014, +0.009] | inconclusive |
| tables: 4 tables | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.086 [-0.235, +0.043] | inconclusive |
| tables: 4 tables | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.085 [-0.232, +0.044] | inconclusive |
| tables: 5 tables | TreeLSTM + MLP vs Fusion MLP | +0.035 [-0.014, +0.114] | inconclusive |
| tables: 5 tables | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.016 [-0.196, +0.161] | inconclusive |
| tables: 5 tables | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.051 [-0.226, +0.112] | inconclusive |
| join shape: branching | TreeLSTM + MLP vs Fusion MLP | +0.070 [+0.001, +0.195] | Fusion MLP better |
| join shape: branching | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.074 [-0.368, +0.171] | inconclusive |
| join shape: branching | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.144 [-0.417, +0.043] | inconclusive |
| join shape: chain | TreeLSTM + MLP vs Fusion MLP | +0.004 [-0.001, +0.008] | inconclusive |
| join shape: chain | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.004 [-0.025, +0.018] | inconclusive |
| join shape: chain | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.008 [-0.029, +0.015] | inconclusive |
| join shape: single table | TreeLSTM + MLP vs Fusion MLP | +0.006 [+0.003, +0.008] | Fusion MLP better |
| join shape: single table | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | +0.010 [+0.007, +0.013] | FactorJoin (Chow-Liu BN) better |
| join shape: single table | Fusion MLP vs FactorJoin (Chow-Liu BN) | +0.005 [+0.004, +0.005] | FactorJoin (Chow-Liu BN) better |
| join shape: star | TreeLSTM + MLP vs Fusion MLP | -0.014 [-0.038, +0.005] | inconclusive |
| join shape: star | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.086 [-0.339, +0.124] | inconclusive |
| join shape: star | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.071 [-0.320, +0.131] | inconclusive |
| predicates: eq only | TreeLSTM + MLP vs Fusion MLP | +0.009 [+0.006, +0.011] | Fusion MLP better |
| predicates: eq only | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.036 [-0.122, +0.014] | inconclusive |
| predicates: eq only | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.045 [-0.130, +0.004] | inconclusive |
| predicates: mixed | TreeLSTM + MLP vs Fusion MLP | +0.005 [-0.007, +0.021] | inconclusive |
| predicates: mixed | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.017 [-0.065, +0.029] | inconclusive |
| predicates: mixed | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.022 [-0.069, +0.022] | inconclusive |
| predicates: range only | TreeLSTM + MLP vs Fusion MLP | +0.010 [+0.006, +0.014] | Fusion MLP better |
| predicates: range only | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | +0.015 [+0.008, +0.023] | FactorJoin (Chow-Liu BN) better |
| predicates: range only | Fusion MLP vs FactorJoin (Chow-Liu BN) | +0.005 [+0.000, +0.010] | FactorJoin (Chow-Liu BN) better |
| true rows: 1-10 | TreeLSTM + MLP vs Fusion MLP | -0.008 [-0.038, +0.019] | inconclusive |
| true rows: 1-10 | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | -0.015 [-0.064, +0.029] | inconclusive |
| true rows: 1-10 | Fusion MLP vs FactorJoin (Chow-Liu BN) | -0.007 [-0.039, +0.016] | inconclusive |
| true rows: 11-1k | TreeLSTM + MLP vs Fusion MLP | +0.012 [+0.006, +0.018] | Fusion MLP better |
| true rows: 11-1k | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | +0.018 [+0.010, +0.027] | FactorJoin (Chow-Liu BN) better |
| true rows: 11-1k | Fusion MLP vs FactorJoin (Chow-Liu BN) | +0.006 [+0.002, +0.012] | FactorJoin (Chow-Liu BN) better |
| true rows: 1k-1M | TreeLSTM + MLP vs Fusion MLP | +0.010 [+0.000, +0.027] | Fusion MLP better |
| true rows: 1k-1M | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | +0.038 [+0.011, +0.071] | FactorJoin (Chow-Liu BN) better |
| true rows: 1k-1M | Fusion MLP vs FactorJoin (Chow-Liu BN) | +0.028 [+0.004, +0.061] | FactorJoin (Chow-Liu BN) better |
| true rows: >1M | TreeLSTM + MLP vs Fusion MLP | +0.008 [+0.004, +0.012] | Fusion MLP better |
| true rows: >1M | TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) | +0.010 [+0.006, +0.015] | FactorJoin (Chow-Liu BN) better |
| true rows: >1M | Fusion MLP vs FactorJoin (Chow-Liu BN) | +0.003 [+0.002, +0.003] | FactorJoin (Chow-Liu BN) better |

Per-seed agreement (share of seed pairs in which the first model wins): TreeLSTM + MLP vs Fusion MLP 20%, TreeLSTM + MLP vs FactorJoin (Chow-Liu BN) 80%, Fusion MLP vs FactorJoin (Chow-Liu BN) 80%, TreeLSTM + MLP vs MSCN (baseline) 100%, TreeLSTM + MLP vs PostgreSQL 16 100%

**Where errors come from** (p95 Q-error per slice of the test set):

| tables | n | PostgreSQL 16 | PostgreSQL 16 + CREATE STATISTICS | FactorJoin (Chow-Liu BN) | MSCN (baseline) | Fusion MLP | TreeLSTM + MLP |
|---|---|---|---|---|---|---|---|
| 1 table | 206 | 1.26× | 1.14× | 1.02× | 1.17× | 1.02× | 1.05× |
| 2 tables | 384 | 3.36× | 3.37× | 1.86× | 2.59× | 1.87× | 2.02× |
| 3 tables | 319 | 5.09× | 3.85× | 1.59× | 2.20× | 1.60× | 1.50× |
| 4 tables | 174 | 7.07× | 7.05× | 3.60× | 4.19× | 1.96× | 1.89× |
| 5 tables | 119 | 53.30× | 53.47× | 5.43× | 5.84× | 5.23× | 5.25× |

| join shape | n | PostgreSQL 16 | PostgreSQL 16 + CREATE STATISTICS | FactorJoin (Chow-Liu BN) | MSCN (baseline) | Fusion MLP | TreeLSTM + MLP |
|---|---|---|---|---|---|---|---|
| branching | 72 | 365× | 342× | 6.45× | 43.12× | 5.55× | 6.08× |
| chain | 828 | 5.09× | 3.94× | 1.87× | 2.43× | 1.85× | 1.77× |
| single table | 206 | 1.26× | 1.14× | 1.02× | 1.17× | 1.02× | 1.05× |
| star | 96 | 38.19× | 38.54× | 5.27× | 5.27× | 2.67× | 2.49× |

| predicates | n | PostgreSQL 16 | PostgreSQL 16 + CREATE STATISTICS | FactorJoin (Chow-Liu BN) | MSCN (baseline) | Fusion MLP | TreeLSTM + MLP |
|---|---|---|---|---|---|---|---|
| eq only | 224 | 2.29× | 1.92× | 1.05× | 1.70× | 1.04× | 1.05× |
| mixed | 643 | 8.97× | 8.26× | 4.59× | 4.66× | 3.04× | 3.10× |
| not-equal only | 18 | 1.08× | 1.08× | 1.00× | 1.13× | 1.01× | 1.01× |
| range only | 317 | 1.31× | 1.31× | 1.17× | 1.29× | 1.18× | 1.23× |

| true rows | n | PostgreSQL 16 | PostgreSQL 16 + CREATE STATISTICS | FactorJoin (Chow-Liu BN) | MSCN (baseline) | Fusion MLP | TreeLSTM + MLP |
|---|---|---|---|---|---|---|---|
| 0 rows | 28 | 380,851× | 345,217× | 578× | 2,449× | 218× | 193× |
| 1-10 | 33 | 8.63× | 8.74× | 5.75× | 5.75× | 6.28× | 5.80× |
| 11-1k | 407 | 3.14× | 2.69× | 2.07× | 2.21× | 2.06× | 2.03× |
| 1k-1M | 617 | 2.73× | 2.51× | 1.43× | 2.19× | 1.52× | 1.43× |
| >1M | 117 | 1.92× | 1.78× | 1.10× | 2.70× | 1.10× | 1.09× |

**Uncertainty calibration on test** (Spearman of seed spread vs error; p95 Q-error on the most-confident 80% / all):

- MSCN (baseline): Spearman -0.21, p95 2.42× / 2.67×, fallback fired on 48.9%
- Fusion MLP: Spearman 0.25, p95 1.38× / 1.85×, fallback fired on 0.0%
- TreeLSTM + MLP: Spearman 0.27, p95 1.36× / 1.79×, fallback fired on 0.0%

**Join-order quality** (DP optimizer on each estimator's sub-plan estimates, scored with true C_out):

| Estimator | optimal plan | mean cost / optimal | worst |
|---|---|---|---|
| PostgreSQL 16 | 94.1% | 7.551× | 2,506× |
| FactorJoin (Chow-Liu BN) | 95.6% | 0.980× | 3.54× |
| MSCN (baseline) | 93.0% | 3.547× | 1,570× |
| Fusion MLP | 95.1% | 0.981× | 3.54× |
| TreeLSTM + MLP | 95.1% | 0.980× | 3.54× |

**Robustness** (guards and property checks):

| Estimator | uncertainty fallback | clamped | order-invariant | monotone | latency p50 / p99 |
|---|---|---|---|---|---|
| PostgreSQL 16 | 0.0% | 0.0% | 79% | 100% | 5.0 / 9.1 ms |
| FactorJoin (Chow-Liu BN) | 0.0% | 0.0% | 100% | 100% | 0.2 / 0.7 ms |
| MSCN (baseline) | 48.9% | 0.2% | 100% | 95% | 9.5 / 18.9 ms |
| Fusion MLP | 0.0% | 0.0% | 100% | 91% | 4.9 / 15.0 ms |
| TreeLSTM + MLP | 0.0% | 0.0% | 100% | 88% | 13.3 / 37.6 ms |
| exact execution | | | | | 4.0 / 18.3 ms |

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
log FactorJoin. Per-node features include the literal's position in the column's value
distribution (CDF, robust to skew), an unseen-value flag, per-predicate matching rows, and
for joins the max key fan-out, key overlap and heavy-hitter share on each side; the query
vector adds the provable upper bound. Training mixes in ~5% generated empty/tiny and
rare-value queries: at ~27% they hurt the main distribution (HR TreeLSTM val mean
log-q 0.129 → 0.227), at ~5% val is unchanged and rare-value error halves.
Model selection (`scripts/select_model.py`) crosses the top tuned configs with plain and
tail-weighted losses (squared hinge beyond 2×) and picks by validation mean + CVaR95.

**Serving guards (`src/estimator/service.py`).**
1. Each learned model is a multi-seed ensemble (mean of log estimates).
2. If the seeds disagree beyond τ, FactorJoin answers. τ is grid-searched on validation
   (quantiles 0.5–0.99 of the spread) to minimise mean + CVaR95 with the fallback applied,
   and the fallback is enabled only if that beats never falling back.
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
python -m scripts.augment_workload --dataset hr      # split_join / split_sel + tiny/rare sets
python -m scripts.augment_workload --dataset stats

python -m scripts.tune  --dataset stats     # Optuna (TPE + median pruning) per model
python -m scripts.select_model --dataset stats   # tail-aware pick: configs x losses
python -m scripts.train --dataset stats --seeds 5
for s in split_tmpl split_shape split_join split_sel; do
  python -m scripts.train --dataset stats --split $s --seeds 3
done
#   (same for --dataset hr; stats_old trains with the STATS configs for the drift study)

cp .env.example .env                        # then set POSTGRES_PASSWORD / PG_DSN
docker compose up -d                        # PostgreSQL 16 + pg_hint_plan on 127.0.0.1:55432
python -m scripts.init_postgres --dataset hr --yes
python -m scripts.init_postgres --dataset stats --yes
python -m scripts.benchmark --dataset hr
python -m scripts.benchmark --dataset stats
python -m scripts.e2e_runtime               # executes STATS-CEB under every estimator
python -m scripts.report                    # refresh the tables above
pytest -q                                   # unit, property, fuzz, API tests
```

App: `python server.py`, then `cd frontend && npm install && npm run dev`, or open
http://127.0.0.1:8000 after `npm run build`. Set
`PG_DSN` in `.env` (copy `.env.example` and choose a password) for live PostgreSQL rows.

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
* Learned models do not extrapolate in cardinality: on `split_sel` (both tails of the
  true-count range held out) they are worse than FactorJoin (HR TreeLSTM p99 5,000× vs
  623×; STATS TreeLSTM p99 2.8 × 10⁹ vs 1.3 × 10⁵). On the i.i.d. HR test set the TreeLSTM vs FactorJoin and MLP vs FactorJoin
  differences are inconclusive (paired CIs contain 0); the learned models' advantage is
  on unseen templates, unseen join sizes and rare/tiny-result queries.
* The v3 features improved validation and the never-trained sets, but the single worst
  HR test query got worse (TreeLSTM max 563× → 977×). The seeds disagree strongly on such
  queries (top 1% of spread), but on validation the high-spread queries are ones where
  FactorJoin is the one that is wrong, so the calibrated fallback stays off for the fusion
  models. MSCN's spread is anti-correlated with its test error (Spearman −0.21).
* End to end, better cardinalities have not turned into faster STATS-CEB execution: every
  configuration lands within about ±10% of PostgreSQL's own total, the totals are
  dominated by 10–13 queries that hit the 30 s timeout under every estimator, and in the
  latest run the learned models were slightly slower (TreeLSTM 621 s vs 593 s). Even the
  oracle (true counts) only saves ~4%.
* Workloads are generated from the schema's join graph. STATS-CEB is the only
  human-written workload.
