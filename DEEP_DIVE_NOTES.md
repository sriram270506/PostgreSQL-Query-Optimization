# Deep Dive — Completed Sections (Detailed)

This document expands every completed section of the prototype in depth: implementation details, tech stack choices, configuration, commands to reproduce, expected outputs/artifacts, and limitations. Use this for technical slides or a handout for colleagues preparing a deep technical review.

---

## 1) Project scaffold & dev environment

- Purpose: create an isolated, reproducible Python workspace with a minimal, testable project layout so experiments are repeatable and CI-friendly.

- Tech stack used:
  - Python 3.14 (project venv at `./.venv`). Reason: modern Python features, available on the developer machine, and matches the local environment where tests were run.
  - Virtual environment: built-in `venv` used to isolate dependencies; keeps the system Python clean and allows reproducible dependency installs.
  - Core libraries: `pandas`, `numpy` for data manipulation; `scikit-learn` for quick prototyping of fusion baselines; `joblib` for model serialization; `psycopg2-binary` for Postgres integration; `Faker` for random realistic names in synthetic data; `pytest` for unit tests.

- Why these libraries:
  - `pandas`/`numpy`: de-facto standard for tabular data and numerical operations; simple CSV I/O and grouping operations to compute exact counts.
  - `scikit-learn`: quick, production-quality implementations of regressors (MLPRegressor or RandomForest) for baselines without needing heavy DL frameworks.
  - `psycopg2-binary`: stable Postgres client to run `EXPLAIN` and load CSVs into Postgres for planner comparisons.

- Project layout conventions (conceptual):
  - `scripts/` — runnable scripts (data generation, training, comparison harness). Run them with the venv Python and `PYTHONPATH=.` so `src` is importable.
  - `src/` — library code (estimators, feature extractors, BN learners, FactorJoin implementation).
  - `data/` — generated CSVs and query CSVs.
  - `models/` — persisted model artifacts (per-table BN objects, fusion model joblib files).

- Typical commands:
```bash
set PYTHONPATH=.&& .venv\Scripts\python.exe -m pip install -r requirements.txt
set PYTHONPATH=.&& .venv\Scripts\python.exe scripts\train_fusion.py
```

Expected artifacts:
- `models/fusion_model.joblib`
- `models/bn_<table>.joblib`

Limitations / notes:
- This scaffold intentionally favors speed and reproducibility over micro-optimizations. For production integration, packaging, and dependency pinning should be stricter (`pyproject.toml` or pinned `requirements.txt`).

---

## 2) Schema & Synthetic Correlated Data Generator (detailed)

- Purpose: produce a controlled dataset with known cross-table correlations so we can measure estimator performance relative to ground truth.

- What is generated (concrete fields & sizes):
  - Tables: `departments`, `employees`, `projects` (small, focused schema to illustrate cross-table effects).
  - `departments` fields: `dept_id` (int primary key), `dept_name` (text), `city` (text). Default number of departments is fixed (4 in prototype: Eng, Sales, HR, Ops).
  - `employees` fields: `emp_id` (int), `name` (text), `dept_id` (FK to departments), `age` (int), `salary` (int). Default rows: 5000 (configurable via `--rows`).
  - `projects` fields: `project_id` (int), `emp_id` (FK to employees), `budget_tier` (int: 1/2/3). There is one project per employee in prototype (so projects row count equals employees count by default).

- How correlation is injected (concept & code-level):
  - We model a conditional dependency: `department.city` → `project.budget_tier` mediated by `employee` (employee belongs to a department and projects are assigned to employees). The generator chooses budget tier probabilities based on department city.
  - Implementation detail: a mapping `base_probs = { city: [p_low, p_med, p_high] }` is defined; for each employee we mix that city's distribution with a uniform distribution controlled by a `corr` parameter (0..1). Higher `corr` increases alignment with `city`-specific budget distribution.

- Why this design:
  - It creates non-trivial cross-table correlation: the `projects` budget distribution depends on the department city of the employee, so queries that join across these tables violate independence assumptions common in histogram-based planners.
  - The structure is compact yet representative of real-world patterns where entity attributes in one table affect attributes in another via FK relationships.

- Configurations / parameters:
  - `--rows` (default 5000): controls number of employee rows (and projects if one project per employee).
  - `--corr` (default 0.9 in experiments): controls how strongly `budget_tier` depends on department city.
  - `--out-dir` (default `data/out`): where CSVs are written.
  - `--to-db` / `--conn`: option to load CSVs into Postgres using the schema in `data/generate_schema.sql`.

- Where data is stored:
  - CSVs: `data/out/departments.csv`, `data/out/employees.csv`, `data/out/projects.csv`.
  - Database schema for loading: `data/generate_schema.sql` (DROP / CREATE statements). When loaded into Postgres, the tables are named `departments`, `employees`, `projects`.

- Commands to reproduce:
```bash
set PYTHONPATH=.&& .venv\Scripts\python.exe scripts\generate_data.py --rows 5000 --corr 0.9 --out-dir data/out
# to also load into Postgres (example with URL-encoded password if needed)
set PYTHONPATH=.&& .venv\Scripts\python.exe scripts\generate_data.py --rows 5000 --corr 0.9 --out-dir data/out --to-db --conn "postgresql://postgres:MyP%40ss@localhost:5432/postgres"
```

- Expected outputs:
  - CSV files in `data/out/` as described.
  - When `--to-db` used: tables created in Postgres with the schema from `data/generate_schema.sql` and rows loaded via `COPY` (bulk load) operations for speed.

- Practical notes and pitfalls:
  - Passwords with special characters must be URL-encoded when embedded in DSNs (e.g., `@` → `%40`) or passed as separate connection args to avoid parsing errors.
  - The generator uses `Faker` for employee names (which is non-deterministic); set a random seed in code for deterministic reproducibility if desired.

---

## 3) Query Workload Generation & Ground-truth Collection (detailed)

- Purpose: produce a realistic-ish query set covering single-table filters and multi-table joins and compute exact counts to use as labels for training and evaluation.

- Workload composition:
  - Mix includes single-table predicates (e.g., `employees WHERE salary > X`), join queries (e.g., `employees JOIN departments WHERE city = '...'`), and join+filter queries that exercise cross-table correlation.
  - Example query forms: `SELECT COUNT(*) FROM employees WHERE salary > 100000`, `SELECT COUNT(*) FROM employees e JOIN projects p ON e.emp_id=p.emp_id WHERE p.budget_tier = 3`, and `SELECT COUNT(*) FROM employees e JOIN departments d ON e.dept_id=d.dept_id WHERE d.city='Bangalore' AND e.salary>90000`.

- Format and storage of workloads:
  - CSV format (e.g., `data/queries/test_queries_with_counts.csv`) with at least two columns: `sql` and `true_count`.
  - Training and test splits are separate CSVs so the fusion model is trained on one set and evaluated on a held-out test set.

- How ground truth is computed:
  - Exact counts are computed by executing the query against the generated CSVs using `pandas` joins or by running the SQL in Postgres after loading the CSVs. The script `scripts/collect_true_counts.py` (prototype) runs queries against CSV-backed dataframes (or DB) and writes the `*_with_counts.csv` file.

- Typical sizes & expectations:
  - Number of queries in experiments: 500 (balanced across selective and non-selective). This is configurable in the workload generator.
  - `true_count` ranges from very small (0 or single-digits) to large (thousands), which motivates log-target modeling.

- Example command to compute truth (conceptual):
```bash
set PYTHONPATH=.&& .venv\Scripts\python.exe scripts\collect_true_counts.py --queries data/queries/test_queries.csv --out data/queries/test_queries_with_counts.csv
```

---

## 4) Per-Table Bayesian Networks (Chow-Liu + CPTs) — deep detail

- Purpose: capture intra-table dependencies compactly so we can answer marginal and conditional probability queries for predicates on a single table.

- Why Chow-Liu / tree-BN:
  - Exact BN structure learning is combinatorial and expensive for many attributes. Chow-Liu learns the maximum-likelihood tree-structured BN (a maximum-weight spanning tree where weights are empirical mutual information between variables). It balances expressive power and tractability.
  - Tree-BNs are efficient to query (marginalization and single-variable conditionals are straightforward) and require only pairwise statistics.

- Implementation sketch (what we produce):
  - Input: per-table data (CSV), list of discrete columns or discretized continuous fields for BN learning.
  - Compute pairwise empirical mutual information for column pairs (discrete). For continuous attributes we discretize (e.g., salary bins) or use rank-based bucketing.
  - Build maximum spanning tree using Kruskal or Prim on the mutual information graph.
  - Estimate conditional probability tables (CPTs) for each node given its parent by empirical frequency counts with Laplace smoothing to avoid zero probabilities.

- Outputs and storage:
  - Stored BN object per table serialized to disk (`models/bn_<table>.joblib`). Each object exposes an API to compute `P(predicate)` or `P(value | evidence)` for elementary predicates used by queries.

- How these outputs are used downstream:
  - Compute per-table selectivity for query predicates (e.g., P(age>30 AND dept_id=2)). These scalars become features for FactorJoin and the fusion model.

- Expected behavior and caveats:
  - Tree-BNs capture first-order dependencies (via pairwise MI) but not higher-order conditional interactions; still they substantially improve marginal estimates over naive histograms when attribute dependencies are strong.
  - Discretization choices for numeric columns materially affect mutual information estimates. For more accurate BNs one can use adaptive binning or Gaussian/Beta parametric assumptions.

---

## 5) FactorJoin Approximation (expanded)

- Purpose: cheaply approximate cross-table joint selectivity induced by join edges using per-table beliefs.

- Conceptual algorithm (what is computed):
  1. Parse the query to identify participating tables and the predicates applied to each (prototype uses regex-based parsing; robust SQL parsing is a later improvement).
  2. For each table, compute per-table selectivity `s_t = P(predicates on table t)` using the BN object for that table.
  3. For each join edge (e.g., `employees.dept_id = departments.dept_id`), estimate conditional probabilities that link the tables (e.g., P(dept attributes | employee predicates)) by querying BN CPTs and marginalizing as appropriate.
  4. Compose a joint selectivity estimate by combining per-table `s_t` and conditional factors; the composition may include normalization or small correction factors to avoid gross overcounting.

- Concrete prototype features produced for each query (example):
  - `s_e`: selectivity on `employees` predicates.
  - `s_d`: selectivity on `departments` predicates.
  - `s_p`: selectivity on `projects` predicates.
  - `p_dept_given_emp`: conditional probability of a department predicate given matching employees.
  - `p_proj_given_emp`: conditional probability of a project predicate given matching employees.
  - `factorjoin_est`: the composed FactorJoin estimate (a numeric selectivity or rough cardinality estimate depending on whether it's normalized by table sizes).

- Why this helps:
  - Avoids the independence assumption across tables by explicitly conditioning via join edges using per-table CPTs. It is orders of magnitude cheaper than full factor-graph inference since it does not attempt to compute the full joint over many variables simultaneously.

- Implementation limitations & future improvements:
  - SQL parsing is fragile in prototype; move to a proper SQL parser (e.g., `sqlparse`, `moz-sql-parser`) before deploying on varied workloads.
  - Composition rule is heuristic; richer factor-graph approximations or approximate inference (loopy belief propagation on a sparse factor graph) could improve accuracy at higher compute cost.

---

## 6) Fusion MLP (detailed)

- Purpose: learn data-driven corrections to the FactorJoin baseline using a compact regressor.

- Model choice & tech stack:
  - `scikit-learn`'s `MLPRegressor` was used for the prototype: easy to set up, well-tested, and sufficient for small-to-medium feature vectors.
  - Target transformation: predict `y = log(1 + true_count)` to stabilize variance and focus on relative error (Q-error metric aligns with log transformations).

- Features supplied to the MLP:
  - Per-table selectivities (`s_e`, `s_d`, `s_p`).
  - Conditional probabilities derived from CPTs (`p_dept_given_emp`, `p_proj_given_emp`).
  - The `factorjoin_est` composed estimate.
  - Optional scalar query meta-features such as number of joins or number of predicates.

- Training details & outputs:
  - Split workload into training/validation/test CSVs.
  - Optionally standardize features using `sklearn.preprocessing.StandardScaler` fit on training set.
  - Train MLP on training set; evaluate MAE and median Q-error on validation/test sets.
  - Persist the trained model with `joblib.dump` to `models/fusion_model.joblib`.

- Example training command:
```bash
set PYTHONPATH=.&& .venv\Scripts\python.exe scripts\train_fusion.py --train-csv data/queries/train_queries_with_counts.csv --model-out models/fusion_model.joblib
```

- Expected artifact:
  - `models/fusion_model.joblib` (contains the MLP and any feature-scaler if used).

- Why MLP instead of complex NN:
  - The input feature vector is small and engineered; an MLP is fast to train, less prone to overfitting on small datasets, and simpler to integrate.
  - A larger DL-based fusion head is a future step when adding AST embeddings and more training data.

---

## 7) Training Loop & Checkpoints (what was done)

- Prototype behavior:
  - For the fusion baseline we rely on `scikit-learn` training semantics (fit once, optionally grid-search hyperparameters). Checkpointing is simply `joblib.dump` after training; no multi-epoch checkpoint architecture was necessary for the MLP baseline.

- Practical notes:
  - For more complex deep models (e.g., Tree-LSTM), a PyTorch training loop with validation checks, learning-rate schedules, and model checkpointing would be recommended.

---

## 8) Postgres planner comparison harness (deep detail)

- Purpose: gather the optimizer's own cardinality estimates for each query (via `EXPLAIN (FORMAT JSON)`) and compare planner's estimate to the fusion model and ground truth.

- What script does and why (robustness details):
  - Connects to Postgres using `psycopg2` with support for passing explicit connection parameters (`--pg-host`, `--pg-port`, `--pg-user`, `--pg-password`, `--pg-db`). This avoids DSN parsing issues when passwords contain special characters.
  - For each query, executes `EXPLAIN (FORMAT JSON) <sql>` and walks the returned plan JSON recursively to extract the first numeric field whose key contains the substring `rows` (this yields the planner's estimated row count for the plan node). This heuristic is robust across several plan node formats.
  - If `EXPLAIN` fails for a query (e.g., missing table, syntax difference), it marks planner estimate as unknown and continues; the fusion model is still evaluated against the ground truth.

- Metrics computed:
  - For planner: median Q-error and MAE computed only over queries where planner provided an estimate.
  - For model: median Q-error and MAE over the full test set.

- Example command:
```bash
set PYTHONPATH=.&& .venv\Scripts\python.exe scripts\compare_with_postgres.py --test-csv data/queries/test_queries_with_counts.csv --pg-host localhost --pg-port 5432 --pg-user postgres --pg-password "Sriram@IITM24" --pg-db postgres
```

- Observed result in prototype run (representative):
  - Postgres: median Q-error ≈ 250.0, MAE ≈ 1007.036 (500 queries evaluated)
  - Fusion model: median Q-error ≈ 1.985, MAE ≈ 673.326 (500 queries)

- Limitations & opportunities:
  - The plan JSON structure differs across PG versions and planner heuristics; the heuristic to find `rows` works for the plan outputs observed, but a fuller implementation should inspect `Plan` object fields explicitly.
  - Some queries might cause planner to use different plan nodes with different `rows` fields (e.g., for subplans); careful aggregation logic yields more stable planner comparisons (e.g., choose top-level `Plan->Plan Rows` or use the `Plan Rows` of the root node if present).

---

## Summary of Completed Artifacts (what to point to in slides)

- Data CSVs: `data/out/departments.csv`, `data/out/employees.csv`, `data/out/projects.csv` (generated with configured row counts and correlation parameter).
- Schema file: `data/generate_schema.sql` (for loading into Postgres).
- Per-table BN artifacts: `models/bn_<table>.joblib` (serialized per-table BN objects providing selectivity APIs).
- FactorJoin code (feature extraction) and the set of extracted features per query.
- Trained fusion model: `models/fusion_model.joblib`.
- Comparison outputs: printed median Q-error and MAE for both Postgres and the fusion model, generated by `scripts/compare_with_postgres.py`.

---

## Recommended Next Actions (concrete engineering tasks)

1. Replace regex SQL parsing with a robust parser (e.g., `moz-sql-parser` or `sqlglot`) to handle a wider variety of SQL and to reliably map predicates to tables/columns.
2. Improve FactorJoin composition rules or add a light-weight factor-graph inference pass (e.g., loopy BP on a sparse join-graph) to capture higher-order interactions.
3. Implement an SQL AST encoder (Tree-LSTM) and produce embeddings for queries; feed them along with FactorJoin features into a PyTorch fusion head and train end-to-end.
4. Add deterministic seeds and a configuration file (`config.yaml`) so experiments are reproducible and parameter sweeps are easy.
5. Add unit tests validating per-table BN probabilities against simple hand-crafted distributions to guard against regression.

---

End of deep dive.
