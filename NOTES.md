# Hybrid Cardinality Estimation — Detailed Notes

Purpose: These notes summarize the prototype implementation, reasoning behind design choices, evaluation methodology, results, limitations, and recommended next steps. Use this as a source for presentation slides aimed at a technical audience (motivation → design → experiments → results → roadmap).

## Executive Summary

- Objective: Build a hybrid cardinality estimator that combines per-table probabilistic models, a lightweight cross-table composition (FactorJoin-style), and a small learned fusion model to produce accurate selectivity estimates for SQL COUNT/aggregation queries.
- Key idea: Model intra-table correlations precisely, approximate cross-table correlations cheaply, and learn residual corrections with a compact regressor.
- Outcome (synthetic workload): the fusion model delivered substantially better median Q-error and reduced MAE compared to the Postgres planner on the tests run (fusion median Q-error ≈ 1.99 vs Postgres ≈ 250; fusion MAE ≈ 673 vs Postgres MAE ≈ 1007). These results demonstrate the potential of the hybrid approach on correlated data.

## Motivation

- Why cardinality estimation matters: Optimizers choose join orders and access methods based on estimated result sizes; large estimation errors lead to poor plans and drastically slower queries.
- Failure mode of conventional optimizers: single-column histograms and independence assumptions break when attributes are correlated within or across tables; this is especially harmful for multi-join queries.
- Design goal: achieve much better estimates than planner heuristics without requiring a full joint model across all tables (which is computationally expensive and hard to maintain).

## High-Level Approach

1. Per-table probabilistic models: learn compact Bayesian-network-style models per table that capture intra-table dependencies (e.g., via a Chow-Liu tree and learned CPTs). These models provide marginal and conditional probabilities for predicates against each table.
2. FactorJoin-style approximation: combine per-table beliefs under join connectivity to approximate cross-table selectivity effects. This uses composition rules that preserve the per-table marginals while approximating interactions through join edges.
3. Fusion model (MLP): use numerical features derived from the per-table models and FactorJoin composition as inputs to a small learned regressor that predicts log(1 + true_count). The regressor corrects systematic residuals left by the approximation.

Rationale: per-table models are cheap and parallelizable; FactorJoin provides a principled, explainable composed estimate; the MLP provides data-driven calibration.

## Data & Workload (Prototype Setup)

- Purpose of synthetic data: create well-understood cross-table correlations so the estimator's ability to capture them can be measured precisely.
- Data generation concept: three tables (departments, employees, projects) with an injected correlation such that department→city influences project→budget_tier via employees (i.e., employees link departments and projects). The generation process controlled correlation strength so we could stress the estimator.
- Workload: a set of training and test queries (counts) covering single-table predicates and multi-table joins of varying selectivity. Ground truth counts computed exactly from the generated dataset.

## Per-Table Probabilistic Models (Details)

- Objective: capture attribute dependencies inside each table.
- Method: learn a sparse tree-structured Bayesian network (Chow-Liu) per table and derive conditional probability tables. This yields fast marginal/conditional queries like P(predicate | table) or P(column=value | other-predicate).
- Practicalities: training is per-table and parallelizable; outputs are scalars (selectivities, conditional probabilities) that are robust features to feed into the FactorJoin composition.

## FactorJoin Approximation (Concept and Benefits)

- Problem addressed: joins create dependencies across tables that simple multiplication of marginals (assuming independence) misses.
- FactorJoin idea (approximate factor-graph composition): evaluate per-table conditional/marginal probabilities given the query predicates, then combine them using a composition rule that accounts for join structure and references. This yields an initial joint estimate — the "FactorJoin estimate" — that is a better starting point than naive independence.
- Benefits: cheap to compute, interpretable (we can trace which table contributed what), and provides a powerful feature for the learned model.

## Fusion Model (Architecture & Training)

- Role: calibrate and correct the FactorJoin-based initial estimate.
- Input features: per-table selectivities, conditional probabilities from CPTs, FactorJoin estimate, and simple query-level metadata (e.g., number of predicates or join count) if available.
- Target: predict `log(1 + true_count)` to stabilize variance across orders of magnitude.
- Model: small MLP regressor (a few dense layers) trained with mean squared error on the log-target. Inference uses `exp(pred) - 1` to produce the final cardinality estimate.

Training notes:
- Log-transforming counts focuses the model on relative differences and reduces the influence of very large counts.
- Standardize input features; use early stopping or validation-based checkpointing to avoid overfitting for small workloads.

## Evaluation Protocol & Metrics

- Ground truth: exact counts from the synthetic dataset.
- Evaluated baselines: Postgres planner estimates (extracted via EXPLAIN (FORMAT JSON) for each query) and the fusion model's predictions.
- Metrics:
  - Q-error = max(pred/true, true/pred). Report median Q-error across queries as the primary robustness metric.
  - Mean Absolute Error (MAE) to quantify absolute deviation.

Handling edge cases:
- If the database EXPLAIN fails for a query (e.g., table not present), the planner estimate is marked unknown and that query is excluded from planner metrics but retained for the fusion model evaluation.

## Experiments Performed

1. Generated synthetic dataset and CSV exports with controlled correlation.
2. Learned per-table probabilistic models to extract selectivity and conditional statistics.
3. Implemented FactorJoin to compute composed estimates per query.
4. Extracted features from per-table models and FactorJoin; trained a compact MLP on the training workload to predict log-counts.
5. Saved the fusion model and evaluated it on the test workload.
6. Loaded the generated CSVs into a Postgres instance and ran EXPLAIN for each test query to gather planner estimates.
7. Compared planner vs fusion vs ground-truth using median Q-error and MAE.

## Representative Results (Prototype run on synthetic data)

- Postgres planner: Median Q-error ≈ 250.0; MAE ≈ 1007.036 (on the evaluated workload).
- Fusion model: Median Q-error ≈ 1.985; MAE ≈ 673.326.

Interpretation: The fusion model significantly reduced median Q-error compared to the planner on the synthetic workload, indicating improved relative accuracy across queries. The MAE improvement indicates improved absolute accuracy as well.

## Why This Works (Intuition)

- Per-table models capture intra-table structure accurately; this prevents mistaken independence assumptions for attributes within a table.
- FactorJoin composes these accurate per-table beliefs into a better joint estimate than naive multiplication.
- The MLP learns residual corrections, systematically reducing over/underestimation arising from the FactorJoin approximation where it is imperfect.

## Limitations

- Results are from synthetic data — behavior on real-world, more complex schemas can differ.
- FactorJoin is an approximation and may miss higher-order interactions across many join edges.
- The current prototype uses hand-engineered features; a learned SQL AST encoder (Tree-LSTM) would be more general and possibly more accurate but requires more engineering and training data.
- Operational concerns: model refreshing, counting rare values, and ensuring robust fallbacks when model inputs change or are unavailable.

## Reproducibility & Demonstration Suggestions

- Reproduce the experiment by generating synthetic data with controlled correlation, computing ground-truth counts, training per-table models, computing FactorJoin features, training the fusion MLP, loading CSVs into Postgres, and running EXPLAIN on the same workload to collect planner estimates.
- For live demos: pick one or two illustrative queries and show the step-by-step breakdown — per-table selectivities, FactorJoin value, fusion model prediction, planner's estimate, and ground-truth. Visualize as a small table or annotated example.

## Suggested Slide Outline (7–10 slides)

1. Title + one-line takeaway (e.g., "Hybrid estimator reduces median Q-error by orders of magnitude on correlated synthetic data").
2. Problem: cardinality estimation failures and why cross-table correlation is hard.
3. Data: synthetic setup and correlation injection (one concise example).
4. Method: diagram showing per-table modeling → FactorJoin composition → fusion MLP.
5. Training & evaluation methodology (metrics and workflow).
6. Results: bar chart for median Q-error and MAE + Q-error distribution plot.
7. Walkthrough: illustrative query with per-step numbers.
8. Limitations & practical concerns.
9. Roadmap / next steps: AST encoder, end-to-end fusion, real datasets.
10. Appendix: reproducibility commands and hyperparameters.

## Next Steps (Technical Roadmap)

1. Integrate per-table BN marginals more exhaustively among fusion features, retrain model, and quantify gains.
2. Implement an SQL AST encoder (Tree-LSTM) to extract structural and literal embeddings from queries.
3. Replace the MLP head with a fusion head that ingests both FactorJoin features and AST embeddings; train end-to-end when feasible.
4. Evaluate on additional synthetic configurations and real-world schemas to measure robustness and deployment readiness.
5. Engineering: build a refresh pipeline for per-table models and a compact serialization format for runtime integration; design safe fallbacks for unseen predicates/columns.

## Appendix — Talking Points

- Motivation: optimizer errors cause plan quality regressions—our method reduces these by modeling data dependencies.
- Per-table models: compact, parallel, and capture important attribute correlations.
- FactorJoin: a pragmatic middle ground between independence assumption and full graphical inference.
- Fusion MLP: a lightweight learned calibration layer that corrects systematic biases in composed estimates.

---

End of notes. Use these sections as one-slide-per-section content and the demonstration/example material for the 'walkthrough' slide where you show numbers step-by-step.
