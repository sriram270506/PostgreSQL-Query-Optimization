import type { PlanNode } from './components/charts'

export type QSummary = { n: number; median: number; mean: number; p90: number; p95: number; p99: number; max: number; within_2x: number; within_10x: number }
export type Latency = { mean_ms: number; p50_ms: number; p95_ms: number }
export type PlanStats = { n: number; optimal_rate: number; median_ratio: number; mean_ratio: number; p95_ratio: number; max_ratio: number }

export type Acc = {
  overall: QSummary; ci95: Record<'median' | 'p95' | 'p99', [number, number]>; under: number
  guards?: { fallback_rate: number; clamp_rate: number }; correlated?: QSummary
} & Record<string, unknown>
export type Sig = { a: string; b: string; mean_log_q_a: number; mean_log_q_b: number; a_better_frac: number; b_better_frac: number; p_value: number }
export type Props = { order_invariant_rate: number; monotone_violation_rate: number; monotone_violation_median_ratio: number; raw_bound_violation_rate: number }
export type MS = { mean: number; std: number }
export type Cmp = { n: number; verdict: string; mean_log_q_diff?: number; ci95_mean?: [number, number]; p95_log_diff?: number; ci95_p95?: [number, number] }

export type Bench = {
  dataset: string; n_test: number; n_train: number; n_val: number; data_hash: string; stale_models: Record<string, boolean>
  estimators: string[]; labels: Record<string, string>
  accuracy: Record<string, Acc>
  significance: Sig[]
  seeds: Record<string, Record<'median' | 'p95' | 'p99' | 'max' | 'within_2x', MS> & { n_seeds: number }> & { agreement?: Record<string, number> }
  scatter: { true: number[]; category: string[] } & Record<string, number[]>
  ceb?: { accuracy: Record<string, Acc>; significance: Sig[]; scatter: { true: number[] } & Record<string, number[]>; plans?: Record<string, PlanStats> }
  ood: Record<string, { n: number; definition?: string; accuracy: Record<string, Acc>; compare?: Record<string, Cmp> }>
  breakdown?: Record<string, Record<string, { n: number } & Record<string, { median: number; p95: number; mean_log_q: number }>>>
  compare?: Record<string, Record<string, Cmp>>
  calibration?: Record<string, { spearman_std_vs_error: number; fallback_rate: number; coverage_curve: { coverage: number; mean_log_q: number; p95_q: number }[] }>
  special?: Record<string, { label: string; n: number; accuracy: Record<string, Acc>; compare: Record<string, Cmp> }>
  provenance?: { git_commit: string | null; git_dirty: boolean; time_utc: string; data_hash: string; workload_sha256: string; hardware: Record<string, unknown>; postgres: { version?: string } | null }
  drift?: Record<string, Record<string, QSummary>>
  plans?: Record<string, PlanStats>
  latency: Record<string, Latency & { p99_ms: number }>
  properties: Record<string, Props>
  tuning: Record<string, { best_params: Record<string, unknown>; best_val: number; n_trials: number; tune_seconds: number; importance: Record<string, number> }>
  ensemble: Record<string, { tau: number; seeds: number; val_mean_log_q_ensemble: number; train_seconds: number[] }>
}

export type Tuning = {
  model: string; best_params: Record<string, string | number>; best_val: number
  tune_seconds: number; importance: Record<string, number>
  ensemble?: { tau: number; seeds: number; val_mean_log_q_ensemble: number; train_seconds: number[] }
  trials: { number: number; state: string; value: number | null; params: Record<string, string | number>; curve: number[] }[]
}

export type EstResult = { name: string; label: string; estimate: number; latency_ms: number; qerror: number; under: boolean; std?: number; fallback?: boolean; clamped?: boolean }
export type RunResult = {
  sql: string; count: number; exec_ms: number; count_ms: number; fetch_ms: number
  rows: { columns: string[]; data: unknown[][]; shown: number; truncated: boolean } | null
  estimators: EstResult[]; best: string
  scan: { alias: string; table: string; actual: number; pg: number | null; factorjoin: number; total: number; preds: string[] }[]
  plans: null | { optimal_cost: number; optimal: PlanNode; estimators: Record<string, { true_cost: number; ratio: number; plan: PlanNode }> }
  postgres_live_error: string | null
  upper_bound: number
}

export type Neuron = { unit: number; profile: number[]; detects: string; detail: string; purity: number; card_corr: number; top: { label: string; act: number }[] }
export type VizGlobal = {
  model: string; n_queries: number
  mlp_layers: { name: string; width: number; probe_r2: number; pca: number[][]; pca_var: number[]; dead_frac: number }[]
  pca_color: number[]
  receptive_fields: { groups: string[]; units: number[][] }
  attribution: { group: string; mean_abs: number }[]
  residual: { mean_abs: number; frac_up: number }
  tree_levels?: { kind: string; n_nodes: number; probe_r2: number; mean_act: number; gate_input: number; gate_output: number; gate_forget: number | null; active_units: number }[]
  node_map?: { xy: number[][]; kind: string[]; label: string[]; var: number[]; target: number[] }
  neurons?: Neuron[]
  kinds?: string[]
}
export type TraceNode = { id: number; label: string; kind: string; parent: number; height: number; h: number[]; gate_i: number; gate_o: number; gate_f: number | null; probe_log10: number | null; true_log10: number }
export type Trace = {
  estimate: number; fj_estimate: number; residual: number; true_count: number
  mlp: { name: string; acts: number[] }[]
  input: { group: string; values: number[]; attribution: number }[]
  nodes?: TraceNode[]
}
