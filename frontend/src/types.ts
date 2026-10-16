import type { PlanNode } from './components/charts'

export type QSummary = { n: number; median: number; mean: number; p90: number; p95: number; p99: number; max: number; within_2x: number; within_10x: number }
export type Latency = { mean_ms: number; p50_ms: number; p95_ms: number }
export type PlanStats = { n: number; optimal_rate: number; median_ratio: number; mean_ratio: number; p95_ratio: number; max_ratio: number }

export type Bench = {
  n_test: number; n_train: number; n_val: number
  estimators: string[]; labels: Record<string, string>
  accuracy: Record<string, { overall: QSummary; correlated: QSummary; by_category: Record<string, QSummary>; under: number }>
  latency: Record<string, Latency>
  plans: Record<string, PlanStats>
  worst_for_postgres: ({ sql: string; category: string; true: number } & Record<string, number | string>)[]
  scatter: { true: number[]; category: string[] } & Record<string, number[]>
  tuning: Record<string, { best_params: Record<string, unknown>; best_val: number; n_params: number; tune_seconds: number; train_seconds: number; n_trials: number }>
}

export type Tuning = {
  model: string; best_params: Record<string, string | number>; best_val: number
  final_history: { epoch: number; train_loss: number; val_mean_log_q: number }[]
  test: Record<string, number>; n_params: number; tune_seconds: number; train_seconds: number
  importance: Record<string, number>
  trials: { number: number; state: string; value: number | null; params: Record<string, string | number>; curve: number[] }[]
}

export type EstResult = { name: string; label: string; estimate: number; latency_ms: number; qerror: number; under: boolean }
export type RunResult = {
  sql: string; count: number; exec_ms: number; count_ms: number; fetch_ms: number
  rows: { columns: string[]; data: unknown[][]; shown: number; truncated: boolean } | null
  estimators: EstResult[]; best: string
  scan: { alias: string; table: string; actual: number; pg: number; factorjoin: number; total: number; preds: string[] }[]
  plans: null | { optimal_cost: number; optimal: PlanNode; estimators: Record<string, { true_cost: number; ratio: number; plan: PlanNode }> }
  postgres_live_error: string | null
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
