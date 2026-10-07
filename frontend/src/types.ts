export interface TableSummary {
  name: string;
  row_count: number;
  columns: { name: string; type: string }[];
  head: Record<string, any>[];
}

export interface TableDataResponse {
  table_name: string;
  total_rows: number;
  page: number;
  page_size: number;
  total_pages: number;
  columns: { name: string; type: string }[];
  rows: Record<string, any>[];
  stats: Record<string, any>;
}

export interface BNNode {
  id: string;
  label: string;
  bins_count?: number;
  bins?: number[];
}

export interface BNEdge {
  source: string;
  target: string;
  label: string;
}

export interface BNTable {
  table_name: string;
  columns: string[];
  nodes: BNNode[];
  edges: BNEdge[];
  cpts_summary: Record<string, { parent?: string; has_cpt: boolean }>;
}

export interface EstimateResult {
  sql: string;
  features: {
    s_e: number;
    s_d: number;
    s_p: number;
    p_dept_given_emp: number;
    p_proj_given_emp: number;
    factorjoin_est: number;
  };
  factorjoin_estimate: number;
  ml_fusion_estimate?: number;
  true_count?: number;
  q_errors?: {
    factorjoin?: number;
    ml_fusion?: number;
  };
}

export interface SampleQuery {
  id: number;
  sql: string;
  true_count: number;
  factorjoin_estimate: number;
  ml_fusion_estimate?: number;
  fj_q_error?: number;
  ml_q_error?: number;
}

export interface BenchmarkData {
  total_test_queries: number;
  factorjoin_median_q: number;
  ml_fusion_median_q: number;
  postgres_baseline_median_q: number;
  q_distribution: {
    ml_fusion: {
      under_2: number;
      under_5: number;
      under_10: number;
      above_10: number;
    };
    factorjoin: {
      under_2: number;
      under_5: number;
      under_10: number;
      above_10: number;
    };
  };
}
