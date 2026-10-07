import os
import math
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
import joblib

def discretize_series(s, n_bins=10):
    if pd.api.types.is_numeric_dtype(s):
        try:
            cats, bins = pd.qcut(s, q=n_bins, retbins=True, duplicates='drop')
            return cats.astype(str), bins
        except Exception:
            cats, bins = pd.cut(s, bins=n_bins, retbins=True)
            return cats.astype(str), bins
    else:
        return s.astype(str), None

def pairwise_mi(df):
    cols = df.columns.tolist()
    n = len(cols)
    mi = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            mi[i, j] = mutual_info_score(df.iloc[:, i], df.iloc[:, j])
            mi[j, i] = mi[i, j]
    return cols, mi

def maximum_spanning_tree(cols, mi_matrix):
    n = len(cols)
    if n <= 1:
        return []
    selected = [False] * n
    parent = [-1] * n
    key = [-math.inf] * n
    key[0] = 0
    for _ in range(n):
        u = max((k for k in range(n) if not selected[k]), key=lambda x: key[x])
        selected[u] = True
        for v in range(n):
            if not selected[v] and mi_matrix[u, v] > key[v]:
                key[v] = mi_matrix[u, v]
                parent[v] = u
    edges = []
    for v in range(1, n):
        if parent[v] != -1:
            edges.append((cols[parent[v]], cols[v]))
    return edges

def estimate_cpts(df, edges, alpha=1.0):
    parent = {child: par for par, child in edges}
    cpts = {}
    N = df.shape[0]
    for col in df.columns:
        if col not in parent or parent[col] is None:
            # Root node
            freqs = df[col].value_counts().to_dict()
            k = len(freqs)
            probs = {str(v): (freqs.get(v, 0) + alpha) / (N + alpha * k) for v in df[col].unique()}
            cpts[col] = {"_prior": probs}
        else:
            par = parent[col]
            groups = df.groupby([par, col]).size().unstack(fill_value=0)
            par_vals = groups.index.tolist()
            child_vals = groups.columns.tolist()
            table = {}
            for pv in par_vals:
                row = groups.loc[pv]
                total = row.sum()
                k = len(child_vals)
                table[str(pv)] = {str(cv): (int(row.get(cv, 0)) + alpha) / (total + alpha * k) for cv in child_vals}
            cpts[col] = {"parent": par, "table": table}
    return cpts

def learn_bn_for_table(csv_path, n_bins=10):
    df = pd.read_csv(csv_path)
    
    # CRITICAL FIX: Explicitly exclude identifier and non-correlated columns (IDs, names)
    exclude_cols = {'name', 'emp_id', 'proj_id', 'assign_id', 'loc_id', 'budget_id'}
    cols_to_use = [c for c in df.columns if c not in exclude_cols]
    df = df.loc[:, cols_to_use]

    disc_info = {}
    disc_df = pd.DataFrame()
    for col in df.columns:
        cats, bins = discretize_series(df[col], n_bins=n_bins)
        disc_df[col] = cats
        disc_info[col] = {"bins": bins.tolist() if bins is not None else None}

    cols, mi = pairwise_mi(disc_df)
    edges = maximum_spanning_tree(cols, mi)
    cpts = estimate_cpts(disc_df, edges)
    model = {"cols": cols, "edges": edges, "cpts": cpts, "discretization": disc_info}
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-dir', default='data/out')
    parser.add_argument('--out-dir', default='models')
    parser.add_argument('--n-bins', type=int, default=10)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    tables = ['departments', 'employees', 'projects', 'assignments', 'locations', 'budgets']
    for tbl in tables:
        fname = f"{tbl}.csv"
        path = os.path.join(args.csv_dir, fname)
        if os.path.exists(path):
            print('Learning clean Chow-Liu BN for', tbl)
            model = learn_bn_for_table(path, n_bins=args.n_bins)
            out_path = os.path.join(args.out_dir, f'bn_{tbl}.joblib')
            joblib.dump(model, out_path)
            print('Saved', out_path)

if __name__ == '__main__':
    main()
