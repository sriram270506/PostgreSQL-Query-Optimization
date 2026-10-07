"""
Train a Fusion MLP on 6-table feature set from FactorJoin.
Outputs models/fusion_model.joblib and prints MAE / Q-error / latency stats.
"""
import os
import time
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from src.estimator.factorjoin import FactorJoin


FEATURE_COLS = ['s_e', 's_d', 's_p', 's_a', 's_l', 's_b', 'p_dept_given_emp', 'p_proj_given_emp', 'factorjoin_estimate']


def extract_features_from_df(df: pd.DataFrame, fj: FactorJoin) -> np.ndarray:
    feats = []
    for sql in df['sql'].tolist():
        f = fj.extract_features(sql)
        feats.append([
            f.get('s_e', 1.0),
            f.get('s_d', 1.0),
            f.get('s_p', 1.0),
            f.get('s_a', 1.0),
            f.get('s_l', 1.0),
            f.get('s_b', 1.0),
            f.get('p_dept_given_emp', 1.0),
            f.get('p_proj_given_emp', 1.0),
            f.get('factorjoin_est', 0.0),
        ])
    return np.array(feats, dtype=np.float64)


def q_error(preds, truth):
    preds = np.maximum(preds, 1e-6)
    truth = np.maximum(truth, 1e-6)
    return np.maximum(preds / truth, truth / preds)


def measure_inference_latency(model, X, n_runs=100) -> float:
    """Return average inference latency in ms over n_runs forward passes."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = model.predict(X[:1])
        times.append((time.perf_counter() - t0) * 1000.0)
    return round(float(np.mean(times)), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries-csv', default='data/queries/test_queries_with_counts.csv')
    parser.add_argument('--out', default='models/fusion_model.joblib')
    parser.add_argument('--test-size', type=float, default=0.2)
    args = parser.parse_args()

    print("Loading query workload from:", args.queries_csv)
    df = pd.read_csv(args.queries_csv)

    # If no train_csv exists, split test_csv 80/20
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42)
    os.makedirs('data/queries', exist_ok=True)
    train_df.to_csv('data/queries/train_queries_with_counts.csv', index=False)
    test_df.to_csv('data/queries/eval_queries_with_counts.csv', index=False)
    print(f"Split: {len(train_df)} train / {len(test_df)} test queries")

    print("Extracting features via FactorJoin...")
    fj = FactorJoin(csv_dir='data/out')
    X_train = extract_features_from_df(train_df, fj)
    y_train = train_df['true_count'].values.astype(float)
    X_test = extract_features_from_df(test_df, fj)
    y_test = test_df['true_count'].values.astype(float)

    # Log-transform targets
    y_train_log = np.log1p(y_train)

    print("Training MLP Fusion Model...")
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False
    )
    model.fit(X_train, y_train_log)

    # Evaluate
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    preds = np.maximum(preds, 0.0)

    mae = mean_absolute_error(y_test, preds)
    q = q_error(preds, y_test)
    med_q = np.median(q)
    p95_q = np.percentile(q, 95)

    # Inference latency
    ml_latency_ms = measure_inference_latency(model, X_test)

    # FactorJoin latency from workload
    fj_latency_mean = float(test_df['factorjoin_latency_ms'].mean()) if 'factorjoin_latency_ms' in test_df.columns else None
    pg_latency_mean = float(test_df['postgres_latency_ms'].mean()) if 'postgres_latency_ms' in test_df.columns else None

    print(f"\n=== ML Fusion Model Results ===")
    print(f"MAE:              {mae:.2f}")
    print(f"Median Q-error:   {med_q:.3f}")
    print(f"P95 Q-error:      {p95_q:.3f}")
    print(f"\n=== Latency Comparison (mean per query) ===")
    if pg_latency_mean:
        print(f"PostgreSQL (simulated):  {pg_latency_mean:.2f} ms")
    if fj_latency_mean:
        print(f"FactorJoin:              {fj_latency_mean:.3f} ms")
    print(f"ML Fusion (inference):   {ml_latency_ms:.4f} ms")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({'model': model, 'stats': {
        'mae': mae, 'median_q': med_q, 'p95_q': p95_q,
        'ml_latency_ms': ml_latency_ms,
        'fj_latency_ms': fj_latency_mean,
        'pg_latency_ms': pg_latency_mean,
        'n_train': len(train_df), 'n_test': len(test_df)
    }}, args.out)
    print(f"\nSaved fusion model to {args.out}")


if __name__ == '__main__':
    main()
