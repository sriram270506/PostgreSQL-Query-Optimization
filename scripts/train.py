"""Train a toy cardinality estimator using synthetic data."""
import argparse
import joblib
import numpy as np
from estimator.model import CardinalityEstimator
from estimator.features import extract_simple_features

def gen_synthetic(n=500):
    sqls = []
    X = []
    y = []
    for i in range(n):
        t = np.random.randint(1, 4)
        p = np.random.randint(0, 5)
        c = np.random.randint(1, 6)
        sql = f"SELECT col1{',' if c>1 else ''}{' ,col2'*(c-1)} FROM t1"
        if t>1:
            for j in range(2, t+1):
                sql += f" JOIN t{j} ON t1.id = t{j}.ref"
        if p>0:
            sql += " WHERE " + " AND ".join([f"col{i}>0" for i in range(p)])
        feats = extract_simple_features(sql)
        # synthetic cardinality: base * tables * columns / (1+predicates)
        card = 100 * t * max(1, c) / (1 + 0.5 * p) + np.random.randn() * 10
        sqls.append(sql)
        X.append(feats)
        y.append(card)
    return np.vstack(X), np.array(y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-model", default="model.joblib")
    args = parser.parse_args()

    X, y = gen_synthetic(800)
    est = CardinalityEstimator()
    est.fit(X, y)
    joblib.dump(est, args.output_model)
    print("Saved model to", args.output_model)

if __name__ == "__main__":
    main()
