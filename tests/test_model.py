import joblib
import numpy as np
from estimator.model import CardinalityEstimator
from estimator.features import extract_simple_features

def test_model_fit_predict(tmp_path):
    # small synthetic dataset
    X = np.array([[1,0,2],[2,1,3],[1,2,1]], dtype=float)
    y = np.array([200.0, 300.0, 120.0])
    est = CardinalityEstimator()
    est.fit(X, y)
    preds = est.predict(X)
    assert preds.shape == y.shape
    # model should be reasonably close on training data
    assert np.mean(np.abs(preds - y)) < 100.0

def test_feature_extractor():
    sql = "SELECT a, b FROM t1 JOIN t2 ON t1.id=t2.ref WHERE a>0 AND b>5"
    feats = extract_simple_features(sql)
    assert feats.shape == (3,)
    assert feats[0] >= 1
