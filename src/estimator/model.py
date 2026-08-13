from sklearn.ensemble import RandomForestRegressor
import numpy as np

class CardinalityEstimator:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
