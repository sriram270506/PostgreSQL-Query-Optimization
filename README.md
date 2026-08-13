PostgreSQL Query Optimization — Cardinality Estimation (Python)

Minimal Python prototype implementing a simple cardinality estimation pipeline inspired by the provided PDF. It includes synthetic data generation, feature extraction, a scikit-learn model, training and evaluation scripts, and unit tests.

Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts\train.py --output-model model.joblib
python scripts\evaluate.py --model model.joblib
```
