"""Pytest configuration — puts the repo root on sys.path and shares fixtures.

Loading the CSVs and learning the per-table BNs takes a moment, so the engine and
FactorJoin instances are session-scoped and reused across every test.
"""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CSV_DIR = os.path.join(ROOT, "data", "out")

requires_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(CSV_DIR, "employees.csv")),
    reason="synthetic data not generated — run `python -m scripts.generate_data`",
)


@pytest.fixture(scope="session")
def engine():
    from src.engine import Engine
    return Engine(CSV_DIR)


@pytest.fixture(scope="session")
def fj(engine):
    from src.estimator.factorjoin import FactorJoin
    return FactorJoin(engine.catalog)


@pytest.fixture(scope="session")
def tables():
    """Raw dataframes, used to compute ground truth independently of the estimator."""
    names = ["departments", "employees", "projects", "assignments", "locations", "budgets"]
    out = {}
    for n in names:
        path = os.path.join(CSV_DIR, f"{n}.csv")
        if os.path.exists(path):
            out[n] = pd.read_csv(path)
    return out
