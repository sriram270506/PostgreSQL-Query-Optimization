"""Pytest configuration — puts the repo root on sys.path and shares fixtures.

Loading the CSVs and learning the per-table BNs takes a moment, so the engine and
FactorJoin instances are session-scoped and reused across every test.
"""
import os
import sys

import pandas as pd
import pytest

# the API's per-client rate limit would throttle the test suite itself
os.environ.setdefault("CARDLAB_RATE_PER_MIN", "1000000")

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
    return Engine("hr")


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


requires_stats = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, "data", "stats", "raw", "stats_CEB.sql")),
    reason="STATS not downloaded - run `python -m scripts.fetch_stats`",
)


@pytest.fixture(scope="session")
def stats_engine():
    from src.engine import Engine
    return Engine("stats")


# CI_REQUIRE_ALL=1 (the nightly job): a skipped test is a failure, so a missing database,
# dataset or model can never turn a red run green by silently skipping
_skipped = []


def pytest_runtest_logreport(report):
    if report.skipped:
        _skipped.append(report.nodeid)


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("CI_REQUIRE_ALL") == "1" and _skipped:
        print(f"\nCI_REQUIRE_ALL: {len(_skipped)} test(s) skipped:\n  " + "\n  ".join(_skipped))
        session.exitstatus = 1
