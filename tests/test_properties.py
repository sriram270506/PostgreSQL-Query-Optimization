"""Property-based checks that must hold for every query, not just the ones in a test set.

* order invariance: the same query written in a different order gets the same estimate
* monotonicity: adding a filter never raises a FactorJoin / PostgreSQL estimate
* soundness of the upper bound: true count <= bound, always
* BN structure: independent noise columns are not linked (G-test), real dependencies are
"""
import random

import numpy as np
import pandas as pd
import pytest

from conftest import requires_data, requires_stats
from src.engine import Join, Pred, Query
from src.engine.datasets import HR
from src.engine.executor import Table
from src.estimator.bn import ChowLiuBN
from src.estimator.bounds import Bounds
from src.workload.generator import WorkloadGenerator

pytestmark = requires_data


@pytest.fixture(scope="module")
def queries(engine):
    gen = WorkloadGenerator(engine.catalog, seed=7)
    return [gen.query() for _ in range(150)]


def _shuffled(q: Query, rng: random.Random) -> Query:
    rels = list(q.rels.items())
    rng.shuffle(rels)
    preds = list(q.preds)
    rng.shuffle(preds)
    return Query(rels=dict(rels), joins=[Join(j.b, j.bcol, j.a, j.acol) for j in reversed(q.joins)], preds=preds)


def test_estimates_are_order_invariant(engine, fj, queries):
    rng = random.Random(0)
    for q in queries:
        s = _shuffled(q, rng)
        assert engine.true_count(s) == engine.true_count(q)
        assert fj.estimate(s) == pytest.approx(fj.estimate(q), rel=1e-9)
        assert engine.planner.estimate(s) == pytest.approx(engine.planner.estimate(q), rel=1e-9)


def test_featurisation_is_order_invariant(engine, fj, queries):
    from src.estimator.neural import Featurizer
    fz = Featurizer(engine.catalog, fj, engine.planner)
    rng = random.Random(1)
    for q in queries[:40]:
        a, b = fz.tree(q), fz.tree(_shuffled(q, rng))
        assert np.array_equal(a.x, b.x) and np.array_equal(a.parent, b.parent)
        assert np.allclose(fz.stats(q), fz.stats(_shuffled(q, rng)))


def test_adding_a_filter_never_raises_fj_or_pg(engine, fj, queries):
    rng = random.Random(2)
    for q in queries:
        a = rng.choice(list(q.rels))
        t = q.rels[a]
        cols = [c for c in HR.filters[t] if engine.catalog.tables[t].numeric[c]]
        if not cols:
            continue
        c = rng.choice(cols)
        med = float(np.median(engine.catalog.tables[t].cols[c]))
        tighter = Query(rels=q.rels, joins=q.joins, preds=q.preds + [Pred(a, c, "ge", int(med))])
        assert fj.estimate(tighter) <= fj.estimate(q) * (1 + 1e-9)
        assert engine.planner.estimate(tighter) <= engine.planner.estimate(q) * (1 + 1e-9) + 1


def test_upper_bound_is_sound(engine, queries):
    b = Bounds(engine.catalog)
    for q in queries:
        assert engine.true_count(q) <= b.upper(q)


@requires_stats
def test_upper_bound_is_sound_on_stats_ceb(stats_engine):
    b = Bounds(stats_engine.catalog)
    for line in open("data/stats/raw/stats_CEB.sql").readlines()[:60]:
        c, sql = line.strip().split("||", 1)
        assert int(c) <= b.upper(stats_engine.parse(sql))


def test_g_test_prunes_noise_keeps_signal():
    rng = np.random.default_rng(0)
    n = 20000
    a = rng.integers(0, 10, n)
    df = pd.DataFrame({"a": a, "b": (a + rng.integers(0, 2, n)) % 10,  # depends on a
                       "noise": rng.integers(0, 64, n)})                 # independent of both
    bn = ChowLiuBN(Table("t", df), ["a", "b", "noise"], {})
    linked = {frozenset((x, y)) for x, y, _ in bn.edges}
    assert frozenset(("a", "b")) in linked
    assert all("noise" not in e for e in linked)
