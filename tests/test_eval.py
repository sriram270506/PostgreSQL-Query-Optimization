"""Evaluation machinery: paired comparisons, slicing, tail loss and selection score."""
import numpy as np
import torch

from conftest import requires_data
from scripts.benchmark import card_bucket, join_shape, paired_diff, pred_type
from src.estimator.neural import make_loss, val_score


def test_paired_diff_verdicts():
    rng = np.random.default_rng(0)
    b = rng.exponential(1.0, 500)
    assert paired_diff(b * 0.5, b)["verdict"] == "a better"
    assert paired_diff(b * 2.0, b)["verdict"] == "b better"
    noise = b + rng.normal(0, 1e-3, 500)
    r = paired_diff(noise, b)
    assert r["verdict"] == "inconclusive" and r["ci95_mean"][0] < 0 < r["ci95_mean"][1]
    assert paired_diff(b[:5], b[:5])["verdict"] == "too few queries"


def test_card_buckets():
    assert [card_bucket(x) for x in (0, 1, 10, 11, 1000, 1001, 10 ** 6, 10 ** 6 + 1)] == \
        ["0 rows", "1-10", "1-10", "11-1k", "11-1k", "1k-1M", "1k-1M", ">1M"]


@requires_data
def test_join_shape_and_pred_type(engine):
    p = engine.parse
    assert join_shape(p("SELECT COUNT(*) FROM employees")) == "single table"
    assert join_shape(p("SELECT COUNT(*) FROM employees e JOIN assignments a ON e.emp_id = a.emp_id "
                        "JOIN projects p ON a.proj_id = p.proj_id")) == "chain"
    assert pred_type(p("SELECT COUNT(*) FROM employees WHERE age > 30")) == "range only"
    assert pred_type(p("SELECT COUNT(*) FROM employees WHERE age > 30 AND city = 'Pune'")) == "mixed"
    assert pred_type(p("SELECT COUNT(*) FROM employees")) == "no filter"


def test_tail_loss_only_adds_beyond_2x():
    y = torch.zeros(4)
    small = torch.tensor([0.1, -0.2, 0.3, 0.0])  # all within 2x (|e| < log 2)
    big = torch.tensor([2.0, 0.0, 0.0, 0.0])
    tail, mse = make_loss({"loss": "tail", "tail_lambda": 1.0}), make_loss({"loss": "mse"})
    assert torch.isclose(tail(small, y), mse(small, y))
    assert tail(big, y) > mse(big, y)


def test_selection_score_penalises_the_tail():
    flat = np.full(100, 0.2)
    spiky = np.r_[np.full(95, 0.05), np.full(5, 1.5)]
    assert val_score(spiky, {}) < val_score(flat, {})  # mean alone prefers the spiky model
    assert val_score(spiky, {"select": "tail"}) > val_score(flat, {"select": "tail"})
