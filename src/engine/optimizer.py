"""Join-order optimizer driven by a pluggable cardinality estimator.

Dynamic programming over connected relation subsets (bushy plans, no cross products),
minimising C_out = sum of intermediate result sizes. Running the same DP with each
estimator and scoring the chosen plan with TRUE sub-join sizes shows how estimation
errors turn into slower plans.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

from .sql import Query

Card = Callable[[Query], float]


def _connected(q: Query, s: FrozenSet[str]) -> bool:
    start = next(iter(s))
    seen, stack = {start}, [start]
    while stack:
        x = stack.pop()
        for j in q.joins:
            for a, b in ((j.a, j.b), (j.b, j.a)):
                if a == x and b in s and b not in seen:
                    seen.add(b)
                    stack.append(b)
    return seen == set(s)


def connected_subsets(q: Query) -> List[FrozenSet[str]]:
    rels = list(q.rels)
    return [frozenset(c) for k in range(2, len(rels) + 1)
            for c in combinations(rels, k) if _connected(q, frozenset(c))]


def best_plan(q: Query, card: Dict[FrozenSet[str], float]) -> Tuple[float, dict]:
    """DP over `card` (subset -> estimated rows). Returns (estimated C_out, plan tree)."""
    best: Dict[FrozenSet[str], Tuple[float, dict]] = {
        frozenset([a]): (0.0, {"rels": [a]}) for a in q.rels
    }
    for s in sorted(card, key=len):
        items = sorted(s)
        cand: Optional[Tuple[float, dict]] = None
        for k in range(1, len(items)):
            for left in combinations(items, k):
                l, r = frozenset(left), s - frozenset(left)
                if items[0] not in l or l not in best or r not in best:
                    continue  # skip mirrored splits and disconnected halves
                cost = best[l][0] + best[r][0] + card[s]
                if cand is None or cost < cand[0]:
                    cand = (cost, {"rels": sorted(s), "left": best[l][1], "right": best[r][1]})
        if cand:
            best[s] = cand
    return best[frozenset(q.rels)]


def plan_true_cost(plan: dict, true_card: Dict[FrozenSet[str], float]) -> float:
    if "left" not in plan:
        return 0.0
    return (true_card[frozenset(plan["rels"])]
            + plan_true_cost(plan["left"], true_card) + plan_true_cost(plan["right"], true_card))


def annotate(plan: dict, est: Dict[FrozenSet[str], float], true: Dict[FrozenSet[str], float]) -> dict:
    out = {"rels": plan["rels"]}
    if "left" in plan:
        key = frozenset(plan["rels"])
        out.update(est=est[key], true=true[key],
                   left=annotate(plan["left"], est, true), right=annotate(plan["right"], est, true))
    return out
