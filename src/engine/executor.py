"""In-memory columnar executor: exact COUNT(*) and row fetch for equi-join queries.

Acyclic joins (every workload query) are counted without materialising the join:
each relation sends its parent a per-join-key sum of row weights (Yannakakis-style
counting), so a 5-way join with 10^8 output rows costs a few group-bys. Row fetch
reuses those weights as a full semi-join reducer, so capping intermediate results
at LIMIT never drops valid rows.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .sql import Join, Pred, Query, SQLError

TABLES = ["departments", "employees", "projects", "assignments", "locations", "budgets"]
# Foreign-key style join edges of the schema: (table, col, table, col).
JOIN_EDGES = [
    ("departments", "dept_id", "employees", "dept_id"),
    ("departments", "dept_id", "projects", "dept_id"),
    ("employees", "dept_id", "projects", "dept_id"),
    ("employees", "emp_id", "assignments", "emp_id"),
    ("projects", "proj_id", "assignments", "proj_id"),
    ("projects", "proj_id", "budgets", "proj_id"),
    ("employees", "city", "locations", "city"),
]
DEFAULT_ROW_LIMIT = 200
CYCLIC_ROW_GUARD = 5_000_000


class Table:
    def __init__(self, name: str, df: pd.DataFrame):
        self.name = name
        self.df = df
        self.n = len(df)
        self.cols: Dict[str, np.ndarray] = {}
        self.numeric: Dict[str, bool] = {}
        for c in df.columns:
            num = pd.api.types.is_numeric_dtype(df[c])
            self.numeric[c] = num
            self.cols[c] = df[c].to_numpy(dtype=np.float64 if num else object)


class Catalog:
    def __init__(self, csv_dir: str = "data/out"):
        self.tables: Dict[str, Table] = {}
        for t in TABLES:
            p = os.path.join(csv_dir, f"{t}.csv")
            if os.path.exists(p):
                self.tables[t] = Table(t, pd.read_csv(p))
        if not self.tables:
            raise FileNotFoundError(f"no table CSVs found in {csv_dir}")

    @property
    def schema(self) -> Dict[str, List[str]]:
        return {t: list(tb.df.columns) for t, tb in self.tables.items()}


def _coerce(table: Table, p: Pred):
    vals = p.value if p.op in ("between", "in") else (p.value,)
    if table.numeric[p.col]:
        out = []
        for v in vals:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                raise SQLError(f'invalid input syntax for type numeric: "{v}" ({p.col})') from None
        return out
    return [str(v) for v in vals]


def pred_mask(table: Table, p: Pred) -> np.ndarray:
    col = table.cols[p.col]
    v = _coerce(table, p)
    if p.op == "eq":
        return col == v[0]
    if p.op == "ne":
        return col != v[0]
    if p.op == "in":
        return np.isin(col, np.array(v, dtype=col.dtype))
    if p.op == "between":
        return (col >= v[0]) & (col <= v[1])
    return {"lt": col < v[0], "le": col <= v[0], "gt": col > v[0], "ge": col >= v[0]}[p.op]


def filter_rows(table: Table, preds: List[Pred]) -> np.ndarray:
    """Indices of rows satisfying all `preds`."""
    m = np.ones(table.n, dtype=bool)
    for p in preds:
        m &= pred_mask(table, p)
    return np.flatnonzero(m)


def _components(q: Query) -> Tuple[List[List[str]], bool]:
    """Connected components of the join graph, and whether the graph is acyclic."""
    adj = {a: set() for a in q.rels}
    for j in q.joins:
        adj[j.a].add(j.b)
        adj[j.b].add(j.a)
    seen, comps = set(), []
    for a in q.rels:
        if a in seen:
            continue
        stack, comp = [a], []
        seen.add(a)
        while stack:
            x = stack.pop()
            comp.append(x)
            for y in adj[x] - seen:
                seen.add(y)
                stack.append(y)
        comps.append(comp)
    acyclic = len(q.joins) == len(q.rels) - len(comps)
    return comps, acyclic


class Executor:
    def __init__(self, catalog: Catalog):
        self.cat = catalog

    def _rows(self, q: Query) -> Dict[str, np.ndarray]:
        return {a: filter_rows(self.cat.tables[t], q.preds_of(a)) for a, t in q.rels.items()}

    def _key(self, q: Query, alias: str, col: str, rows: np.ndarray) -> np.ndarray:
        return self.cat.tables[q.rels[alias]].cols[col][rows]

    def count(self, q: Query) -> Tuple[int, Dict[str, int], Dict[str, int]]:
        """Exact COUNT(*). Also returns rows-after-filter per relation and the join size of each
        relation's subtree (rooted at the first relation of each component)."""
        rows = self._rows(q)
        comps, acyclic = _components(q)
        filtered = {a: int(len(r)) for a, r in rows.items()}
        if not acyclic:
            return int(len(self._materialize(q, rows))), filtered, {}
        total, w = 1.0, {}
        for comp in comps:
            total *= self._collect_weights(q, rows, comp[0], None, w).sum()
        return int(round(total)), filtered, {a: int(round(v.sum())) for a, v in w.items()}

    def fetch(self, q: Query, limit: Optional[int] = None) -> pd.DataFrame:
        """Up to `limit` result rows (LIMIT, else DEFAULT_ROW_LIMIT)."""
        limit = q.limit if q.limit is not None else (limit or DEFAULT_ROW_LIMIT)
        rows = self._rows(q)
        comps, acyclic = _components(q)
        if not acyclic:
            idx = self._materialize(q, rows).iloc[:limit]
        else:
            parts = [self._fetch_tree(q, rows, comp, limit) for comp in comps]
            idx = parts[0]
            for p in parts[1:]:  # cross product between disconnected components
                idx = idx.merge(p, how="cross").iloc[:limit]
            idx = idx.iloc[:limit]
        cols = q.select or [(a, c) for a in q.rels for c in self.cat.tables[q.rels[a]].df.columns]
        data = {}
        for a, c in cols:
            src = self.cat.tables[q.rels[a]].df[c].to_numpy()
            name = c if len(q.rels) == 1 else f"{a}.{c}"
            data[name] = src[idx[a].to_numpy()] if len(idx) else src[:0]
        return pd.DataFrame(data)

    def _fetch_tree(self, q: Query, rows, comp: List[str], limit: int) -> pd.DataFrame:
        root = comp[0]
        w = {}
        # bottom-up weights double as the "row extends to a full result" test
        self._collect_weights(q, rows, root, None, w)
        useful = {a: rows[a][w[a] > 0] for a in comp}
        # top-down: keep only child rows whose key survives in the parent
        order: List[Tuple[str, str, Join]] = []

        def descend(a, parent):
            for j in q.joins:
                if j is parent or a not in (j.a, j.b):
                    continue
                mycol, child, ccol = j.other(a)
                pk = self._key(q, a, mycol, useful[a])
                ck = self._key(q, child, ccol, useful[child])
                useful[child] = useful[child][np.isin(ck, pk)]
                order.append((a, child, j))
                descend(child, j)

        descend(root, None)
        cur = {root: useful[root][:limit]}
        for parent, child, j in order:
            mycol, _, ccol = j.other(parent)
            pk = self._key(q, parent, mycol, cur[parent])
            crow = useful[child]
            ck = self._key(q, child, ccol, crow)
            o = np.argsort(ck, kind="stable")
            ck, crow = ck[o], crow[o]
            lo, hi = np.searchsorted(ck, pk, "left"), np.searchsorted(ck, pk, "right")
            cnt = hi - lo
            keep = np.cumsum(cnt) <= limit
            if not keep.all():  # take the partial last group too
                first = int(np.argmin(keep))
                cnt = cnt.copy()
                cnt[first] = limit - int(cnt[:first].sum())
                cnt[first + 1:] = 0
            rep = np.repeat(np.arange(len(pk)), cnt)
            off = np.arange(len(rep)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
            cur = {a: v[rep] for a, v in cur.items()}
            cur[child] = crow[np.repeat(lo, cnt) + off]
        return pd.DataFrame(cur)

    def _collect_weights(self, q: Query, rows, alias: str, parent: Optional[Join], out) -> np.ndarray:
        """Row weights of `alias` = number of join results of its subtree each row takes part in."""
        wv = np.ones(len(rows[alias]), dtype=np.float64)
        for j in q.joins:
            if j is parent or alias not in (j.a, j.b):
                continue
            mycol, child, ccol = j.other(alias)
            cw = self._collect_weights(q, rows, child, j, out)
            keys = self._key(q, child, ccol, rows[child])
            msg = pd.Series(cw).groupby(keys).sum()
            pos = msg.index.get_indexer(self._key(q, alias, mycol, rows[alias]))
            wv *= np.append(msg.to_numpy(), 0.0)[pos]  # pos -1 -> appended 0 (no match)
        out[alias] = wv
        return wv

    def _materialize(self, q: Query, rows) -> pd.DataFrame:
        """Fallback for cyclic join graphs: hash-join along a spanning tree, then apply the
        cycle-closing predicates. Frame columns are per-alias row indices."""
        frames = {a: pd.DataFrame({a: r}) for a, r in rows.items()}
        cur = frames[next(iter(q.rels))]
        placed = {next(iter(q.rels))}
        pending = list(q.joins)
        while pending:
            j = next((j for j in pending if (j.a in placed) != (j.b in placed)), None)
            if j is None:
                break
            pending.remove(j)
            have, new = (j.a, j.b) if j.a in placed else (j.b, j.a)
            hcol, _, ncol = j.other(have)
            cur = cur.assign(_k=self._key(q, have, hcol, cur[have].to_numpy()))
            nf = frames[new].assign(_k=self._key(q, new, ncol, frames[new][new].to_numpy()))
            cur = cur.merge(nf, on="_k").drop(columns="_k")
            placed.add(new)
            if len(cur) > CYCLIC_ROW_GUARD:
                raise SQLError("cyclic join too large for the in-memory fallback")
        for a in q.rels:
            if a not in placed:
                cur = cur.merge(frames[a], how="cross")
                placed.add(a)
        for j in pending:
            ka = self._key(q, j.a, j.acol, cur[j.a].to_numpy())
            kb = self._key(q, j.b, j.bcol, cur[j.b].to_numpy())
            cur = cur[ka == kb]
        return cur.reset_index(drop=True)
