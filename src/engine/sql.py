"""SQL front end: parse a SELECT into a logical Query (relations, join edges, predicates).

Supported: SELECT COUNT(*) | * | cols FROM t [alias] [JOIN t2 ON a.x = b.y ...]
WHERE <conjunction of col OP literal | col BETWEEN lo AND hi | col IN (...) | a.x = b.y>
[LIMIT n]. OR / subqueries / GROUP BY raise SQLError.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
import sqlglot.expressions as exp
from sqlglot.errors import SqlglotError


class SQLError(ValueError):
    pass


OPS = ("eq", "ne", "lt", "le", "gt", "ge", "between", "in")
_CMP = {exp.EQ: "eq", exp.NEQ: "ne", exp.LT: "lt", exp.LTE: "le", exp.GT: "gt", exp.GTE: "ge"}
_FLIP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}
_SQL_OP = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}


@dataclass(frozen=True)
class Pred:
    alias: str
    col: str
    op: str
    value: Any  # scalar; (lo, hi) for between; tuple of values for in
    ts: bool = False  # value is epoch seconds of a TIMESTAMP column

    def to_sql(self) -> str:
        c = f"{self.alias}.{self.col}"
        lit = (lambda v: _ts_lit(v)) if self.ts else _lit
        if self.op == "between":
            return f"{c} BETWEEN {lit(self.value[0])} AND {lit(self.value[1])}"
        if self.op == "in":
            return f"{c} IN ({', '.join(lit(v) for v in self.value)})"
        return f"{c} {_SQL_OP[self.op]} {lit(self.value)}"


@dataclass(frozen=True)
class Join:
    a: str
    acol: str
    b: str
    bcol: str

    def other(self, alias: str) -> Tuple[str, str, str]:
        """(my_col, other_alias, other_col) seen from `alias`."""
        return (self.acol, self.b, self.bcol) if alias == self.a else (self.bcol, self.a, self.acol)

    def to_sql(self) -> str:
        return f"{self.a}.{self.acol} = {self.b}.{self.bcol}"


@dataclass
class Query:
    rels: Dict[str, str]                      # alias -> table, in FROM order
    joins: List[Join] = field(default_factory=list)
    preds: List[Pred] = field(default_factory=list)
    count: bool = True
    select: Optional[List[Tuple[str, str]]] = None  # None + not count => SELECT *
    limit: Optional[int] = None

    def preds_of(self, alias: str) -> List[Pred]:
        return [p for p in self.preds if p.alias == alias]

    def sub(self, aliases) -> "Query":
        """COUNT(*) sub-query over a subset of relations (used by the join-order optimizer)."""
        s = set(aliases)
        return Query(
            rels={a: t for a, t in self.rels.items() if a in s},
            joins=[j for j in self.joins if j.a in s and j.b in s],
            preds=[p for p in self.preds if p.alias in s],
        )

    def canonical(self) -> "Query":
        """Same query, independent of how it was written: relations ordered by (table,
        alias), each join edge oriented a<b and sorted, predicates sorted. Learned models
        featurise this form, so FROM order / ON-side order cannot change an estimate."""
        rels = dict(sorted(self.rels.items(), key=lambda kv: (kv[1], kv[0])))
        order = {a: i for i, a in enumerate(rels)}
        joins = sorted((j if order[j.a] <= order[j.b] else Join(j.b, j.bcol, j.a, j.acol) for j in self.joins),
                       key=lambda j: (order[j.a], order[j.b], j.acol, j.bcol))
        preds = sorted(self.preds, key=lambda p: (order[p.alias], p.col, p.op, str(p.value)))
        return Query(rels=rels, joins=joins, preds=preds, count=self.count, select=self.select, limit=self.limit)

    def key(self) -> str:
        return "|".join(sorted(self.rels)) + "#" + ";".join(sorted(p.to_sql() for p in self.preds))

    def to_sql(self) -> str:
        if self.count:
            head = "COUNT(*)"
        elif self.select:
            head = ", ".join(f"{a}.{c}" for a, c in self.select)
        else:
            head = "*"
        aliases = list(self.rels)
        sql = f"SELECT {head} FROM {self.rels[aliases[0]]} {aliases[0]}"
        placed, pending = {aliases[0]}, list(self.joins)
        # emit joins in an order where each ON references an already-placed alias
        while pending:
            for j in pending:
                if (j.a in placed) != (j.b in placed):
                    new = j.b if j.a in placed else j.a
                    sql += f" JOIN {self.rels[new]} {new} ON {j.to_sql()}"
                    placed.add(new)
                    pending.remove(j)
                    break
            else:
                break
        extra = [j.to_sql() for j in pending]  # cycle-closing edges
        for a in aliases:
            if a not in placed:
                sql += f", {self.rels[a]} {a}"
        conds = extra + [p.to_sql() for p in self.preds]
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        if self.limit is not None:
            sql += f" LIMIT {self.limit}"
        return sql


def _lit(v: Any) -> str:
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return str(int(v)) if float(v).is_integer() else str(v)


def _ts_lit(v) -> str:
    return "'" + datetime.fromtimestamp(int(v), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + "'::timestamp"


def _literal(node: exp.Expression) -> Any:
    if isinstance(node, exp.Cast):  # '2014-09-11 08:55:52'::timestamp -> epoch seconds
        inner = _literal(node.this)
        if isinstance(inner, str) and node.to.this in (exp.DataType.Type.TIMESTAMP, exp.DataType.Type.DATE,
                                                       exp.DataType.Type.TIMESTAMPTZ, exp.DataType.Type.DATETIME):
            try:
                fmt = "%Y-%m-%d %H:%M:%S" if len(inner) > 10 else "%Y-%m-%d"
                return int(datetime.strptime(inner[:19], fmt).replace(tzinfo=timezone.utc).timestamp())
            except ValueError:
                raise SQLError(f"invalid timestamp literal: {inner}") from None
        return inner
    if isinstance(node, exp.Neg):
        return -_literal(node.this)
    if isinstance(node, exp.Paren):
        return _literal(node.this)
    if not isinstance(node, exp.Literal):
        raise SQLError(f"expected a literal, got: {node.sql()}")
    if node.is_string:
        return node.this
    num = float(node.this)
    return int(num) if num.is_integer() else num


def parse(sql: str, schema: Dict[str, List[str]]) -> Query:
    """Parse `sql` against `schema` (table -> column names)."""
    try:
        tree = sqlglot.parse_one(sql.strip().rstrip(";"), dialect="postgres")
    except SqlglotError as e:
        raise SQLError(f"syntax error: {str(e).splitlines()[0]}") from None
    if not isinstance(tree, exp.Select):
        raise SQLError("only SELECT statements are supported")
    for bad in ("group", "having", "order"):
        if tree.args.get(bad):
            raise SQLError(f"{bad.upper()} BY is not supported")
    if tree.find(exp.Subquery) or tree.find(exp.Or) or tree.find(exp.Union):
        raise SQLError("subqueries / OR / UNION are not supported")

    rels: Dict[str, str] = {}

    def add_rel(t: exp.Table):
        name = t.name.lower()
        if name not in schema:
            raise SQLError(f'relation "{name}" does not exist')
        alias = (t.alias or name).lower()
        if alias in rels:
            raise SQLError(f'table alias "{alias}" specified more than once')
        rels[alias] = name

    from_ = tree.args.get("from_") or tree.args.get("from")
    if from_ is None:
        raise SQLError("missing FROM clause")
    for t in from_.find_all(exp.Table):
        add_rel(t)
    conds: List[exp.Expression] = []
    for j in tree.args.get("joins") or []:
        add_rel(j.this)
        if j.args.get("on") is not None:
            conds.append(j.args["on"])

    def resolve(col: exp.Column) -> Tuple[str, str]:
        name, tab = col.name.lower(), (col.table or "").lower()
        if tab:
            if tab not in rels:
                raise SQLError(f'missing FROM-clause entry for table "{tab}"')
            if name not in schema[rels[tab]]:
                raise SQLError(f'column {tab}.{name} does not exist')
            return tab, name
        owners = [a for a, t in rels.items() if name in schema[t]]
        if not owners:
            raise SQLError(f'column "{name}" does not exist')
        if len(owners) > 1:
            raise SQLError(f'column reference "{name}" is ambiguous')
        return owners[0], name

    where = tree.args.get("where")
    if where is not None:
        conds.append(where.this)

    joins: List[Join] = []
    preds: List[Pred] = []

    def walk(node: exp.Expression):
        if isinstance(node, exp.Paren):
            return walk(node.this)
        if isinstance(node, exp.And):
            walk(node.left)
            walk(node.right)
            return
        if isinstance(node, exp.Between):
            a, c = resolve(node.this)
            preds.append(Pred(a, c, "between", (_literal(node.args["low"]), _literal(node.args["high"]))))
            return
        if isinstance(node, exp.In):
            a, c = resolve(node.this)
            preds.append(Pred(a, c, "in", tuple(_literal(v) for v in node.expressions)))
            return
        if type(node) in _CMP:
            op = _CMP[type(node)]
            l, r = node.left, node.right
            if isinstance(l, exp.Column) and isinstance(r, exp.Column):
                if op != "eq":
                    raise SQLError("only equi-joins are supported")
                (a, ac), (b, bc) = resolve(l), resolve(r)
                if a == b:
                    raise SQLError("same-table column comparison is not supported")
                joins.append(Join(a, ac, b, bc))
                return
            if isinstance(r, exp.Column):
                l, r, op = r, l, _FLIP[op]
            if not isinstance(l, exp.Column):
                raise SQLError(f"unsupported condition: {node.sql()}")
            a, c = resolve(l)
            preds.append(Pred(a, c, op, _literal(r)))
            return
        raise SQLError(f"unsupported condition: {node.sql()}")

    for c in conds:
        walk(c)

    q = Query(rels=rels, joins=joins, preds=preds)
    sel = tree.expressions
    if len(sel) == 1 and isinstance(sel[0], exp.Count):
        q.count = True
    else:
        q.count = False
        if any(isinstance(s, exp.Count) for s in sel):
            raise SQLError("COUNT(*) cannot be mixed with other columns")
        if not (len(sel) == 1 and isinstance(sel[0], exp.Star)):
            cols = []
            for s in sel:
                if isinstance(s, exp.Column) and isinstance(s.this, exp.Star):
                    t = (s.table or "").lower()
                    cols += [(t, c) for c in schema[rels[t]]]
                elif isinstance(s, exp.Column):
                    cols.append(resolve(s))
                else:
                    raise SQLError(f"unsupported select item: {s.sql()}")
            q.select = cols
    lim = tree.args.get("limit")
    if lim is not None:
        q.limit = int(_literal(lim.expression))
    return q
