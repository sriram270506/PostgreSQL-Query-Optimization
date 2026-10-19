"""Dataset specs: which tables exist, how they join, which columns queries filter on.

Everything downstream (engine, BNs, FactorJoin, workload generator, featuriser) is
driven by a spec, so the same code runs on the synthetic HR schema and on the real
STATS benchmark (Stack Exchange dump, Han et al. VLDB 2022).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

Edge = Tuple[str, str, str, str]  # (table, col, table, col)


@dataclass(frozen=True)
class Dataset:
    name: str
    csv_dir: str
    files: Dict[str, str]                 # table -> csv file name
    join_edges: List[Edge]
    filters: Dict[str, List[str]]         # table -> filterable columns
    aliases: Dict[str, str]
    timestamps: Dict[str, List[str]] = field(default_factory=dict)
    workload: str = ""
    models_dir: str = ""

    @property
    def tables(self) -> List[str]:
        return list(self.files)

    def key_columns(self, table: str) -> List[str]:
        out = []
        for t1, c1, t2, c2 in self.join_edges:
            for t, c in ((t1, c1), (t2, c2)):
                if t == table and c not in out:
                    out.append(c)
        return out

    def bn_columns(self, table: str) -> List[str]:
        """Columns a query can touch: filters + join keys. Surrogate ids that are never
        joined or filtered (e.g. assignments.assign_id) carry no information and only
        add spurious, noise-level MI edges, so they are left out."""
        return list(dict.fromkeys(self.filters.get(table, []) + self.key_columns(table)))


HR = Dataset(
    name="hr",
    csv_dir="data/out",
    files={t: f"{t}.csv" for t in ["departments", "employees", "projects", "assignments", "locations", "budgets"]},
    join_edges=[
        ("departments", "dept_id", "employees", "dept_id"),
        ("departments", "dept_id", "projects", "dept_id"),
        ("employees", "dept_id", "projects", "dept_id"),
        ("employees", "emp_id", "assignments", "emp_id"),
        ("projects", "proj_id", "assignments", "proj_id"),
        ("projects", "proj_id", "budgets", "proj_id"),
        ("employees", "city", "locations", "city"),
    ],
    filters={
        "departments": ["dept_name", "region"],
        "employees": ["age", "experience_years", "salary", "city", "dept_id"],
        "projects": ["priority", "budget_tier", "dept_id"],
        "assignments": ["hours_per_week"],
        "locations": ["city", "office_tier"],
        "budgets": ["approved_amount"],
    },
    aliases={"departments": "d", "employees": "e", "projects": "p", "assignments": "a", "locations": "l", "budgets": "b"},
    workload="data/workload/workload.csv",
    models_dir="models",
)

_U = [("users", "id"), ("badges", "userid"), ("comments", "userid"), ("posthistory", "userid"),
      ("votes", "userid"), ("posts", "owneruserid")]
_P = [("posts", "id"), ("comments", "postid"), ("votes", "postid"), ("posthistory", "postid"),
      ("postlinks", "postid"), ("postlinks", "relatedpostid"), ("tags", "excerptpostid")]

STATS = Dataset(
    name="stats",
    csv_dir="data/stats/raw",
    files={"users": "users.csv", "posts": "posts.csv", "postlinks": "postLinks.csv", "posthistory": "postHistory.csv",
           "comments": "comments.csv", "votes": "votes.csv", "badges": "badges.csv", "tags": "tags.csv"},
    # every join that appears in STATS-CEB: PK-FK edges to users.id / posts.id plus the
    # FK-FK edges the benchmark uses (many-to-many, e.g. comments.userid = badges.userid)
    join_edges=[(a, b, c, d) for (a, b), (c, d) in
                [(_U[0], x) for x in _U[1:]] + [(_P[0], x) for x in _P[1:]] + [
                    (("badges", "userid"), ("comments", "userid")), (("comments", "userid"), ("posthistory", "userid")),
                    (("comments", "userid"), ("posts", "owneruserid")), (("posts", "owneruserid"), ("votes", "userid")),
                    (("badges", "userid"), ("votes", "userid")), (("posthistory", "userid"), ("votes", "userid")),
                    (("posthistory", "userid"), ("posts", "owneruserid")), (("badges", "userid"), ("posthistory", "userid")),
                    (("comments", "userid"), ("votes", "userid")), (("comments", "postid"), ("postlinks", "postid")),
                    (("badges", "userid"), ("posts", "owneruserid")), (("posthistory", "postid"), ("postlinks", "postid")),
                    (("postlinks", "postid"), ("votes", "postid")), (("comments", "postid"), ("posthistory", "postid")),
                    (("posthistory", "postid"), ("votes", "postid")),
                ]],
    filters={
        "users": ["reputation", "creationdate", "views", "upvotes", "downvotes"],
        "posts": ["posttypeid", "creationdate", "score", "viewcount", "answercount", "commentcount", "favoritecount"],
        "postlinks": ["creationdate", "linktypeid"],
        "posthistory": ["posthistorytypeid", "creationdate"],
        "comments": ["score", "creationdate"],
        "votes": ["votetypeid", "creationdate", "bountyamount"],
        "badges": ["date"],
        "tags": [],
    },
    aliases={"users": "u", "posts": "p", "postlinks": "pl", "posthistory": "ph", "comments": "c", "votes": "v",
             "badges": "b", "tags": "t"},
    timestamps={"users": ["creationdate"], "posts": ["creationdate"], "postlinks": ["creationdate"],
                "posthistory": ["creationdate"], "comments": ["creationdate"], "votes": ["creationdate"],
                "badges": ["date"]},
    workload="data/workload/stats.csv",
    models_dir="models/stats",
)

# STATS as of 2013-06-01 (scripts.make_drift): train on the old database, test on today's
STATS_OLD = Dataset(**{**STATS.__dict__, "name": "stats_old", "csv_dir": "data/stats/old",
                       "workload": "data/workload/stats_old.csv", "models_dir": "models/stats_old"})

DATASETS = {"hr": HR, "stats": STATS, "stats_old": STATS_OLD}


def get(name: str) -> Dataset:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    return DATASETS[name]
