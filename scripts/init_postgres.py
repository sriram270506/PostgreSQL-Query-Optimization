"""
Load a dataset into PostgreSQL, index its join keys and ANALYZE.
Run once after `docker compose up -d`.

Usage:
  python -m scripts.init_postgres --yes                    # HR schema  -> database carddb
  python -m scripts.init_postgres --dataset stats --yes    # STATS      -> database stats
Reads PG_DSN from .env; refuses non-localhost servers unless --allow-remote.
"""
import argparse
import os
import sys
import pandas as pd

from src.engine import datasets
from src.env import load_dotenv

load_dotenv()

try:
    import psycopg2
    from psycopg2 import sql as pg_sql
except ImportError:
    print("psycopg2-binary not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

CSV_DIR = "data/out"

DDL = {
    "departments": """
        CREATE TABLE IF NOT EXISTS departments (
            dept_id   INTEGER PRIMARY KEY,
            dept_name VARCHAR(100),
            region    VARCHAR(100)
        )
    """,
    "employees": """
        CREATE TABLE IF NOT EXISTS employees (
            emp_id           INTEGER PRIMARY KEY,
            name             VARCHAR(200),
            dept_id          INTEGER REFERENCES departments(dept_id),
            age              INTEGER,
            experience_years INTEGER,
            salary           NUMERIC,
            city             VARCHAR(100)
        )
    """,
    "projects": """
        CREATE TABLE IF NOT EXISTS projects (
            proj_id     INTEGER PRIMARY KEY,
            dept_id     INTEGER REFERENCES departments(dept_id),
            priority    VARCHAR(50),
            budget_tier INTEGER
        )
    """,
    "assignments": """
        CREATE TABLE IF NOT EXISTS assignments (
            assign_id      INTEGER PRIMARY KEY,
            emp_id         INTEGER REFERENCES employees(emp_id),
            proj_id        INTEGER REFERENCES projects(proj_id),
            hours_per_week NUMERIC
        )
    """,
    "locations": """
        CREATE TABLE IF NOT EXISTS locations (
            loc_id      INTEGER PRIMARY KEY,
            city        VARCHAR(100),
            office_tier INTEGER
        )
    """,
    "budgets": """
        CREATE TABLE IF NOT EXISTS budgets (
            budget_id       INTEGER PRIMARY KEY,
            proj_id         INTEGER REFERENCES projects(proj_id),
            approved_amount NUMERIC
        )
    """,
}

# Load order (respect FK deps)
LOAD_ORDER = ["departments", "employees", "projects", "assignments", "locations", "budgets"]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_emp_dept    ON employees(dept_id)",
    "CREATE INDEX IF NOT EXISTS idx_emp_city    ON employees(city)",
    "CREATE INDEX IF NOT EXISTS idx_emp_salary  ON employees(salary)",
    "CREATE INDEX IF NOT EXISTS idx_proj_dept   ON projects(dept_id)",
    "CREATE INDEX IF NOT EXISTS idx_proj_tier   ON projects(budget_tier)",
    "CREATE INDEX IF NOT EXISTS idx_asgn_emp    ON assignments(emp_id)",
    "CREATE INDEX IF NOT EXISTS idx_asgn_proj   ON assignments(proj_id)",
    "CREATE INDEX IF NOT EXISTS idx_asgn_hours  ON assignments(hours_per_week)",
    "CREATE INDEX IF NOT EXISTS idx_loc_city    ON locations(city)",
    "CREATE INDEX IF NOT EXISTS idx_bud_proj    ON budgets(proj_id)",
]


def _copy(cur, table: str, path: str):
    """Bulk-load a CSV with COPY (empty field = NULL)."""
    with open(path, encoding="utf8") as f:
        cur.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')", f)


def load_stats(dsn: str):
    ds = datasets.get("stats")
    admin = psycopg2.connect(dsn.rsplit("/", 1)[0] + "/postgres")
    admin.autocommit = True
    with admin.cursor() as c:
        c.execute("SELECT 1 FROM pg_database WHERE datname = 'stats'")
        if not c.fetchone():
            c.execute("CREATE DATABASE stats")
    admin.close()
    conn = psycopg2.connect(dsn.rsplit("/", 1)[0] + "/stats")
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS " + ", ".join(ds.files) + " CASCADE")
    with open(os.path.join(ds.csv_dir, "stats.sql"), encoding="utf8") as f:
        cur.execute(f.read())
    for t, f in ds.files.items():
        _copy(cur, t, os.path.join(ds.csv_dir, f))
        print(f"  loaded {t}")
    for t in ds.files:  # index every join key, as the STATS-CEB benchmark setup does
        for c in ds.key_columns(t):
            if c != "id":
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{t}_{c} ON {t}({c})")
    conn.commit()
    conn.autocommit = True
    cur.execute("ANALYZE")
    conn.close()
    print("Done: database 'stats'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", default=os.environ.get("PG_DSN"), help="defaults to PG_DSN from the environment / .env")
    parser.add_argument("--csv-dir", default=CSV_DIR)
    parser.add_argument("--dataset", default="hr", choices=["hr", "stats"])
    parser.add_argument("--yes", action="store_true", help="confirm dropping/truncating the dataset's tables")
    parser.add_argument("--allow-remote", action="store_true", help="allow a non-localhost server")
    args = parser.parse_args()
    if not args.dsn:
        sys.exit("No DSN: set PG_DSN in .env (see .env.example) or pass --dsn")
    from psycopg2.extensions import parse_dsn
    host = parse_dsn(args.dsn).get("host", "localhost")
    if host not in ("localhost", "127.0.0.1", "::1") and not args.allow_remote:
        sys.exit(f"Refusing to initialise non-local server {host!r}; pass --allow-remote if you really mean it")
    if not args.yes:
        sys.exit(f"This DROPS/TRUNCATES the {args.dataset} tables on {host}. Re-run with --yes to confirm.")

    print("Connecting to:", args.dsn.split("@")[-1])
    if args.dataset == "stats":
        load_stats(args.dsn)
        return
    conn = psycopg2.connect(args.dsn)
    conn.autocommit = False
    cur = conn.cursor()

    print("Creating tables...")
    for tbl in LOAD_ORDER:
        cur.execute(DDL[tbl])
    conn.commit()

    print("Loading CSVs...")
    for tbl in LOAD_ORDER:
        fpath = os.path.join(args.csv_dir, f"{tbl}.csv")
        if not os.path.exists(fpath):
            print(f"  [skip] {fpath} not found")
            continue
        df = pd.read_csv(fpath)
        cur.execute(f"TRUNCATE TABLE {tbl} CASCADE")
        cols = list(df.columns)
        col_str = ", ".join(cols)
        placeholders = ", ".join(["%s"] * len(cols))
        insert_sql = f"INSERT INTO {tbl} ({col_str}) VALUES ({placeholders})"
        rows = [tuple(row) for row in df.itertuples(index=False, name=None)]
        cur.executemany(insert_sql, rows)
        conn.commit()
        print(f"  Loaded {len(rows):,} rows -> {tbl}")

    print("Creating indexes...")
    for idx in INDEXES:
        cur.execute(idx)
    conn.commit()

    print("Running ANALYZE...")
    old_iso = conn.isolation_level
    conn.set_isolation_level(0)
    cur.execute("ANALYZE")
    conn.set_isolation_level(old_iso)

    cur.close()
    conn.close()

    print(f"\nDone. Set PG_DSN={args.dsn!r} in your environment to enable live EXPLAIN.")


if __name__ == "__main__":
    main()
