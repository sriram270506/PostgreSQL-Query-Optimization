"""Generate SQL query workloads (train/test) from generated CSV data.

Produces CSVs with columns: query_id, sql

Optional: execute queries against a local Postgres to collect true counts when
`--to-db` is passed and a `--conn` string is provided.
"""
import argparse
import csv
import os
import random
import textwrap
import math
import psycopg2
import numpy as np
import pandas as pd


def load_data(csv_dir):
    deps = pd.read_csv(os.path.join(csv_dir, "departments.csv"))
    emps = pd.read_csv(os.path.join(csv_dir, "employees.csv"))
    projs = pd.read_csv(os.path.join(csv_dir, "projects.csv"))
    return deps, emps, projs


def sample_salary_range(emp_salaries, width_pct=0.1):
    p = np.random.rand()
    low = np.quantile(emp_salaries, p)
    high = low + width_pct * (emp_salaries.max() - emp_salaries.min())
    return int(low), int(min(high, emp_salaries.max()))


def sample_age_range(emp_ages, width=5):
    low = int(np.random.choice(emp_ages))
    return low, low + width


def gen_single_table(emps, deps, projs):
    t = random.choice(["employees", "departments", "projects"])
    if t == "employees":
        choice = random.choice(["dept_id_eq", "salary_range", "age_range"]) 
        if choice == "dept_id_eq":
            dept = int(random.choice(deps["dept_id"].values))
            sql = f"SELECT COUNT(*) FROM employees WHERE dept_id = {dept};"
        elif choice == "salary_range":
            low, high = sample_salary_range(emps["salary"].values)
            sql = f"SELECT COUNT(*) FROM employees WHERE salary BETWEEN {low} AND {high};"
        else:
            low, high = sample_age_range(emps["age"].values)
            sql = f"SELECT COUNT(*) FROM employees WHERE age BETWEEN {low} AND {high};"
    elif t == "departments":
        city = random.choice(deps["city"].values.tolist())
        sql = f"SELECT COUNT(*) FROM departments WHERE city = '{city}';"
    else:
        tier = int(random.choice(projs["budget_tier"].values))
        sql = f"SELECT COUNT(*) FROM projects WHERE budget_tier = {tier};"
    return sql


def gen_join_2(emps, deps, projs):
    # employees JOIN departments
    city = random.choice(deps["city"].values.tolist())
    low, high = sample_salary_range(emps["salary"].values)
    sql = textwrap.dedent(f"""
        SELECT COUNT(*)
        FROM employees e
        JOIN departments d ON e.dept_id = d.dept_id
        WHERE d.city = '{city}' AND e.salary BETWEEN {low} AND {high};
    """)
    return sql.strip()


def gen_join_3(emps, deps, projs):
    # employees JOIN departments JOIN projects
    city = random.choice(deps["city"].values.tolist())
    tier = int(random.choice(projs["budget_tier"].values))
    low, high = sample_salary_range(emps["salary"].values)
    sql = textwrap.dedent(f"""
        SELECT COUNT(*)
        FROM employees e
        JOIN departments d ON e.dept_id = d.dept_id
        JOIN projects p ON p.emp_id = e.emp_id
        WHERE d.city = '{city}' AND p.budget_tier = {tier} AND e.salary BETWEEN {low} AND {high};
    """)
    return sql.strip()


def generate(n, deps, emps, projs):
    queries = []
    for i in range(n):
        r = random.random()
        if r < 0.5:
            q = gen_single_table(emps, deps, projs)
        elif r < 0.85:
            q = gen_join_2(emps, deps, projs)
        else:
            q = gen_join_3(emps, deps, projs)
        queries.append(q)
    return queries


def run_against_db(conn, queries):
    cur = conn.cursor()
    results = []
    for q in queries:
        cur.execute(q)
        cnt = cur.fetchone()[0]
        results.append(int(cnt))
    return results


def write_csv(out_path, queries, counts=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        if counts is None:
            w.writerow(["query_id", "sql"])
            for i, q in enumerate(queries, start=1):
                w.writerow([i, q])
        else:
            w.writerow(["query_id", "sql", "true_count"])
            for i, (q, c) in enumerate(zip(queries, counts), start=1):
                w.writerow([i, q, c])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--test", type=int, default=500)
    parser.add_argument("--csv-dir", default="data/out")
    parser.add_argument("--out-dir", default="data/queries")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--to-db", action="store_true")
    parser.add_argument("--conn", default="postgresql://postgres:postgres@localhost:5432/postgres")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    deps, emps, projs = load_data(args.csv_dir)

    train_q = generate(args.train, deps, emps, projs)
    test_q = generate(args.test, deps, emps, projs)

    if args.to_db:
        conn = psycopg2.connect(args.conn)
        train_counts = run_against_db(conn, train_q)
        test_counts = run_against_db(conn, test_q)
        conn.close()
    else:
        train_counts = None
        test_counts = None

    write_csv(os.path.join(args.out_dir, "train_queries.csv"), train_q, train_counts)
    write_csv(os.path.join(args.out_dir, "test_queries.csv"), test_q, test_counts)
    print(f"Wrote {len(train_q)} train and {len(test_q)} test queries to {args.out_dir}")


if __name__ == "__main__":
    main()
