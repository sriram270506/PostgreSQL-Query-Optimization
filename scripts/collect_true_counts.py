"""Evaluate generated SQL COUNT queries against CSV data using pandas.

Writes output CSVs with `true_count` column appended.
"""
import argparse
import os
import re
import pandas as pd


def load_data(csv_dir):
    deps = pd.read_csv(os.path.join(csv_dir, "departments.csv"))
    emps = pd.read_csv(os.path.join(csv_dir, "employees.csv"))
    projs = pd.read_csv(os.path.join(csv_dir, "projects.csv"))
    return deps, emps, projs


def parse_between(expr):
    m = re.search(r"between\s+(\d+)\s+and\s+(\d+)", expr, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def eval_query(q, deps, emps, projs):
    qlow = q.lower()
    # single-table employees
    if "from employees" in qlow and "join" not in qlow:
        if "dept_id =" in qlow:
            v = int(re.search(r"dept_id\s*=\s*(\d+)", qlow).group(1))
            return int((emps[emps["dept_id"] == v]).shape[0])
        if "salary between" in qlow:
            lo, hi = parse_between(qlow)
            return int(emps[(emps["salary"] >= lo) & (emps["salary"] <= hi)].shape[0])
        if "age between" in qlow:
            lo, hi = parse_between(qlow)
            return int(emps[(emps["age"] >= lo) & (emps["age"] <= hi)].shape[0])
    # single-table departments
    if "from departments" in qlow and "join" not in qlow:
        m = re.search(r"where\s+city\s*=\s*'([^']+)'", q, flags=re.IGNORECASE)
        if m:
            city = m.group(1)
            return int(deps[deps["city"] == city].shape[0])
    # single-table projects
    if "from projects" in qlow and "join" not in qlow:
        m = re.search(r"where\s+budget_tier\s*=\s*(\d+)", qlow)
        if m:
            t = int(m.group(1))
            return int(projs[projs["budget_tier"] == t].shape[0])

    # employees JOIN departments
    if "from employees" in qlow and "join departments" in qlow and "join projects" not in qlow:
        m = re.search(r"where\s+d\.city\s*=\s*'([^']+)'", q, flags=re.IGNORECASE)
        lohi = re.search(r"e\.salary\s+between\s+(\d+)\s+and\s+(\d+)", qlow)
        if m and lohi:
            city = m.group(1)
            lo, hi = int(lohi.group(1)), int(lohi.group(2))
            merged = emps.merge(deps, left_on="dept_id", right_on="dept_id", suffixes=("_e","_d"))
            return int(merged[(merged["city"] == city) & (merged["salary"] >= lo) & (merged["salary"] <= hi)].shape[0])

    # employees JOIN departments JOIN projects
    if "join projects" in qlow:
        m_city = re.search(r"d\.city\s*=\s*'([^']+)'", q, flags=re.IGNORECASE)
        m_tier = re.search(r"p\.budget_tier\s*=\s*(\d+)", qlow)
        lohi = re.search(r"e\.salary\s+between\s+(\d+)\s+and\s+(\d+)", qlow)
        if m_city and m_tier and lohi:
            city = m_city.group(1)
            tier = int(m_tier.group(1))
            lo, hi = int(lohi.group(1)), int(lohi.group(2))
            merged = emps.merge(deps, on="dept_id").merge(projs, left_on="emp_id", right_on="emp_id")
            return int(merged[(merged["city"] == city) & (merged["budget_tier"] == tier) & (merged["salary"] >= lo) & (merged["salary"] <= hi)].shape[0])

    # fallback: return -1 to indicate unknown
    return -1


def process(in_queries_csv, out_csv, csv_dir):
    deps, emps, projs = load_data(csv_dir)
    df = pd.read_csv(in_queries_csv)
    counts = []
    for q in df['sql'].tolist():
        c = eval_query(q, deps, emps, projs)
        counts.append(c)
    df['true_count'] = counts
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='in_csv', required=True)
    parser.add_argument('--out', dest='out_csv', required=True)
    parser.add_argument('--csv-dir', default='data/out')
    args = parser.parse_args()
    process(args.in_csv, args.out_csv, args.csv_dir)


if __name__ == '__main__':
    main()
