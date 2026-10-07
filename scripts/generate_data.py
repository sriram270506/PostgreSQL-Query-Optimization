"""Generate synthetic correlated data and optionally load into Postgres.

Usage:
  python scripts/generate_data.py --rows 10000 --corr 0.9 --to-db

This script creates Departments, Employees, Projects with injected cross-table
correlation between department city and project budget_tier.
"""
import argparse
import csv
import random
from faker import Faker
import numpy as np
import os

try:
    import psycopg2
except Exception:
    psycopg2 = None

fake = Faker()

def generate(rows=5000, corr=0.8, out_dir="data/out"):
    os.makedirs(out_dir, exist_ok=True)
    # create a small set of cities and departments
    cities = ["Bangalore", "Seattle", "Austin", "Berlin"]
    depts = ["Eng", "Sales", "HR", "Ops"]

    # Departments: assign a city to each dept with some randomness
    departments = []
    for i, d in enumerate(depts, start=1):
        # pick a primary city biased by i
        city = cities[i % len(cities)]
        departments.append({"dept_id": i, "dept_name": d, "city": city})

    # Employees: sample department, salary/age
    employees = []
    emp_id = 1
    for i in range(rows):
        dept = random.choice(departments)
        employees.append({
            "emp_id": emp_id,
            "name": fake.name(),
            "dept_id": dept["dept_id"],
            "age": random.randint(22, 60),
            "salary": random.randint(40000, 200000),
        })
        emp_id += 1

    # Projects: correlate budget_tier with employee's department city.
    # We'll create a mapping from city -> budget distribution depending on corr
    base_probs = {
        "Bangalore": [0.6, 0.3, 0.1],
        "Seattle": [0.2, 0.5, 0.3],
        "Austin": [0.3, 0.5, 0.2],
        "Berlin": [0.4, 0.4, 0.2],
    }

    projects = []
    project_id = 1
    for emp in employees:
        # lookup employee dept city
        dept = next(d for d in departments if d["dept_id"] == emp["dept_id"])
        city = dept["city"]
        probs = base_probs[city]
        # mix with uniform to control correlation strength
        uniform = [1/3,1/3,1/3]
        mixed = [corr*p + (1-corr)*u for p,u in zip(probs, uniform)]
        tier = np.random.choice([1,2,3], p=mixed)
        projects.append({"project_id": project_id, "emp_id": emp["emp_id"], "budget_tier": int(tier)})
        project_id += 1

    # write CSVs
    with open(os.path.join(out_dir, "departments.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dept_id","dept_name","city"])
        w.writeheader(); w.writerows(departments)

    with open(os.path.join(out_dir, "employees.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["emp_id","name","dept_id","age","salary"])
        w.writeheader(); w.writerows(employees)

    with open(os.path.join(out_dir, "projects.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["project_id","emp_id","budget_tier"])
        w.writeheader(); w.writerows(projects)

    return os.path.abspath(out_dir)

def load_into_db(conninfo, csv_dir):
    if psycopg2 is None:
        raise RuntimeError("psycopg2 not installed in the environment")
    conn = psycopg2.connect(conninfo)
    cur = conn.cursor()
    # execute schema
    schema_sql = open("data/generate_schema.sql","r").read()
    cur.execute(schema_sql)
    conn.commit()
    # copy csvs
    with open(os.path.join(csv_dir, "departments.csv"), "r") as f:
        cur.copy_expert("COPY departments(dept_id,dept_name,city) FROM STDIN WITH CSV HEADER", f)
    with open(os.path.join(csv_dir, "employees.csv"), "r") as f:
        cur.copy_expert("COPY employees(emp_id,name,dept_id,age,salary) FROM STDIN WITH CSV HEADER", f)
    with open(os.path.join(csv_dir, "projects.csv"), "r") as f:
        cur.copy_expert("COPY projects(project_id,emp_id,budget_tier) FROM STDIN WITH CSV HEADER", f)
    conn.commit()
    cur.close(); conn.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--corr", type=float, default=0.9)
    parser.add_argument("--out-dir", default="data/out")
    parser.add_argument("--to-db", action="store_true")
    parser.add_argument("--conn", default="postgresql://postgres:postgres@localhost:5432/postgres")
    args = parser.parse_args()
    out = generate(rows=args.rows, corr=args.corr, out_dir=args.out_dir)
    print("Wrote CSVs to", out)
    if args.to_db:
        load_into_db(args.conn, out)
        print("Loaded into Postgres")

if __name__ == "__main__":
    main()
