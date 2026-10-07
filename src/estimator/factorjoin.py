import os
import re
import time
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any

class FactorJoin:
    """Multi-table FactorJoin Estimator supporting 6-table correlated dataset"""
    def __init__(self, csv_dir="data/out"):
        self.csv_dir = csv_dir
        self.tables = {}
        table_names = ['departments', 'employees', 'projects', 'assignments', 'locations', 'budgets']
        for tbl in table_names:
            fpath = os.path.join(csv_dir, f"{tbl}.csv")
            if os.path.exists(fpath):
                self.tables[tbl] = pd.read_csv(fpath)

        self.emps = self.tables.get('employees', pd.DataFrame())
        self.deps = self.tables.get('departments', pd.DataFrame())
        self.projs = self.tables.get('projects', pd.DataFrame())
        self.assigns = self.tables.get('assignments', pd.DataFrame())
        self.locs = self.tables.get('locations', pd.DataFrame())
        self.budgets = self.tables.get('budgets', pd.DataFrame())

        self.tot_emps = len(self.emps) if len(self.emps) > 0 else 5000
        self.tot_deps = len(self.deps) if len(self.deps) > 0 else 6
        self.tot_projs = len(self.projs) if len(self.projs) > 0 else 5000

        # Build 2D Contingency Table Bridge: (dept_id, city) -> budget_tier
        if len(self.emps) > 0 and len(self.projs) > 0:
            merged = pd.merge(self.emps, self.projs, on='dept_id')
            contingency = merged.groupby(['dept_id', 'city', 'budget_tier']).size().unstack(fill_value=0)
            self.contingency_probs = contingency.div(contingency.sum(axis=1), axis=0)
        else:
            self.contingency_probs = None

    def per_table_selectivities(self, sql: str) -> Dict[str, float]:
        q = sql.lower()
        selectivities = {
            's_e': 1.0,
            's_d': 1.0,
            's_p': 1.0,
            's_a': 1.0,
            's_l': 1.0,
            's_b': 1.0
        }

        # Employees filters
        if "employees" in q:
            m_dept = re.search(r"dept_id\s*=\s*(\d+)", q)
            if m_dept:
                v = int(m_dept.group(1))
                selectivities['s_e'] *= float((self.emps["dept_id"] == v).mean())
            m_salary = re.search(r"salary\s+between\s+(\d+)\s+and\s+(\d+)", q)
            if m_salary:
                lo, hi = int(m_salary.group(1)), int(m_salary.group(2))
                selectivities['s_e'] *= float(((self.emps["salary"] >= lo) & (self.emps["salary"] <= hi)).mean())
            m_city = re.search(r"city\s*=\s*'([^']+)'", sql, flags=re.IGNORECASE)
            if m_city and "employees" in q:
                c = m_city.group(1)
                selectivities['s_e'] *= float((self.emps["city"].str.lower() == c.lower()).mean())

        # Departments filters
        if "departments" in q:
            m_dname = re.search(r"dept_name\s*=\s*'([^']+)'", sql, flags=re.IGNORECASE)
            if m_dname:
                selectivities['s_d'] *= float((self.deps["dept_name"].str.lower() == m_dname.group(1).lower()).mean())

        # Projects filters
        if "projects" in q:
            m_tier = re.search(r"budget_tier\s*=\s*(\d+)", q)
            if m_tier:
                t = int(m_tier.group(1))
                selectivities['s_p'] *= float((self.projs["budget_tier"] == t).mean())

        # Assignments filters
        if "assignments" in q:
            m_hours = re.search(r"hours_per_week\s+between\s+(\d+)\s+and\s+(\d+)", q)
            if m_hours:
                lo, hi = int(m_hours.group(1)), int(m_hours.group(2))
                selectivities['s_a'] *= float(((self.assigns["hours_per_week"] >= lo) & (self.assigns["hours_per_week"] <= hi)).mean())

        # Locations filters
        if "locations" in q:
            m_off = re.search(r"office_tier\s*=\s*(\d+)", q)
            if m_off:
                ot = int(m_off.group(1))
                selectivities['s_l'] *= float((self.locs["office_tier"] == ot).mean())

        # Budgets filters
        if "budgets" in q:
            m_amt = re.search(r"approved_amount\s+between\s+(\d+)\s+and\s+(\d+)", q)
            if m_amt:
                lo, hi = int(m_amt.group(1)), int(m_amt.group(2))
                selectivities['s_b'] *= float(((self.budgets["approved_amount"] >= lo) & (self.budgets["approved_amount"] <= hi)).mean())

        return selectivities

    def conditional_probs(self, sql: str) -> Tuple[float, float]:
        q = sql.lower()
        p_dept_given_emp = 1.0
        p_proj_given_emp = 1.0

        m_city = re.search(r"city\s*=\s*'([^']+)'", sql, flags=re.IGNORECASE)
        m_tier = re.search(r"budget_tier\s*=\s*(\d+)", q)

        if m_city and m_tier and self.contingency_probs is not None:
            city_val = m_city.group(1)
            tier_val = int(m_tier.group(1))
            try:
                # Query contingency bridge P(budget_tier | city)
                matching = self.contingency_probs.xs(tier_val, level='budget_tier')
                p_proj_given_emp = float(matching.mean().mean())
            except Exception:
                p_proj_given_emp = 0.2

        return p_dept_given_emp, p_proj_given_emp

    def estimate(self, sql: str) -> Tuple[float, float]:
        t0 = time.perf_counter()
        sel = self.per_table_selectivities(sql)
        p_dept, p_proj = self.conditional_probs(sql)

        q = sql.lower()
        if "from employees" in q:
            est = self.tot_emps * sel['s_e'] * sel['s_d'] * sel['s_p'] * sel['s_a'] * sel['s_l'] * sel['s_b'] * p_proj
        elif "from projects" in q:
            est = self.tot_projs * sel['s_p'] * sel['s_d'] * sel['s_b']
        else:
            est = self.tot_emps * sel['s_e'] * sel['s_p'] * p_proj

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return float(max(0.0, est)), round(latency_ms, 3)

    def extract_features(self, sql: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        sel = self.per_table_selectivities(sql)
        p_dept, p_proj = self.conditional_probs(sql)
        est, _ = self.estimate(sql)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "s_e": sel['s_e'],
            "s_d": sel['s_d'],
            "s_p": sel['s_p'],
            "s_a": sel['s_a'],
            "s_l": sel['s_l'],
            "s_b": sel['s_b'],
            "p_dept_given_emp": p_dept,
            "p_proj_given_emp": p_proj,
            "factorjoin_est": est,
            "latency_ms": round(latency_ms, 3)
        }
