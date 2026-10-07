import os
import yaml
import numpy as np
import pandas as pd
from faker import Faker
from typing import Dict, Any

class CorrelatedDataGenerator:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.schema = self.config['schema']
        self.corr_config = self.config['correlation']
        self.seed = self.config['project']['seed']
        self.rng = np.random.RandomState(self.seed)
        self.fake = Faker()
        Faker.seed(self.seed)
        self.out_dir = self.config['project']['out_dir']
        os.makedirs(self.out_dir, exist_ok=True)

    def generate_departments(self) -> pd.DataFrame:
        depts = ['Engineering', 'Sales', 'HR', 'Operations', 'Marketing', 'Finance']
        regions = ['South', 'North', 'West', 'East', 'South', 'West']
        df = pd.DataFrame({
            'dept_id': list(range(1, len(depts) + 1)),
            'dept_name': depts,
            'region': regions
        })
        return df

    def generate_employees(self, depts_df: pd.DataFrame) -> pd.DataFrame:
        n_rows = self.schema['tables']['employees']['num_rows']
        dept_ids = depts_df['dept_id'].values
        dept_names = depts_df['dept_name'].values
        dept_map = dict(zip(dept_ids, dept_names))

        assigned_dept_ids = self.rng.choice(dept_ids, size=n_rows)
        names = [self.fake.name() for _ in range(n_rows)]
        ages = self.rng.randint(22, 65, size=n_rows)
        # Experience correlates with age
        experiences = np.maximum(0, ages - 22 - self.rng.randint(0, 5, size=n_rows))
        # Salary correlates with experience and age
        base_salary = 35000 + experiences * 3500 + (ages - 22) * 1200
        noise = self.rng.normal(0, 8000, size=n_rows)
        salaries = np.clip((base_salary + noise).astype(int), 30000, 250000)

        # Injected Same-Table Correlation: City depends on Department
        city_probs = self.corr_config['same_table']['employees']['dept_to_city']
        cities = []
        for d_id in assigned_dept_ids:
            d_name = dept_map[d_id]
            if d_name in city_probs:
                c_choices = list(city_probs[d_name].keys())
                c_p = list(city_probs[d_name].values())
                cities.append(self.rng.choice(c_choices, p=c_p))
            else:
                cities.append(self.rng.choice(['Bangalore', 'Delhi', 'Mumbai', 'Hyderabad', 'Chennai']))

        df = pd.DataFrame({
            'emp_id': list(range(1001, 1001 + n_rows)),
            'name': names,
            'dept_id': assigned_dept_ids,
            'age': ages,
            'experience_years': experiences,
            'salary': salaries,
            'city': cities
        })
        return df

    def generate_projects(self, depts_df: pd.DataFrame, emps_df: pd.DataFrame) -> pd.DataFrame:
        n_rows = self.schema['tables']['projects']['num_rows']
        dept_ids = depts_df['dept_id'].values

        assigned_dept_ids = self.rng.choice(dept_ids, size=n_rows)
        priorities = self.rng.choice(['Low', 'Medium', 'High', 'Critical'], size=n_rows, p=[0.3, 0.4, 0.2, 0.1])
        
        # Cross-table correlation target: P(budget_tier=5 | Bangalore) = 0.60
        budget_tiers = []
        dept_blr_ratio = emps_df.groupby('dept_id')['city'].apply(lambda s: (s == 'Bangalore').mean()).to_dict()

        for d_id in assigned_dept_ids:
            blr_ratio = dept_blr_ratio.get(d_id, 0.0)
            if blr_ratio > 0.5:
                budget_tiers.append(self.rng.choice([5, 4, 3, 2, 1], p=[0.82, 0.10, 0.04, 0.02, 0.02]))
            elif blr_ratio > 0.1:
                budget_tiers.append(self.rng.choice([5, 4, 3, 2, 1], p=[0.30, 0.30, 0.20, 0.10, 0.10]))
            else:
                budget_tiers.append(self.rng.choice([1, 2, 3, 4, 5], p=[0.50, 0.30, 0.10, 0.05, 0.05]))

        df = pd.DataFrame({
            'proj_id': list(range(5001, 5001 + n_rows)),
            'dept_id': assigned_dept_ids,
            'priority': priorities,
            'budget_tier': budget_tiers
        })
        return df

    def generate_assignments(self, emps_df: pd.DataFrame, projs_df: pd.DataFrame) -> pd.DataFrame:
        n_rows = self.schema['tables']['assignments']['num_rows']
        emp_ids = emps_df['emp_id'].values
        proj_ids = projs_df['proj_id'].values

        assigned_emps = self.rng.choice(emp_ids, size=n_rows)
        assigned_projs = self.rng.choice(proj_ids, size=n_rows)
        hours = self.rng.randint(5, 50, size=n_rows)

        df = pd.DataFrame({
            'assign_id': list(range(10001, 10001 + n_rows)),
            'emp_id': assigned_emps,
            'proj_id': assigned_projs,
            'hours_per_week': hours
        })
        return df

    def generate_locations(self) -> pd.DataFrame:
        n_rows = self.schema['tables']['locations']['num_rows']
        cities = self.rng.choice(['Bangalore', 'Delhi', 'Mumbai', 'Hyderabad', 'Pune', 'Chennai'], size=n_rows)
        loc_corr = self.corr_config['same_table']['locations']['city_to_office_tier']

        office_tiers = []
        for city in cities:
            if city in loc_corr:
                tiers = list(loc_corr[city].keys())
                probs = list(loc_corr[city].values())
                office_tiers.append(self.rng.choice(tiers, p=probs))
            else:
                office_tiers.append(self.rng.choice([1, 2, 3]))

        df = pd.DataFrame({
            'loc_id': list(range(2001, 2001 + n_rows)),
            'city': cities,
            'office_tier': office_tiers
        })
        return df

    def generate_budgets(self, projs_df: pd.DataFrame) -> pd.DataFrame:
        n_rows = len(projs_df)
        proj_ids = projs_df['proj_id'].values
        tiers = projs_df['budget_tier'].values

        approved_amounts = []
        for tier in tiers:
            base = tier * 150000
            noise = self.rng.normal(0, 20000)
            approved_amounts.append(int(np.clip(base + noise, 10000, 1000000)))

        df = pd.DataFrame({
            'budget_id': list(range(3001, 3001 + n_rows)),
            'proj_id': proj_ids,
            'approved_amount': approved_amounts
        })
        return df

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        print("Generating synthetic 6-table correlated dataset...")
        depts = self.generate_departments()
        emps = self.generate_employees(depts)
        projs = self.generate_projects(depts, emps)
        assigns = self.generate_assignments(emps, projs)
        locs = self.generate_locations()
        budgets = self.generate_budgets(projs)

        dfs = {
            'departments': depts,
            'employees': emps,
            'projects': projs,
            'assignments': assigns,
            'locations': locs,
            'budgets': budgets
        }

        for name, df in dfs.items():
            out_path = os.path.join(self.out_dir, f"{name}.csv")
            df.to_csv(out_path, index=False)
            print(f"Saved {name}.csv ({len(df)} rows)")

        return dfs

    def validate_correlations(self, dfs: Dict[str, pd.DataFrame]):
        """Hand-validation of injected correlations against ground truth"""
        print("\n--- GROUND TRUTH CORRELATION VALIDATION ---")
        emps = dfs['employees']
        depts = dfs['departments']
        projs = dfs['projects']

        # 1. Validate Same-Table: P(city | Engineering)
        eng_dept_id = depts[depts['dept_name'] == 'Engineering']['dept_id'].values[0]
        eng_cities = emps[emps['dept_id'] == eng_dept_id]['city'].value_counts(normalize=True)
        bangalore_prop = eng_cities.get('Bangalore', 0.0)
        print(f"P(city=Bangalore | dept=Engineering): {bangalore_prop:.3f} (Injected Target: 0.850)")
        assert 0.75 <= bangalore_prop <= 0.95, f"Validation failed: expected ~0.850, got {bangalore_prop:.3f}"

        # 2. Validate Cross-Table: P(budget_tier=5 | Bangalore employees' dept)
        merged = pd.merge(emps, projs, on='dept_id')
        blr_projs = merged[merged['city'] == 'Bangalore']['budget_tier'].value_counts(normalize=True)
        tier5_prop = blr_projs.get(5, 0.0)
        print(f"P(budget_tier=5 | employee_city=Bangalore): {tier5_prop:.3f} (Injected Target: 0.600)")
        assert 0.50 <= tier5_prop <= 0.75, f"Validation failed: expected ~0.600, got {tier5_prop:.3f}"
        print("[OK] All Correlation Ground-Truth Validation Checks PASSED!\n")


if __name__ == '__main__':
    gen = CorrelatedDataGenerator()
    dfs = gen.generate_all()
    gen.validate_correlations(dfs)
