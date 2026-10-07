-- Schema for synthetic dataset: employees, departments, projects
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS departments;
DROP TABLE IF EXISTS projects;

CREATE TABLE departments (
  dept_id SERIAL PRIMARY KEY,
  dept_name TEXT,
  city TEXT
);

CREATE TABLE employees (
  emp_id SERIAL PRIMARY KEY,
  name TEXT,
  dept_id INT REFERENCES departments(dept_id),
  age INT,
  salary INT
);

CREATE TABLE projects (
  project_id SERIAL PRIMARY KEY,
  emp_id INT REFERENCES employees(emp_id),
  budget_tier INT -- 1 low, 2 med, 3 high
);
