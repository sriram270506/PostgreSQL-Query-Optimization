import re
import numpy as np

def extract_simple_features(sql_text: str) -> np.ndarray:
    """Very small feature extractor for SQL text.

    Features:
    - number of tables (count of FROM/JOIN)
    - number of predicates (WHERE, AND, OR)
    - number of selected columns (approx by commas in SELECT)
    """
    t = sql_text.lower()
    num_tables = len(re.findall(r"\bfrom\b|\bjoin\b", t))
    num_preds = len(re.findall(r"\bwhere\b|\band\b|\bor\b", t))
    select_clause = re.search(r"select(.*?)from", t, re.S)
    if select_clause:
        cols = select_clause.group(1)
        num_cols = cols.count(",") + 1 if cols.strip() else 0
    else:
        num_cols = 0
    return np.array([num_tables, num_preds, num_cols], dtype=float)
