"""Snapshot STATS as of a cutoff date (the 'old' database) for the data-drift experiment.

    python -m scripts.make_drift --cutoff 2013-06-01

Every row created on/after the cutoff is removed (tables without a date, i.e. tags, are
kept). Models trained on this snapshot are then evaluated on the full, newer data, the
way a deployed estimator ages as the database grows.
"""
import argparse
import os

import pandas as pd

from src.engine import datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2013-06-01")
    args = ap.parse_args()
    src, dst = datasets.get("stats"), datasets.get("stats_old")
    os.makedirs(dst.csv_dir, exist_ok=True)
    cut = pd.Timestamp(args.cutoff)
    for t, f in src.files.items():
        df = pd.read_csv(os.path.join(src.csv_dir, f))
        cols = {c.lower(): c for c in df.columns}
        ts = src.timestamps.get(t, [])
        before = len(df)
        if ts:
            d = pd.to_datetime(df[cols[ts[0]]])
            df = df[d < cut]
        df.to_csv(os.path.join(dst.csv_dir, f), index=False)
        print(f"{t}: {before} -> {len(df)} rows ({len(df) / before:.0%})")


if __name__ == "__main__":
    main()
