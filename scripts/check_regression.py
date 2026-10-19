"""Fail (exit 1) if a fresh benchmark is worse than the committed one.

    python -m scripts.benchmark --dataset hr --quick --out bench_ci.json
    python -m scripts.check_regression --current bench_ci.json --baseline models/benchmark.json

Compares median and p99 Q-error of every estimator present in both files; a metric may
get at most `--tolerance` (default 10%) worse. The quick run uses a 300-query subsample,
so the baseline is re-scored on the same subsample size by comparing like with like:
the committed file's quick section if present, else its full-test numbers.
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--tolerance", type=float, default=0.10)
    args = ap.parse_args()
    cur = json.load(open(args.current))["accuracy"]
    base_doc = json.load(open(args.baseline))
    base = base_doc.get("quick_accuracy", base_doc["accuracy"])
    failed = False
    for name in sorted(set(cur) & set(base)):
        if name.startswith("postgres_"):
            continue  # needs a live server; not part of CI
        for metric in ("median", "p99"):
            c, b = cur[name]["overall"][metric], base[name]["overall"][metric]
            ok = c <= b * (1 + args.tolerance)
            failed |= not ok
            print(f"{'OK  ' if ok else 'FAIL'} {name:12s} {metric:6s} {c:10.3f}  (baseline {b:.3f})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
