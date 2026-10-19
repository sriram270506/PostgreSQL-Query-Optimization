"""Download the STATS dataset and the STATS-CEB workload (Han et al., VLDB 2022).

    python -m scripts.fetch_stats

Files land in data/stats/raw/ (git-ignored, ~40 MB). Source:
https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark
"""
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/Nathaniel-Han/End-to-End-CardEst-Benchmark/master"
FILES = [f"datasets/stats_simplified/{t}.csv" for t in
         ("badges", "comments", "postHistory", "postLinks", "posts", "tags", "users", "votes")]
FILES += ["datasets/stats_simplified/stats.sql", "workloads/stats_CEB/stats_CEB.sql",
          "workloads/stats_CEB/sub_plan_queries/stats_CEB_sub_queries.sql"]
OUT = "data/stats/raw"


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in FILES:
        dst = os.path.join(OUT, os.path.basename(f))
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        print("fetching", f)
        urllib.request.urlretrieve(f"{BASE}/{f}", dst)
    print("done:", OUT)


if __name__ == "__main__":
    main()
