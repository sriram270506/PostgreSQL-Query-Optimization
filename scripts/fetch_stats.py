"""Download the STATS dataset and the STATS-CEB workload (Han et al., VLDB 2022).

    python -m scripts.fetch_stats

Files land in data/stats/raw/ (git-ignored, ~40 MB). Source, pinned to one commit so every
run evaluates on byte-identical data; each file is checked against its SHA-256:
https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark
"""
import hashlib
import os
import urllib.request

COMMIT = "670cb8d4bf4cbfa32f94fdf17f33973d3fd67d1b"  # master as of 2023-03-14
BASE = f"https://raw.githubusercontent.com/Nathaniel-Han/End-to-End-CardEst-Benchmark/{COMMIT}"
SHA256 = {
    "datasets/stats_simplified/badges.csv": "ad99b96804a61dc2d608504518d47bf584e174bcce74fd229e0487db0391902d",
    "datasets/stats_simplified/comments.csv": "691c11551ecec80a125f9671d6f370e1d0935f44d249a02aae933ce0b92f81c7",
    "datasets/stats_simplified/postHistory.csv": "b155a8564aac8fb8a278ebbb2648af3cbdb7017165ebfb6866c8fb4fc7e9c757",
    "datasets/stats_simplified/postLinks.csv": "25161d611c2b557f087dc7d29832ee3d232c8fdb08955847ae6d842a4f66377e",
    "datasets/stats_simplified/posts.csv": "91a170bbadb95b8f350e56b4ddea3eaefcdb19d0641484cc14c8d611a5bd1e9f",
    "datasets/stats_simplified/tags.csv": "92491cf4891a56bc5b8a83c2370ac6ec5e9dfd9619616638f0693974afcad108",
    "datasets/stats_simplified/users.csv": "b661c8a090af80e644db63daa93f0944ee5c018a7c26f32d519123da0704cd6d",
    "datasets/stats_simplified/votes.csv": "fadd897b84f5ed715ad4b8d66b6d2d1da8463ff1fafcc86dce9a7210786c9531",
    "datasets/stats_simplified/stats.sql": "adb88e0ab3640fa75bfb4eb67cf128742c0559fb5d0c26aa6e41243ee765b291",
    "workloads/stats_CEB/stats_CEB.sql": "8c4f9dca3e4aef8f7d86a3662739e3c2bfa3caec01b23f30b143325a306f86a3",
    "workloads/stats_CEB/sub_plan_queries/stats_CEB_sub_queries.sql":
        "e5cd89985493cff0cb03561e2bf6c1bcaec5c2a95aa4c7e0e7daffc9e58c7897",
}
OUT = "data/stats/raw"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    for f, want in SHA256.items():
        dst = os.path.join(OUT, os.path.basename(f))
        if not (os.path.exists(dst) and sha256(dst) == want):
            print("fetching", f)
            urllib.request.urlretrieve(f"{BASE}/{f}", dst + ".part")
            got = sha256(dst + ".part")
            if got != want:
                os.remove(dst + ".part")
                raise SystemExit(f"checksum mismatch for {f}: got {got}, expected {want}")
            os.replace(dst + ".part", dst)
    print(f"done: {OUT} (commit {COMMIT[:12]}, all checksums verified)")


if __name__ == "__main__":
    main()
