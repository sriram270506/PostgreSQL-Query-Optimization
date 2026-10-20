"""Render the measured results (benchmark + end-to-end JSON) into README.md.

    python -m scripts.report

Replaces everything between <!-- RESULTS:BEGIN --> and <!-- RESULTS:END --> so the README
can never disagree with models/*/benchmark.json and models/stats/e2e.json.
"""
import json
import os

ORDER = ["postgres_live", "postgres_xstats", "factorjoin", "mscn", "mlp", "treelstm"]
LABEL = {"postgres_live": "PostgreSQL 16", "postgres_xstats": "PostgreSQL 16 + CREATE STATISTICS",
         "factorjoin": "FactorJoin (Chow-Liu BN)", "mscn": "MSCN (baseline)", "mlp": "Fusion MLP", "treelstm": "TreeLSTM + MLP"}


def q(v):
    return f"{v:,.0f}×" if v >= 100 else f"{v:.2f}×"


def acc_table(acc, ci=True):
    rows = ["| Estimator | Median | p95 | p99 (95% CI) | Max | ≤2× |", "|---|---|---|---|---|---|"]
    for n in ORDER:
        if n not in acc:
            continue
        o = acc[n]["overall"]
        c = acc[n].get("ci95", {}).get("p99")
        p99 = f"**{q(o['p99'])}**" + (f" ({q(c[0])}–{q(c[1])})" if ci and c else "")
        rows.append(f"| {LABEL[n]} | {q(o['median'])} | {q(o['p95'])} | {p99} | {q(o['max'])} | {o['within_2x'] * 100:.1f}% |")
    return "\n".join(rows)


def verdict(c, a, b):
    return {"a better": f"{LABEL[a]} better", "b better": f"{LABEL[b]} better"}.get(c["verdict"], c["verdict"])


def cmp_line(pairs):
    return "Matched: " + "; ".join(f"{LABEL[p.split(' vs ')[0]]} vs {LABEL[p.split(' vs ')[1]]}: "
                                   f"{c['mean_log_q_diff']:+.3f} [{c['ci95_mean'][0]:+.3f}, {c['ci95_mean'][1]:+.3f}] "
                                   f"({verdict(c, *p.split(' vs '))})" for p, c in pairs.items() if c.get("ci95_mean"))


def section(path, title):
    if not os.path.exists(path):
        return f"### {title}\n\n_not run yet_\n"
    b = json.load(open(path))
    pv = b.get("provenance", {})
    prov = (f" Produced by commit `{(pv.get('git_commit') or '?')[:10]}`{' (dirty tree)' if pv.get('git_dirty') else ''}, "
            f"data `{pv.get('data_hash')}`, workload `{pv.get('workload_sha256')}`." if pv else "")
    out = [f"### {title}", "", f"{b['n_test']:,} held-out test queries ({b['n_train']:,} train / {b['n_val']:,} val).{prov}", "",
           acc_table(b["accuracy"]), ""]
    if "ceb" in b:
        out += ["**Official STATS-CEB workload (146 queries):**", "", acc_table(b["ceb"]["accuracy"]), ""]
    sig = [s for s in b.get("significance", []) if s["b"] in ("factorjoin", "mlp", "postgres_live", "postgres_xstats")]
    if sig:
        out += ["Paired Wilcoxon signed-rank tests on per-query log Q-error:", "",
                "| A vs B | mean log-Q A / B | A closer / B closer | p-value |", "|---|---|---|---|"]
        out += [f"| {LABEL[s['a']]} vs {LABEL[s['b']]} | {s['mean_log_q_a']:.3f} / {s['mean_log_q_b']:.3f} | "
                f"{s['a_better_frac'] * 100:.0f}% / {s['b_better_frac'] * 100:.0f}% | {s['p_value']:.2g} |" for s in sig]
        out.append("")
    if b.get("seeds"):
        out += ["Seed spread (each learned model trained with independent seeds):", "",
                "| Model | seeds | median | p95 | p99 |", "|---|---|---|---|---|"]
        for n in ("mscn", "mlp", "treelstm"):
            if n in b["seeds"]:
                s = b["seeds"][n]
                out.append(f"| {LABEL[n]} | {s['n_seeds']} | {s['median']['mean']:.3f} ± {s['median']['std']:.3f} | "
                           f"{s['p95']['mean']:.2f} ± {s['p95']['std']:.2f} | {s['p99']['mean']:.1f} ± {s['p99']['std']:.1f} |")
        out.append("")
    for k, label in (("split_tmpl", "Unseen query templates"), ("split_shape", "Unseen join sizes (train 1–3 tables, test 4–5)"),
                     ("split_join", "Unseen join graphs"), ("split_sel", "Unseen cardinality range (both tails held out)")):
        if k in b.get("ood", {}):
            out += [f"**Out of distribution: {label}** ({b['ood'][k]['n']:,} queries, models retrained without them)", "",
                    acc_table(b["ood"][k]["accuracy"], ci=False), ""]
            if b["ood"][k].get("compare"):
                out += [cmp_line(b["ood"][k]["compare"]), ""]
    for k, v in b.get("special", {}).items():
        out += [f"**Never-trained set: {v['label']}** ({v['n']:,} queries)", "", acc_table(v["accuracy"], ci=False), ""]
    if b.get("compare"):
        out += ["**Matched comparisons** (paired bootstrap 95% CI of the difference in mean log Q-error; "
                "negative = first is better; CI containing 0 = inconclusive):", "",
                "| Queries | Pair | Δ mean log-q [95% CI] | verdict |", "|---|---|---|---|"]
        for sl, pairs in b["compare"].items():
            for pair, c in pairs.items():
                if c.get("ci95_mean"):
                    a_, b_ = pair.split(" vs ")
                    out.append(f"| {sl} | {LABEL[a_]} vs {LABEL[b_]} | {c['mean_log_q_diff']:+.3f} "
                               f"[{c['ci95_mean'][0]:+.3f}, {c['ci95_mean'][1]:+.3f}] | {verdict(c, a_, b_)} |")
        ag = b.get("seeds", {}).get("agreement")
        if ag:
            out += ["", "Per-seed agreement (share of seed pairs in which the first model wins): "
                    + ", ".join(f"{LABEL[p.split(' vs ')[0]]} vs {LABEL[p.split(' vs ')[1]]} {v * 100:.0f}%" for p, v in ag.items())]
        out.append("")
    if b.get("breakdown"):
        bd = b["breakdown"]
        out += ["**Where errors come from** (p95 Q-error per slice of the test set):", ""]
        for dim, rows in bd.items():
            names = [n for n in ORDER if any(n in r for r in rows.values())]
            out += [f"| {dim} | n | " + " | ".join(LABEL[n] for n in names) + " |", "|---|---|" + "---|" * len(names)]
            for k_, r in rows.items():
                out.append(f"| {k_} | {r['n']:,} | " + " | ".join(q(r[n]["p95"]) for n in names) + " |")
            out.append("")
    if b.get("calibration"):
        out += ["**Uncertainty calibration on test** (Spearman of seed spread vs error; p95 Q-error on the most-confident 80% / all):", ""]
        for n, c in b["calibration"].items():
            cc = {x["coverage"]: x["p95_q"] for x in c["coverage_curve"]}
            out.append(f"- {LABEL[n]}: Spearman {c['spearman_std_vs_error']:.2f}, p95 {q(cc[0.8])} / {q(cc[1.0])}, "
                       f"fallback fired on {c['fallback_rate'] * 100:.1f}%")
        out.append("")
    if b.get("drift"):
        out += ["**Data drift** (models built on the 2013-06 snapshot, about half of today's rows):", "",
                "| Variant | Estimator | median | p95 | p99 |", "|---|---|---|---|---|"]
        for v, r in b["drift"].items():
            for n in ORDER:
                if n in r:
                    out.append(f"| {v} | {LABEL[n]} | {q(r[n]['median'])} | {q(r[n]['p95'])} | {q(r[n]['p99'])} |")
        out.append("")
    if b.get("plans"):
        out += ["**Join-order quality** (DP optimizer on each estimator's sub-plan estimates, scored with true C_out):", "",
                "| Estimator | optimal plan | mean cost / optimal | worst |", "|---|---|---|---|"]
        for n in ORDER:
            if n in b["plans"]:
                p = b["plans"][n]
                out.append(f"| {LABEL[n]} | {p['optimal_rate'] * 100:.1f}% | {p['mean_ratio']:.3f}× | {q(p['max_ratio'])} |")
        out.append("")
    if b.get("properties"):
        out += ["**Robustness** (guards and property checks):", "",
                "| Estimator | uncertainty fallback | clamped | order-invariant | monotone | latency p50 / p99 |",
                "|---|---|---|---|---|---|"]
        for n in ORDER:
            if n in b["properties"]:
                g = b["accuracy"].get(n, {}).get("guards", {})
                p, l = b["properties"][n], b["latency"].get(n, {})
                out.append(f"| {LABEL[n]} | {g.get('fallback_rate', 0) * 100:.1f}% | {g.get('clamp_rate', 0) * 100:.1f}% | "
                           f"{p['order_invariant_rate'] * 100:.0f}% | {(1 - p['monotone_violation_rate']) * 100:.0f}% | "
                           f"{l.get('p50_ms', 0):.1f} / {l.get('p99_ms', 0):.1f} ms |")
        l = b["latency"]["exact_execution"]
        out += [f"| exact execution | | | | | {l['p50_ms']:.1f} / {l['p99_ms']:.1f} ms |", ""]
    return "\n".join(out)


def e2e_section(path):
    if not os.path.exists(path):
        return "### End-to-end runtime on PostgreSQL 16\n\n_not run yet_\n"
    e = json.load(open(path))
    out = ["### End-to-end runtime on PostgreSQL 16 (STATS-CEB)", "",
           f"Each estimator's sub-plan cardinalities injected with pg_hint_plan, all {e['n_queries']} STATS-CEB queries "
           f"executed, timeout {e['timeout_s']:.0f} s (timeouts charged in full)."
           + (f" Note: {e['baseline_note']}." if e.get("baseline_note") else ""), "",
           "| Configuration | execution | + estimation | net vs PostgreSQL | timeouts | faster / slower queries (>10%) |",
           "|---|---|---|---|---|---|"]
    lab = {**LABEL, "postgres": "PostgreSQL 16 (own estimates)", "oracle": "True cardinalities (oracle)"}  # e2e config names
    for n in ["postgres", "postgres_xstats", "oracle", "factorjoin", "mscn", "mlp", "treelstm"]:
        if n in e["summary"]:
            s = e["summary"][n]
            base = e["summary"]["postgres"]["total_s"]
            net = s.get("net_total_s", s["total_s"])
            est = f"{s['estimation_s']:.1f} s" if "estimation_s" in s else ("–" if n.startswith("postgres") else "n/a")
            out.append(f"| {lab[n]} | {s['total_s']:,.0f} s | {est} | {base / net:.3f}× | {s['timeouts']} | "
                       f"{s.get('faster_queries', 0)} / {s.get('slower_queries', 0)} |")
    out.append("")
    out.append(f"Estimation = time to estimate all {e.get('n_subplans', '')} sub-plans (one batched call per query, "
               "cold caches) after the latency optimisations; PostgreSQL's own planning time is excluded from every "
               "row alike. The oracle's true counts cannot be produced at planning time, so it is an upper bound.")
    return "\n".join(out) + "\n"


def main():
    body = "\n\n".join([section("models/stats/benchmark.json", "STATS (real Stack Exchange data)"),
                        e2e_section("models/stats/e2e.json"),
                        section("models/benchmark.json", "HR (synthetic, planted correlations)")])
    readme = open("README.md", encoding="utf8").read()
    a, b = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
    readme = readme[: readme.index(a) + len(a)] + "\n\n" + body + "\n" + readme[readme.index(b):]
    open("README.md", "w", encoding="utf8").write(readme)
    print("README results updated")


if __name__ == "__main__":
    main()
