"""
significance_2wiki.py — Revision Stage R3 step 4: significance on 2wiki results.

Identical methodology to significance_test.py (paired Wilcoxon signed-rank +
10,000-resample percentile bootstrap CI on mean paired differences,
pairwise-complete rows only) but reading results/results_2wiki.csv.

Output: results/significance_2wiki.csv  (then run
        python multiple_comparisons.py --input results/significance_2wiki.csv
               --output results/significance_holm_2wiki.csv)
Also prints per-condition refusal counts and mean latency.
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from significance_test import METRICS, PAIRS, bootstrap_ci

REFUSAL = "does not contain enough information"
CONDITIONS = ["baseline", "reranker", "agentic"]


def main():
    df = pd.read_csv("results/results_2wiki.csv")
    df = df[df["question"] != "** AGGREGATE MEAN **"].reset_index(drop=True)
    print(f"Loaded {len(df)} per-query rows from results/results_2wiki.csv")

    rows = []
    for metric in METRICS:
        for cond_a, cond_b in PAIRS:
            col_a, col_b = f"{cond_a}_{metric}", f"{cond_b}_{metric}"
            paired = df[[col_a, col_b]].dropna()
            n = len(paired)
            if n < 5:
                print(f"  [skip] {metric} {cond_a} vs {cond_b}: only {n} pairs")
                continue
            diffs = (paired[col_b] - paired[col_a]).to_numpy()
            try:
                _, p = stats.wilcoxon(paired[col_a], paired[col_b])
                p = float(p)
            except ValueError:
                p = float("nan")
            ci_lo, ci_hi = bootstrap_ci(diffs)
            rows.append({
                "metric":      metric,
                "comparison":  f"{cond_a} vs {cond_b}",
                "n_pairs":     n,
                "mean_diff":   float(diffs.mean()),
                "wilcoxon_p":  p,
                "ci95_low":    ci_lo,
                "ci95_high":   ci_hi,
                "significant": bool(p == p and p < 0.05),
            })

    out = pd.DataFrame(rows)
    out.to_csv("results/significance_2wiki.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nPAIRED SIGNIFICANCE — 2WikiMultiHopQA (mean_diff = second − first)")
    print(out.to_string(index=False))

    print("\nPer-condition refusals / latency (n=100):")
    for cond in CONDITIONS:
        ref = int(df[f"{cond}_answer"].fillna("").str.lower()
                  .str.contains(REFUSAL).sum())
        lat = float(df[f"{cond}_latency"].mean(skipna=True))
        print(f"  {cond:<10} refusals={ref:>3}/100   mean latency={lat:.2f}s")
    print("\nSaved -> results/significance_2wiki.csv")


if __name__ == "__main__":
    main()
