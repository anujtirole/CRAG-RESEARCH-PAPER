"""
multiple_comparisons.py
-----------------------
Holm-Bonferroni correction of the Wilcoxon p-values in results/significance.csv.

Each metric family (faithfulness, answer_relevancy, context_precision) contains
m=3 pairwise comparisons; Holm is applied WITHIN each family.

Status rules:
  robust     = Holm-adjusted p < 0.05 AND the 95% bootstrap CI excludes zero
  suggestive = raw p < 0.05 but not robust
  n.s.       = otherwise

Usage:
  python multiple_comparisons.py                          # n=150 default
  python multiple_comparisons.py --input results/significance_300.csv \
                                 --output results/significance_holm_300.csv
"""

import argparse
import pandas as pd


def holm_adjust(pvals):
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [None] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, min(adj, 1.0))
        adjusted[idx] = running_max
    return adjusted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/significance.csv")
    ap.add_argument("--output", default="results/significance_holm.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    out_rows = []
    for metric, fam in df.groupby("metric", sort=False):
        fam = fam.reset_index(drop=True)
        adj = holm_adjust(fam["wilcoxon_p"].tolist())
        for i, row in fam.iterrows():
            ci_excludes_zero = (row["ci95_low"] > 0) or (row["ci95_high"] < 0)
            if adj[i] < 0.05 and ci_excludes_zero:
                status = "robust"
            elif row["wilcoxon_p"] < 0.05:
                status = "suggestive"
            else:
                status = "n.s."
            out_rows.append({
                "metric":          metric,
                "comparison":      row["comparison"],
                "n_pairs":         row["n_pairs"],
                "mean_diff":       row["mean_diff"],
                "p_raw":           row["wilcoxon_p"],
                "p_holm":          adj[i],
                "ci95_low":        row["ci95_low"],
                "ci95_high":       row["ci95_high"],
                "ci_excludes_zero": ci_excludes_zero,
                "status":          status,
            })

    out = pd.DataFrame(out_rows)
    out.to_csv(args.output, index=False)
    pd.set_option("display.width", 200)
    print(f"Holm-Bonferroni correction (within each metric family, m=3)")
    print(f"input:  {args.input}\noutput: {args.output}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
