"""
Paired significance tests on per-query RAGAS scores.

Loads results/results_main.csv (which, after the Phase-2 evaluate.py change,
holds one RAGAS score per query per condition) and, for each metric and each
condition pair, reports:

  • mean difference (second condition minus first; positive = second is better)
  • paired Wilcoxon signed-rank p-value (scipy.stats.wilcoxon)
  • 95% bootstrap confidence interval on the mean difference (10,000 resamples)

Pairs are formed only from queries with a non-NaN score in BOTH conditions.

Usage:
  python significance_test.py
Output:
  printed table + results/significance.csv
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

import config

METRICS = ["faithfulness", "answer_relevancy", "context_precision"]
PAIRS = [
    ("baseline", "reranker"),
    ("reranker", "agentic"),
    ("baseline", "agentic"),
]
N_BOOTSTRAP = 10_000


def bootstrap_ci(diffs: np.ndarray, n_boot: int = N_BOOTSTRAP,
                 seed: int = config.RANDOM_SEED) -> tuple:
    """95% percentile bootstrap CI on the mean of paired differences."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    results_path = config.RESULTS_DIR / "results_main.csv"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run evaluate.py first.")
        sys.exit(1)

    df = pd.read_csv(results_path)
    df = df[df["question"] != "** AGGREGATE MEAN **"].reset_index(drop=True)
    print(f"Loaded {len(df)} per-query rows from {results_path}")

    # Per-query scores are required; broadcast aggregate means would make every
    # paired difference identical (or zero) and the test meaningless.
    for col in [f"{c}_{m}" for c, _ in PAIRS for m in METRICS]:
        if col in df.columns and df[col].notna().sum() > 1 and df[col].nunique() == 1:
            print(f"ERROR: column '{col}' has a single repeated value — "
                  f"results_main.csv contains broadcast means, not per-query scores.")
            print("Re-run evaluate.py (Phase-2 version) before testing significance.")
            sys.exit(1)

    rows = []
    for metric in METRICS:
        for cond_a, cond_b in PAIRS:
            col_a, col_b = f"{cond_a}_{metric}", f"{cond_b}_{metric}"
            if col_a not in df.columns or col_b not in df.columns:
                print(f"  [skip] {metric}: missing column {col_a} or {col_b}")
                continue

            paired = df[[col_a, col_b]].dropna()
            n = len(paired)
            if n < 5:
                print(f"  [skip] {metric} {cond_a} vs {cond_b}: only {n} paired rows")
                continue

            diffs = (paired[col_b] - paired[col_a]).to_numpy()
            mean_diff = float(diffs.mean())

            # Wilcoxon drops zero differences; if every pair is tied it raises.
            try:
                _, p_value = stats.wilcoxon(paired[col_a], paired[col_b])
                p_value = float(p_value)
            except ValueError:
                p_value = float("nan")  # all differences zero — no evidence either way

            ci_lo, ci_hi = bootstrap_ci(diffs)

            rows.append({
                "metric":      metric,
                "comparison":  f"{cond_a} vs {cond_b}",
                "n_pairs":     n,
                "mean_diff":   mean_diff,   # cond_b minus cond_a
                "wilcoxon_p":  p_value,
                "ci95_low":    ci_lo,
                "ci95_high":   ci_hi,
                "significant": bool(p_value == p_value and p_value < 0.05),
            })

    if not rows:
        print("No testable metric/condition pairs found.")
        sys.exit(1)

    out = pd.DataFrame(rows)
    out_path = config.RESULTS_DIR / "significance.csv"
    out.to_csv(out_path, index=False)

    hdr = (f"{'metric':<20} {'comparison':<24} {'n':>4} {'mean_diff':>10} "
           f"{'p (Wilcoxon)':>13} {'95% CI':>22} {'sig?':>5}")
    print("\n" + "=" * len(hdr))
    print("PAIRED SIGNIFICANCE TESTS  (mean_diff = second condition − first)")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        p_str = f"{r['wilcoxon_p']:.4g}" if r["wilcoxon_p"] == r["wilcoxon_p"] else "N/A"
        ci_str = f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]"
        print(f"{r['metric']:<20} {r['comparison']:<24} {r['n_pairs']:>4} "
              f"{r['mean_diff']:>+10.4f} {p_str:>13} {ci_str:>22} "
              f"{'*' if r['significant'] else '':>5}")
    print("-" * len(hdr))
    print("* = p < 0.05 (paired Wilcoxon signed-rank)")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
