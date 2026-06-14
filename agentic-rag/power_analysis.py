"""
power_analysis.py — Revision Stage R0b: paired effect sizes + post-hoc power.

For each of the 9 comparisons (3 metrics x 3 condition pairs) on the 300-row
results/results_main.csv:
  - Cohen's d_z on the paired differences (pairwise-complete rows only):
        d_z = mean(diff) / sd(diff, ddof=1)
  - Achieved (post-hoc) power at alpha = .05, two-sided, for the WILCOXON
    signed-rank test, approximated as the power of a paired t-test with the
    observed d_z and an effective sample size n_eff = ARE * n, where
    ARE = 0.955 (asymptotic relative efficiency of Wilcoxon vs t under
    normal shift alternatives).
    Primary method: statsmodels TTestPower (noncentral-t exact power).
    Fallback if statsmodels is unavailable: normal approximation
        power = Phi(|d_z|*sqrt(n_eff) - z_{0.975}) + Phi(-|d_z|*sqrt(n_eff) - z_{0.975})
    The method actually used is stated in the output.

No Ollama usage. Output: results/power_analysis.csv + printed table.
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MAIN = "results/results_main.csv"
ALPHA = 0.05
ARE_WILCOXON = 0.955
METRICS = ["faithfulness", "answer_relevancy", "context_precision"]
PAIRS = [
    ("baseline", "reranker"),
    ("reranker", "agentic"),
    ("baseline", "agentic"),
]

try:
    from statsmodels.stats.power import TTestPower
    _POWER_METHOD = ("statsmodels TTestPower (noncentral t), "
                     f"n_eff = {ARE_WILCOXON} * n (Wilcoxon ARE)")

    def achieved_power(d_z: float, n: int) -> float:
        return float(TTestPower().power(effect_size=abs(d_z),
                                        nobs=ARE_WILCOXON * n,
                                        alpha=ALPHA,
                                        alternative="two-sided"))
except ImportError:
    _POWER_METHOD = ("normal approximation "
                     "Phi(|d_z|*sqrt(n_eff) - z_.975) + Phi(-|d_z|*sqrt(n_eff) - z_.975), "
                     f"n_eff = {ARE_WILCOXON} * n (Wilcoxon ARE)")

    def achieved_power(d_z: float, n: int) -> float:
        n_eff = ARE_WILCOXON * n
        z_crit = stats.norm.ppf(1 - ALPHA / 2)
        nc = abs(d_z) * np.sqrt(n_eff)
        return float(stats.norm.cdf(nc - z_crit) + stats.norm.cdf(-nc - z_crit))


def main():
    df = pd.read_csv(MAIN)
    df = df[df["question"] != "** AGGREGATE MEAN **"].reset_index(drop=True)
    assert len(df) == 300, f"expected 300 per-query rows, got {len(df)}"
    print(f"Loaded {len(df)} per-query rows from {MAIN}")
    print(f"Power method: {_POWER_METHOD}\n")

    rows = []
    for metric in METRICS:
        for cond_a, cond_b in PAIRS:
            col_a, col_b = f"{cond_a}_{metric}", f"{cond_b}_{metric}"
            paired = df[[col_a, col_b]].dropna()
            n = len(paired)
            diffs = (paired[col_b] - paired[col_a]).to_numpy()
            sd = float(diffs.std(ddof=1))
            d_z = float(diffs.mean() / sd) if sd > 0 else float("nan")
            rows.append({
                "metric":        metric,
                "comparison":    f"{cond_a} vs {cond_b}",
                "n_pairs":       n,
                "mean_diff":     round(float(diffs.mean()), 4),
                "sd_diff":       round(sd, 4),
                "cohens_dz":     round(d_z, 4),
                "achieved_power": round(achieved_power(d_z, n), 4),
            })

    out = pd.DataFrame(rows)
    out["power_method"] = _POWER_METHOD
    out.to_csv("results/power_analysis.csv", index=False)
    pd.set_option("display.width", 200)
    print("PAIRED EFFECT SIZES + POST-HOC POWER  (alpha=.05 two-sided; "
          "d_z = mean(diff)/sd(diff))")
    print("=" * 100)
    print(out.drop(columns=["power_method"]).to_string(index=False))
    print(f"\nMethod: {_POWER_METHOD}")
    print("Saved -> results/power_analysis.csv")


if __name__ == "__main__":
    main()
