"""
sig_holm_2wiki.py — Revision Stage R3 step 4: significance + Holm on 2wiki.

Pure computation on results/scores_2wiki.csv (300 rows = 100 per condition,
long format: question, condition, faithfulness, answer_relevancy,
context_precision). No Ollama, no generation.

For each metric and each pair (base-vs-rerank, rerank-vs-agentic,
base-vs-agentic):
  * pivot to per-question pairs, drop pairs where either condition is NaN
    (pairwise-complete),
  * paired Wilcoxon signed-rank p,
  * 10,000-resample percentile bootstrap 95% CI on the MEAN difference
    (second - first), seed = config.RANDOM_SEED (identical to
    significance_test.bootstrap_ci),
  * Holm correction WITHIN each metric family (3 comparisons),
  * status: robust  = Holm p<0.05 AND CI excludes 0
            suggestive = exactly one of those two
            n.s.    = neither.
For faithfulness, also report how many pairs were dropped to NaN per
comparison (the coverage-bias issue).

Output: results/significance_holm_2wiki.csv  +  printed table + verdicts.
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
from significance_test import METRICS, PAIRS, bootstrap_ci

SCORES = "results/scores_2wiki.csv"
OUT = "results/significance_holm_2wiki.csv"


def holm(pvals):
    """Holm step-down adjusted p-values; preserves input order. NaN -> NaN."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.full(m, np.nan)
    order = [i for i in np.argsort(p, kind="stable") if not np.isnan(p[i])]
    running = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * p[idx], 1.0)
        running = max(running, val)   # enforce monotone non-decreasing
        adj[idx] = running
    return adj


def main():
    df = pd.read_csv(SCORES)
    n_total = len(df)
    # wide per-question table: index=question, columns=(condition, metric)
    wide = df.pivot_table(index="question", columns="condition",
                          values=METRICS, aggfunc="first")
    print(f"Loaded {n_total} score rows; "
          f"{wide.shape[0]} unique questions x 3 conditions.\n")

    rows = []
    for metric in METRICS:
        family = []
        for cond_a, cond_b in PAIRS:
            a = wide[(metric, cond_a)]
            b = wide[(metric, cond_b)]
            both = pd.concat([a, b], axis=1).dropna()
            n = len(both)
            n_dropped = len(wide) - n
            va = both.iloc[:, 0].to_numpy()
            vb = both.iloc[:, 1].to_numpy()
            diffs = vb - va
            try:
                _, p = stats.wilcoxon(va, vb)
                p = float(p)
            except ValueError:
                p = float("nan")
            ci_lo, ci_hi = bootstrap_ci(diffs)
            ci_excl = bool(not (ci_lo <= 0.0 <= ci_hi))
            family.append({
                "metric": metric,
                "comparison": f"{cond_a} vs {cond_b}",
                "n_pairs": int(n),
                "n_dropped_nan": int(n_dropped),
                "mean_diff": float(diffs.mean()),
                "wilcoxon_p": p,
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "ci_excludes_zero": ci_excl,
            })
        holm_p = holm([r["wilcoxon_p"] for r in family])
        for r, hp in zip(family, holm_p):
            r["holm_p"] = float(hp)
            holm_sig = bool(hp == hp and hp < 0.05)
            crit = int(holm_sig) + int(r["ci_excludes_zero"])
            r["status"] = ("robust" if crit == 2
                           else "suggestive" if crit == 1 else "n.s.")
            rows.append(r)

    cols = ["metric", "comparison", "n_pairs", "n_dropped_nan", "mean_diff",
            "wilcoxon_p", "holm_p", "ci95_low", "ci95_high",
            "ci_excludes_zero", "status"]
    out = pd.DataFrame(rows)[cols]
    out.to_csv(OUT, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("PAIRED SIGNIFICANCE + HOLM — 2WikiMultiHopQA "
          "(mean_diff = second - first; positive favours the later condition)")
    print(out.to_string(index=False,
                        float_format=lambda x: f"{x:.4f}"))

    # faithfulness coverage-bias call-out
    print("\nFAITHFULNESS coverage (pairs dropped to NaN per comparison):")
    for r in rows:
        if r["metric"] == "faithfulness":
            print(f"  {r['comparison']:<22} tested on n={r['n_pairs']:>3} "
                  f"pairs  (dropped {r['n_dropped_nan']} to NaN)")

    # one-line verdict per metric, focused on the agentic advantage
    print("\nVERDICT (agentic advantage on 2wiki):")
    for metric in METRICS:
        ba = next(r for r in rows if r["metric"] == metric
                  and r["comparison"] == "baseline vs agentic")
        ra = next(r for r in rows if r["metric"] == metric
                  and r["comparison"] == "reranker vs agentic")
        note = (f"  base-vs-agentic={ba['status']} "
                f"(Holm p={ba['holm_p']:.4f}, "
                f"CI[{ba['ci95_low']:+.4f},{ba['ci95_high']:+.4f}]); "
                f"rerank-vs-agentic={ra['status']}")
        if metric == "faithfulness":
            note += (f"  [tested on n={ba['n_pairs']} (base-v-ag) / "
                     f"n={ra['n_pairs']} (rerank-v-ag)]")
        print(f"{metric:>18}:{note}")

    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
