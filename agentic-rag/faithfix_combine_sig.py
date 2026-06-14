"""
faithfix_combine_sig.py — Revision R3 faithfulness coverage-bias fix: COMBINE.

Combines Method A (refusal imputation, faithfix_methodA.csv) and Method B
(tolerant re-parse of the non-refusal NaN rows, faithfix_methodB.csv) into one
bias-free faithfulness value per (question, condition) at full n=100:
    status 'scored'  -> original ragas faithfulness
    status 'imputed' -> 1.0 (refusal asserts no unsupported claims)  [Method A]
    status 'pending' -> Method B re-parsed faith_B
Then re-runs paired Wilcoxon + 10k-bootstrap CI + Holm (within faithfulness,
3 comparisons), pairwise-complete, on the corrected faithfulness.

Also reports the Method-A-only result (refusals imputed, the 6 non-refusal NaN
dropped as pairwise-incomplete) so we can check Methods A and B agree.

Writes results/significance_holm_2wiki_faithfix.csv. Prints corrected
per-condition means, the per-condition coverage breakdown, the significance
table, and an honest one-line verdict.
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from significance_test import PAIRS, bootstrap_ci

CONDITIONS = ["baseline", "reranker", "agentic"]


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    adj = np.full(m, np.nan)
    order = [i for i in np.argsort(p, kind="stable") if not np.isnan(p[i])]
    running = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * p[idx], 1.0)
        running = max(running, val)
        adj[idx] = running
    return adj


def pairwise_sig(wide, label):
    """wide: DataFrame indexed by question with one column per condition."""
    fam = []
    for a, b in PAIRS:
        both = wide[[a, b]].dropna()
        n = len(both)
        diffs = (both[b] - both[a]).to_numpy()
        try:
            _, p = stats.wilcoxon(both[a], both[b])
            p = float(p)
        except ValueError:
            p = float("nan")
        lo, hi = bootstrap_ci(diffs)
        fam.append({"set": label, "comparison": f"{a} vs {b}", "n_pairs": n,
                    "mean_diff": float(diffs.mean()), "wilcoxon_p": p,
                    "ci95_low": lo, "ci95_high": hi,
                    "ci_excludes_zero": bool(not (lo <= 0 <= hi))})
    for r, hp in zip(fam, holm([r["wilcoxon_p"] for r in fam])):
        r["holm_p"] = float(hp)
        sig = bool(hp == hp and hp < 0.05)
        crit = int(sig) + int(r["ci_excludes_zero"])
        r["status"] = ("robust" if crit == 2
                       else "suggestive" if crit == 1 else "n.s.")
    return fam


def main():
    a = pd.read_csv("results/faithfix_methodA.csv")
    b = pd.read_csv("results/faithfix_methodB.csv")
    bmap = {(r["question"], r["condition"]): r["faith_B"]
            for _, r in b.iterrows()}

    # combined faithfulness column
    def combined(row):
        if row["status"] == "pending":
            return bmap.get((row["question"], row["condition"]), float("nan"))
        return row["faith_A"]          # scored orig, or imputed 1.0
    a["faith_fixed"] = a.apply(combined, axis=1)

    # per-condition coverage breakdown + means
    print("CORRECTED FAITHFULNESS — per-condition coverage & means (n=100)")
    print(f"{'cond':9s} {'orig':>5s} {'imp':>4s} {'reparse':>8s} {'stillNaN':>9s}"
          f" {'mean_orig':>10s} {'mean_A':>8s} {'mean_FIX':>9s} {'n_fix':>6s}")
    orig = pd.read_csv("results/scores_2wiki.csv")
    for c in CONDITIONS:
        sub = a[a.condition == c]
        n_orig = (sub.status == "scored").sum()
        n_imp = (sub.status == "imputed").sum()
        pend = sub[sub.status == "pending"]
        n_rep = int(pend["faith_fixed"].notna().sum())
        n_still = int(pend["faith_fixed"].isna().sum())
        mean_orig = pd.to_numeric(
            orig[orig.condition == c]["faithfulness"], errors="coerce").mean()
        mean_a = sub["faith_A"].mean(skipna=True)
        mean_fix = sub["faith_fixed"].mean(skipna=True)
        n_fix = int(sub["faith_fixed"].notna().sum())
        print(f"{c:9s} {n_orig:5d} {n_imp:4d} {n_rep:8d} {n_still:9d}"
              f" {mean_orig:10.4f} {mean_a:8.4f} {mean_fix:9.4f} {n_fix:6d}")

    # wide tables
    wide_fix = a.pivot_table(index="question", columns="condition",
                             values="faith_fixed", aggfunc="first")
    wide_a = a.pivot_table(index="question", columns="condition",
                           values="faith_A", aggfunc="first")

    fam_fix = pairwise_sig(wide_fix, "combined_A+B")
    fam_a = pairwise_sig(wide_a, "methodA_only")

    cols = ["set", "comparison", "n_pairs", "mean_diff", "wilcoxon_p",
            "holm_p", "ci95_low", "ci95_high", "ci_excludes_zero", "status"]
    out = pd.DataFrame(fam_fix + fam_a)[cols]
    out.to_csv("results/significance_holm_2wiki_faithfix.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nSIGNIFICANCE on corrected faithfulness "
          "(mean_diff = second - first; positive favours later condition)")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # verdict
    def get(fam, comp):
        return next(r for r in fam if r["comparison"] == comp)
    ba_fix = get(fam_fix, "baseline vs agentic")
    ra_fix = get(fam_fix, "reranker vs agentic")
    ba_a = get(fam_a, "baseline vs agentic")
    print("\nVERDICT (faithfulness, coverage-bias removed):")
    print(f"  combined A+B (n=100): base-vs-agentic={ba_fix['status']} "
          f"(mean_diff={ba_fix['mean_diff']:+.4f}, Holm p={ba_fix['holm_p']:.4f}, "
          f"CI[{ba_fix['ci95_low']:+.4f},{ba_fix['ci95_high']:+.4f}]); "
          f"rerank-vs-agentic={ra_fix['status']} "
          f"(mean_diff={ra_fix['mean_diff']:+.4f})")
    agree = (ba_a["status"] == ba_fix["status"]
             and np.sign(ba_a["mean_diff"]) == np.sign(ba_fix["mean_diff"]))
    print(f"  Method A only (n={ba_a['n_pairs']}): base-vs-agentic="
          f"{ba_a['status']} (mean_diff={ba_a['mean_diff']:+.4f})")
    print(f"  Methods A and B AGREE on base-vs-agentic direction+verdict: "
          f"{'YES' if agree else 'NO'}")
    print("\nSaved -> results/significance_holm_2wiki_faithfix.csv")


if __name__ == "__main__":
    main()
