"""
hetero_compare_150.py — Revision Stage R2 final step: llama-vs-qwen comparison
on the 150-query hetero-judge sample.

- Per-condition mean F/AR/CP under llama3.1:8b (from results/results_main.csv)
  vs qwen2.5:7b (from results/hetero_judge_150.csv) on the SAME 150 rows.
- Paired Wilcoxon under the qwen judge for faithfulness:
  baseline-vs-agentic, baseline-vs-reranker, reranker-vs-agentic
  (pairwise-complete rows only), with mean diff and n_pairs.
- Verdict lines: did the faithfulness ordering baseline<reranker<agentic HOLD
  at n=150, and does reranker-vs-agentic separate significantly under qwen?

No Ollama usage. Output: results/hetero_judge_comparison_150.csv + print.
"""

import sys

import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from hetero_judge_eval import sample_indices

CONDITIONS = ["baseline", "reranker", "agentic"]
METRICS = ["faithfulness", "answer_relevancy", "context_precision"]
PAIRS = [
    ("baseline", "reranker"),
    ("baseline", "agentic"),
    ("reranker", "agentic"),
]


def main():
    cache = pd.read_csv("results/generation_cache.csv")
    _, idxs150 = sample_indices()
    sample_qs = set(cache.iloc[idxs150]["question"].tolist())

    qwen = pd.read_csv("results/hetero_judge_150.csv")
    qwen = qwen.drop_duplicates(["condition", "question"], keep="last")
    main_df = pd.read_csv("results/results_main.csv")
    main_df = main_df[main_df["question"] != "** AGGREGATE MEAN **"]
    llama = main_df[main_df["question"].isin(sample_qs)]
    print(f"sample questions: {len(sample_qs)}; qwen rows: {len(qwen)}; "
          f"llama rows matched: {len(llama)}\n")

    # ── Per-condition means, llama vs qwen ────────────────────────────────────
    rows = []
    for cond in CONDITIONS:
        qc = qwen[qwen["condition"] == cond]
        qc = qc[qc["question"].isin(sample_qs)]
        for m in METRICS:
            rows.append({
                "section":     "means",
                "condition":   cond,
                "metric":      m,
                "llama_mean":  round(float(llama[f"{cond}_{m}"].mean()), 4),
                "qwen_mean":   round(float(qc[m].mean()), 4),
                "qwen_n":      int(qc[m].notna().sum()),
                "comparison":  "", "n_pairs": "", "mean_diff": "",
                "wilcoxon_p":  "", "significant": "",
            })

    print(f"{'condition':<10} {'metric':<20} {'llama3.1:8b':>12} "
          f"{'qwen2.5:7b':>12} {'qwen_n':>8}")
    for r in rows:
        print(f"{r['condition']:<10} {r['metric']:<20} {r['llama_mean']:>12.4f} "
              f"{r['qwen_mean']:>12.4f} {r['qwen_n']:>5}/150")

    # ── Paired Wilcoxon on FAITHFULNESS under the qwen judge ──────────────────
    wide = qwen[qwen["question"].isin(sample_qs)].pivot_table(
        index="question", columns="condition", values="faithfulness",
        aggfunc="last")
    print("\nPaired Wilcoxon on faithfulness, judge = qwen2.5:7b")
    print(f"{'comparison':<26} {'n_pairs':>8} {'mean_diff':>10} "
          f"{'p':>10} {'sig?':>5}")
    for cond_a, cond_b in PAIRS:
        paired = wide[[cond_a, cond_b]].dropna()
        diffs = paired[cond_b] - paired[cond_a]
        try:
            _, p = stats.wilcoxon(paired[cond_a], paired[cond_b])
            p = float(p)
        except ValueError:
            p = float("nan")
        sig = bool(p == p and p < 0.05)
        rows.append({
            "section":     "wilcoxon_qwen_faithfulness",
            "condition":   "", "metric": "faithfulness",
            "llama_mean":  "", "qwen_mean": "", "qwen_n": "",
            "comparison":  f"{cond_a} vs {cond_b}",
            "n_pairs":     len(paired),
            "mean_diff":   round(float(diffs.mean()), 4),
            "wilcoxon_p":  round(p, 6) if p == p else "nan",
            "significant": sig,
        })
        p_str = f"{p:.4g}" if p == p else "N/A"
        print(f"{cond_a + ' vs ' + cond_b:<26} {len(paired):>8} "
              f"{diffs.mean():>+10.4f} {p_str:>10} {'*' if sig else '':>5}")

    out = pd.DataFrame(rows)
    out.to_csv("results/hetero_judge_comparison_150.csv", index=False)

    # ── Verdicts ──────────────────────────────────────────────────────────────
    qmeans = {c: float(qwen[(qwen["condition"] == c)
                            & (qwen["question"].isin(sample_qs))]
                       ["faithfulness"].mean()) for c in CONDITIONS}
    ordering_held = qmeans["baseline"] < qmeans["reranker"] < qmeans["agentic"]
    ra = [r for r in rows if r.get("comparison") == "reranker vs agentic"][0]
    print(f"\nVERDICT 1 — faithfulness ordering baseline<reranker<agentic under "
          f"qwen at n=150: {'HELD' if ordering_held else 'BROKE'} "
          f"({qmeans['baseline']:.4f} / {qmeans['reranker']:.4f} / "
          f"{qmeans['agentic']:.4f})")
    print(f"VERDICT 2 — reranker-vs-agentic separates significantly under qwen: "
          f"{'YES' if ra['significant'] else 'NO'} "
          f"(p={ra['wilcoxon_p']}, mean_diff={ra['mean_diff']:+.4f}, "
          f"n={ra['n_pairs']})")
    print("\nSaved -> results/hetero_judge_comparison_150.csv")


if __name__ == "__main__":
    main()
