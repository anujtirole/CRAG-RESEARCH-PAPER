"""
variance_stats.py — Revision Stage R0a: dispersion statistics from existing data.

From the 300-row results/results_main.csv, for each condition x metric reports:
mean, SD (ddof=1), median, IQR (q75 - q25), and n-scored (non-NaN rows).
Covers the three RAGAS metrics plus latency. No Ollama usage.

Output: results/variance_stats.csv + printed table.
"""

import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MAIN = "results/results_main.csv"
CONDITIONS = ["baseline", "reranker", "agentic"]
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "latency"]


def main():
    df = pd.read_csv(MAIN)
    df = df[df["question"] != "** AGGREGATE MEAN **"].reset_index(drop=True)
    assert len(df) == 300, f"expected 300 per-query rows, got {len(df)}"
    print(f"Loaded {len(df)} per-query rows from {MAIN}\n")

    rows = []
    for cond in CONDITIONS:
        for met in METRICS:
            col = f"{cond}_{met}"
            s = df[col].dropna()
            q25, q75 = s.quantile(0.25), s.quantile(0.75)
            rows.append({
                "condition": cond,
                "metric":    met,
                "n_scored":  int(s.count()),
                "mean":      round(float(s.mean()), 4),
                "sd":        round(float(s.std(ddof=1)), 4),
                "median":    round(float(s.median()), 4),
                "q25":       round(float(q25), 4),
                "q75":       round(float(q75), 4),
                "iqr":       round(float(q75 - q25), 4),
            })

    out = pd.DataFrame(rows)
    out.to_csv("results/variance_stats.csv", index=False)
    pd.set_option("display.width", 200)
    print("VARIANCE / DISPERSION STATISTICS (n=300 main run)")
    print("=" * 90)
    print(out.to_string(index=False))
    print("\nSaved -> results/variance_stats.csv")


if __name__ == "__main__":
    main()
