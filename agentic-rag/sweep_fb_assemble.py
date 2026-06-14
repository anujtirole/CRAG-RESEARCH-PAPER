"""
sweep_fb_assemble.py — Revision Stage R1 final step: build the fallback-sweep table.

SUBSET: all three tau rows cover the SAME first 150 queries of
data/queries_hotpot_300.json.
  tau=0.40 : extracted from the main n=300 run (results/results_main.csv)
             filtered to those 150 questions — NO regeneration.
  tau=0.50 : results/sweep_fb_tau050_cache.csv (generated + scored by sweep_fb.py)
  tau=0.60 : results/sweep_fb_tau060_cache.csv

Columns: mean faithfulness / answer_relevancy / context_precision, mean latency,
refusal count, fallback-fired count, low-confidence count (each out of 150).
Refusal = answer contains "does not contain enough information" (case-insens.).

No Ollama usage. Output: results/threshold_sweep_fallback.csv + printed table.
"""

import json
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

QUERIES_300 = Path("data/queries_hotpot_300.json")
N_SUBSET = 150
REFUSAL = "does not contain enough information"
METRICS = ["faithfulness", "answer_relevancy", "context_precision"]


def _as_bool(series):
    return series.astype(str).str.strip().str.lower() == "true"


def summarize(df, tau, source):
    assert len(df) == N_SUBSET, f"tau={tau}: {len(df)} rows, expected {N_SUBSET}"
    row = {"tau": tau, "n_queries": len(df), "source": source}
    for m in METRICS:
        row[f"mean_{m}"] = round(float(df[m].mean(skipna=True)), 4)
        row[f"n_scored_{m}"] = int(df[m].notna().sum())
    row["mean_latency_s"] = round(float(df["latency"].mean(skipna=True)), 2)
    row["refusal_count"] = int(df["answer"].fillna("").str.lower()
                               .str.contains(REFUSAL).sum())
    row["fallback_fired"] = int(_as_bool(df["used_fallback"]).sum())
    row["low_conf_count"] = int(_as_bool(df["low_conf"]).sum())
    return row


def main():
    first150 = [q["question"] for q in
                json.loads(QUERIES_300.read_text(encoding="utf-8"))[:N_SUBSET]]
    first150_set = set(first150)
    assert len(first150_set) == N_SUBSET

    rows = []

    # tau=0.40 from the main n=300 run, filtered to the same 150 queries
    main_df = pd.read_csv("results/results_main.csv")
    main_df = main_df[main_df["question"].isin(first150_set)]
    sub40 = pd.DataFrame({
        "answer":        main_df["agentic_answer"],
        "latency":       main_df["agentic_latency"],
        "used_fallback": main_df["agentic_used_fallback"],
        "low_conf":      main_df["agentic_low_conf"],
        **{m: main_df[f"agentic_{m}"] for m in METRICS},
    })
    rows.append(summarize(sub40, 0.40, "results_main.csv (first-150 filter)"))

    # tau=0.50 / 0.60 from the dedicated sweep caches
    for tau, tag in [(0.50, "050"), (0.60, "060")]:
        cache = pd.read_csv(f"results/sweep_fb_tau{tag}_cache.csv")
        assert set(cache["question"]) == first150_set, \
            f"tau={tau}: cache questions do not match the first-150 subset"
        sub = pd.DataFrame({
            "answer":        cache["agentic_answer"],
            "latency":       cache["agentic_latency"],
            "used_fallback": cache["agentic_used_fallback"],
            "low_conf":      cache["agentic_low_conf"],
            **{m: cache[m] for m in METRICS},
        })
        rows.append(summarize(sub, tau, f"sweep_fb_tau{tag}_cache.csv"))

    out = pd.DataFrame(rows)
    out.to_csv("results/threshold_sweep_fallback.csv", index=False)
    pd.set_option("display.width", 250)
    print("THRESHOLD SWEEP WITH FALLBACK — agentic condition, "
          f"first {N_SUBSET} queries of queries_hotpot_300.json")
    print("=" * 110)
    print(out.to_string(index=False))
    print("\nSaved -> results/threshold_sweep_fallback.csv")


if __name__ == "__main__":
    main()
