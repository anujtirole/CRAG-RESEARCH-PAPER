"""Stage 5 helper: n=300 main table, refusal counts, n=150 vs n=300 comparison."""
import pandas as pd

df = pd.read_csv("results/results_main.csv")
df = df[df["question"] != "** AGGREGATE MEAN **"]
print(f"n = {len(df)}")
for c in ["baseline", "reranker", "agentic"]:
    f = df[f"{c}_faithfulness"].mean()
    ar = df[f"{c}_answer_relevancy"].mean()
    cp = df[f"{c}_context_precision"].mean()
    lat = df[f"{c}_latency"].mean()
    ref = df[f"{c}_answer"].astype(str).str.contains(
        "does not contain enough information", case=False).sum()
    print(f"{c:<10} F={f:.4f}  AR={ar:.4f}  CP={cp:.4f}  "
          f"latency={lat:.2f}s  refusals={ref}/{len(df)}")

print("\nagentic extras: n_attempts mean=%.3f  used_fallback=%d  low_conf=%d" % (
    df["agentic_n_attempts"].astype(float).mean(),
    df["agentic_used_fallback"].astype(str).str.strip().str.lower().eq("true").sum(),
    df["agentic_low_conf"].astype(str).str.strip().str.lower().eq("true").sum()))

old = pd.read_csv("results/results_main_150.csv")
old = old[old["question"] != "** AGGREGATE MEAN **"]
print(f"\nold n=150 refusal counts:")
for c in ["baseline", "reranker", "agentic"]:
    ref = old[f"{c}_answer"].astype(str).str.contains(
        "does not contain enough information", case=False).sum()
    print(f"{c:<10} refusals={ref}/{len(old)}  lat={old[f'{c}_latency'].mean():.2f}s")

new = df[~df["question"].isin(set(old["question"]))]
print(f"\nnew-150-only means (rows={len(new)}):")
for c in ["baseline", "reranker", "agentic"]:
    print(f"{c:<10} F={new[f'{c}_faithfulness'].mean():.4f}  "
          f"AR={new[f'{c}_answer_relevancy'].mean():.4f}  "
          f"CP={new[f'{c}_context_precision'].mean():.4f}  "
          f"lat={new[f'{c}_latency'].mean():.2f}s")
