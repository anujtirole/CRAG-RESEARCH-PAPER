"""Stage 5 helper: qwen2.5:7b vs llama3.1:8b judge comparison on the same 50 queries."""
import random

import pandas as pd

cache = pd.read_csv("results/generation_cache.csv")
idxs = sorted(random.Random(42).sample(range(300), 50))
sample_qs = cache.iloc[idxs]["question"].tolist()

qwen = pd.read_csv("results/hetero_judge_50.csv")
main = pd.read_csv("results/results_main.csv")
main = main[main["question"] != "** AGGREGATE MEAN **"]
llama = main[main["question"].isin(sample_qs)]
print(f"sample questions: {len(sample_qs)}; qwen rows: {len(qwen)}; "
      f"llama rows matched: {len(llama)}")

print(f"\n{'condition':<10} {'metric':<20} {'llama3.1:8b':>12} {'qwen2.5:7b':>12} "
      f"{'qwen_nonNaN':>12}")
for cond in ["baseline", "reranker", "agentic"]:
    qc = qwen[qwen["condition"] == cond]
    for m in ["faithfulness", "answer_relevancy", "context_precision"]:
        lm = llama[f"{cond}_{m}"].mean()
        qm = qc[m].mean()
        print(f"{cond:<10} {m:<20} {lm:>12.4f} {qm:>12.4f} "
              f"{int(qc[m].notna().sum()):>9}/50")
