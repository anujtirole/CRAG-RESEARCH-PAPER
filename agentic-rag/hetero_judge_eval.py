"""
hetero_judge_eval.py — Stage 4: heterogeneous-judge validation.

- random.Random(42).sample over the 300-row cache -> 50 query indices.
- RAGAS-scores all three conditions on those 50 rows using CACHED answers and
  contexts (no regeneration), with judge = qwen2.5:7b via the same
  LangchainLLMWrapper/ChatOllama pattern as evaluate.py (max_workers=1) and
  the same local HuggingFace embeddings.
- CHECKPOINT PER BATCH (batch=10): appended to results/hetero_judge_50.csv as
  (question, condition, faithfulness, answer_relevancy, context_precision);
  restart skips (condition, question) pairs already present.

Run DETACHED: output to results/hetero.log via shell redirection.
"""

import json
import random
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd

import config
config.LLM_MODEL = "qwen2.5:7b"  # judge override BEFORE building the wrapper

from evaluate import (_build_ragas_llm_and_embeddings, _compute_ragas,
                      _METRIC_KEYS)

CACHE = Path("results/generation_cache.csv")
OUT = Path("results/hetero_judge_50.csv")
BATCH = 10
CONDITIONS = ["baseline", "reranker", "agentic"]


def main():
    cache = pd.read_csv(CACHE)
    assert len(cache) == 300, f"expected 300 cache rows, got {len(cache)}"

    idxs = sorted(random.Random(42).sample(range(300), 50))
    sample = cache.iloc[idxs].reset_index(drop=True)
    print(f"[hetero] judge={config.LLM_MODEL}; sampled 50 rows "
          f"(seed=42), indices {idxs[:6]}...{idxs[-3:]}", flush=True)

    if OUT.exists():
        sc = pd.read_csv(OUT)
        done = set(zip(sc["condition"], sc["question"]))
        print(f"[hetero] resuming: {len(sc)} scores on disk", flush=True)
    else:
        OUT.write_text("question,condition," + ",".join(_METRIC_KEYS) + "\n",
                       encoding="utf-8")
        done = set()

    llm_w, emb_w = _build_ragas_llm_and_embeddings()
    if llm_w is None:
        print("[hetero] ABORT: RAGAS LLM unavailable.", flush=True)
        sys.exit(1)

    t0 = time.time()
    for cond in CONDITIONS:
        todo = sample[~sample["question"].isin(
            {q for c, q in done if c == cond})].reset_index(drop=True)
        n_batches = (len(todo) + BATCH - 1) // BATCH
        print(f"\n[hetero] condition={cond}: {len(todo)} rows "
              f"({n_batches} batches of {BATCH})", flush=True)

        for bi in range(n_batches):
            sl = todo.iloc[bi * BATCH:(bi + 1) * BATCH]
            qs = sl["question"].tolist()
            ans = sl[f"{cond}_answer"].fillna("").tolist()
            ctxs = [json.loads(x) for x in sl[f"{cond}_contexts"]]
            gts = sl["ground_truth"].fillna("").tolist()

            print(f"[hetero] {cond} batch {bi + 1}/{n_batches} "
                  f"(elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)
            per_df, _, _ = _compute_ragas(qs, ans, ctxs, gts, llm_w, emb_w,
                                          batch_size=BATCH)

            out = pd.DataFrame({"question": qs, "condition": cond})
            for m in _METRIC_KEYS:
                out[m] = per_df[m].values
            out.to_csv(OUT, mode="a", header=False, index=False)

    print(f"\n[hetero] COMPLETE in {(time.time() - t0) / 60:.1f} min "
          f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
