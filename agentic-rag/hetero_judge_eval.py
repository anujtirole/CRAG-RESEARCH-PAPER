"""
hetero_judge_eval.py — heterogeneous-judge validation (extended 50 -> 150).

Sampling (seed=42, reproduces the original Stage-4 sample exactly):
  - idxs50  = sorted(random.Random(42).sample(range(300), 50))   # original 50
  - idxs100 = sorted(random.Random(42).sample(remainder, 100))   # 100 NEW
  - the 150-query sample = union; the original 50 are a strict subset.

Scoring: all three conditions on CACHED answers/contexts from
results/generation_cache.csv (no regeneration), judge = qwen2.5:7b via the
same LangchainLLMWrapper/ChatOllama pattern as evaluate.py (max_workers=1)
and the same local HuggingFace embeddings.

Already-scored rows in results/hetero_judge_50.csv are COPIED into
results/hetero_judge_150.csv on first run and skipped (no re-scoring).
CHECKPOINT PER BATCH (batch=10): appended to results/hetero_judge_150.csv;
restart skips (condition, question) pairs already present.

Run DETACHED: output to results/hetero150.log via shell redirection.
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
SEED50 = Path("results/hetero_judge_50.csv")
OUT = Path("results/hetero_judge_150.csv")
BATCH = 1  # checkpoint after EVERY query (battery-safe resume)
CONDITIONS = ["baseline", "reranker", "agentic"]


def sample_indices():
    idxs50 = sorted(random.Random(42).sample(range(300), 50))
    remainder = [i for i in range(300) if i not in set(idxs50)]
    idxs100 = sorted(random.Random(42).sample(remainder, 100))
    idxs150 = sorted(set(idxs50) | set(idxs100))
    assert len(idxs150) == 150 and set(idxs50) <= set(idxs150)
    return idxs50, idxs150


def main():
    cache = pd.read_csv(CACHE)
    assert len(cache) == 300, f"expected 300 cache rows, got {len(cache)}"

    idxs50, idxs150 = sample_indices()
    sample = cache.iloc[idxs150].reset_index(drop=True)
    print(f"[hetero150] judge={config.LLM_MODEL}; 150-query sample (seed=42) = "
          f"original 50 + 100 new; indices {idxs150[:5]}...{idxs150[-3:]}",
          flush=True)

    if not OUT.exists():
        # Seed the 150-file with the 50 rows already scored by qwen (Stage 4).
        header = "question,condition," + ",".join(_METRIC_KEYS) + "\n"
        OUT.write_text(header, encoding="utf-8")
        if SEED50.exists():
            seed = pd.read_csv(SEED50)
            seed = seed[["question", "condition"] + list(_METRIC_KEYS)]
            seed.to_csv(OUT, mode="a", header=False, index=False)
            print(f"[hetero150] pre-seeded {len(seed)} rows from {SEED50}",
                  flush=True)

    sc = pd.read_csv(OUT)
    done = set(zip(sc["condition"], sc["question"]))
    print(f"[hetero150] {len(sc)} scores on disk; "
          f"target = {150 * len(CONDITIONS)}", flush=True)

    llm_w, emb_w = _build_ragas_llm_and_embeddings()
    if llm_w is None:
        print("[hetero150] ABORT: RAGAS LLM unavailable.", flush=True)
        sys.exit(1)

    t0 = time.time()
    for cond in CONDITIONS:
        todo = sample[~sample["question"].isin(
            {q for c, q in done if c == cond})].reset_index(drop=True)
        n_batches = (len(todo) + BATCH - 1) // BATCH
        print(f"\n[hetero150] condition={cond}: {len(todo)} rows "
              f"({n_batches} batches of {BATCH})", flush=True)

        for bi in range(n_batches):
            sl = todo.iloc[bi * BATCH:(bi + 1) * BATCH]
            qs = sl["question"].tolist()
            ans = sl[f"{cond}_answer"].fillna("").tolist()
            ctxs = [json.loads(x) for x in sl[f"{cond}_contexts"]]
            gts = sl["ground_truth"].fillna("").tolist()

            print(f"[hetero150] {cond} batch {bi + 1}/{n_batches} "
                  f"(elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)
            per_df, _, _ = _compute_ragas(qs, ans, ctxs, gts, llm_w, emb_w,
                                          batch_size=BATCH)

            out = pd.DataFrame({"question": qs, "condition": cond})
            for m in _METRIC_KEYS:
                out[m] = per_df[m].values
            out.to_csv(OUT, mode="a", header=False, index=False)

    print(f"\n[hetero150] COMPLETE in {(time.time() - t0) / 60:.1f} min "
          f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
