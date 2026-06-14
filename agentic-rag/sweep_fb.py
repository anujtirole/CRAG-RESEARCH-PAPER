"""
sweep_fb.py — Revision Stage R1: threshold sweep WITH fallback (agentic only).

SUBSET: the FIRST 150 queries of data/queries_hotpot_300.json (indices 0-149).
tau=0.40 is NOT re-run here — it is taken from the main n=300 results filtered
to the same first-150 queries (see sweep_fb_assemble.py).

Phase 1 (generation): for each of the 150 queries not yet in the cache, runs
  the CURRENT full agentic pipeline (merge-dedup loop + best-effort fallback)
  with the given --tau, appending ONE row per query immediately (kill-safe).
  Cache: results/sweep_fb_tau{XXX}_cache.csv
Phase 2 (RAGAS scoring): judge llama3.1:8b, same wrapper/settings as
  evaluate.py / score300.py, batches of 15, checkpointed per batch to
  results/sweep_fb_tau{XXX}_scores.csv; restart skips questions already scored.
Phase 3 (merge): when all 150 are scored, metric columns are merged INTO the
  cache file (faithfulness, answer_relevancy, context_precision).

Both phases run in this one process so a single detached launch per tau keeps
Ollama usage strictly sequential.

Usage (detached):  python sweep_fb.py --tau 0.50  > results/sweep_fb_tau050.log
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd

QUERIES_300 = Path("data/queries_hotpot_300.json")
N_SUBSET = 150
GEN_COLUMNS = [
    "question", "ground_truth",
    "agentic_answer", "agentic_contexts", "agentic_latency",
    "agentic_n_attempts", "agentic_confidence", "agentic_low_conf",
    "agentic_used_fallback",
]
METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision"]
BATCH = 15


def phase1_generate(queries, cache_path, tau):
    from retriever import Retriever
    from agentic_rag import run_agentic

    if cache_path.exists():
        cache_df = pd.read_csv(cache_path)
        done = set(cache_df["question"].tolist())
    else:
        with open(cache_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(GEN_COLUMNS)
        done = set()

    todo = [q for q in queries if q["question"] not in done]
    print(f"[sweep_fb] tau={tau}  SUBSET=first {N_SUBSET} queries of "
          f"queries_hotpot_300.json; cached={len(done)} todo={len(todo)}",
          flush=True)
    if not todo:
        print("[sweep_fb] phase 1 already complete.", flush=True)
        return

    retriever = Retriever()
    t0 = time.time()
    for j, item in enumerate(todo):
        q = item["question"]
        gt = item.get("ground_truth", "")
        print(f"\n[sweep_fb] gen {j + 1}/{len(todo)}  {q[:90]}", flush=True)
        try:
            a = run_agentic(q, retriever, tau=tau)
        except Exception as exc:
            print(f"  [agentic] ERROR: {exc}", flush=True)
            a = {"answer": "", "contexts": [], "latency_s": float("nan"),
                 "n_attempts": 0, "final_confidence": float("nan"),
                 "low_confidence": True, "used_fallback": False}
        row = [q, gt, a["answer"], json.dumps(a["contexts"]), a["latency_s"],
               a["n_attempts"], a["final_confidence"], a["low_confidence"],
               a["used_fallback"]]
        with open(cache_path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(row)
        rate = (time.time() - t0) / (j + 1)
        print(f"  [sweep_fb] row appended. avg {rate:.1f}s/query, "
              f"ETA {rate * (len(todo) - j - 1) / 60:.0f} min", flush=True)
    print(f"\n[sweep_fb] phase 1 COMPLETE in {(time.time() - t0) / 60:.1f} min",
          flush=True)


def phase2_score(cache_path, scores_path):
    from evaluate import (_build_ragas_llm_and_embeddings, _compute_ragas,
                          _METRIC_KEYS)
    assert list(_METRIC_KEYS) == METRIC_COLS

    cache = pd.read_csv(cache_path)
    assert len(cache) == N_SUBSET, \
        f"cache has {len(cache)} rows, need {N_SUBSET} before scoring"

    if scores_path.exists():
        sc = pd.read_csv(scores_path)
        done = set(sc["question"].tolist())
        print(f"[sweep_fb] scoring resume: {len(done)} rows already scored",
              flush=True)
    else:
        scores_path.write_text("question," + ",".join(METRIC_COLS) + "\n",
                               encoding="utf-8")
        done = set()

    todo = cache[~cache["question"].isin(done)].reset_index(drop=True)
    if len(todo) == 0:
        print("[sweep_fb] phase 2 already complete.", flush=True)
        return

    llm_w, emb_w = _build_ragas_llm_and_embeddings()
    if llm_w is None:
        print("[sweep_fb] ABORT: RAGAS LLM unavailable.", flush=True)
        sys.exit(1)

    n_batches = (len(todo) + BATCH - 1) // BATCH
    t0 = time.time()
    for bi in range(n_batches):
        sl = todo.iloc[bi * BATCH:(bi + 1) * BATCH]
        qs = sl["question"].tolist()
        ans = sl["agentic_answer"].fillna("").tolist()
        ctxs = [json.loads(x) for x in sl["agentic_contexts"]]
        gts = sl["ground_truth"].fillna("").tolist()
        print(f"[sweep_fb] score batch {bi + 1}/{n_batches} "
              f"(elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)
        per_df, _, _ = _compute_ragas(qs, ans, ctxs, gts, llm_w, emb_w,
                                      batch_size=BATCH)
        out = pd.DataFrame({"question": qs})
        for m in METRIC_COLS:
            out[m] = per_df[m].values
        out.to_csv(scores_path, mode="a", header=False, index=False)
    print(f"[sweep_fb] phase 2 COMPLETE in {(time.time() - t0) / 60:.1f} min",
          flush=True)


def phase3_merge(cache_path, scores_path):
    cache = pd.read_csv(cache_path)
    sc = pd.read_csv(scores_path).drop_duplicates("question", keep="last")
    assert len(sc) >= N_SUBSET, f"only {len(sc)} scored rows"
    merged = cache.drop(columns=[c for c in METRIC_COLS if c in cache.columns])
    merged = merged.merge(sc, on="question", how="left")
    assert len(merged) == N_SUBSET
    merged.to_csv(cache_path, index=False)
    print(f"[sweep_fb] metrics merged into {cache_path}", flush=True)
    for m in METRIC_COLS:
        print(f"  mean {m}: {merged[m].mean(skipna=True):.4f} "
              f"(n={int(merged[m].notna().sum())})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, required=True)
    args = ap.parse_args()
    tag = f"{args.tau:.2f}".replace(".", "")[:3]  # 0.50 -> 050
    cache_path = Path(f"results/sweep_fb_tau{tag}_cache.csv")
    scores_path = Path(f"results/sweep_fb_tau{tag}_scores.csv")

    queries = json.loads(QUERIES_300.read_text(encoding="utf-8"))[:N_SUBSET]
    assert len(queries) == N_SUBSET

    phase1_generate(queries, cache_path, args.tau)
    phase2_score(cache_path, scores_path)
    phase3_merge(cache_path, scores_path)
    print(f"[sweep_fb] ALL DONE tau={args.tau}", flush=True)


if __name__ == "__main__":
    main()
