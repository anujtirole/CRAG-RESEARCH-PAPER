"""
run_2wiki.py — Revision Stage R3 step 3: run + score all conditions on 2wiki.

Phase 1 (generation): all three conditions (baseline, reranker, full SC-ARAG
  tau=0.40 with fallback) over the 100 queries in data/queries_2wiki_100.json,
  retrieving from the SEPARATE ChromaDB collection 'corpus_2wiki'.
  One row appended per query (kill-safe) -> results/generation_cache_2wiki.csv
  (same columns as the main generation cache).
Phase 2 (RAGAS scoring): judge llama3.1:8b, same wrapper/settings as
  evaluate.py / score300.py, batches of 15 per condition, checkpointed per
  batch to results/scores_2wiki.csv; restart skips pairs already present.
Phase 3 (merge): wide per-query table + ** AGGREGATE MEAN ** row ->
  results/results_2wiki.csv (same format as results_main.csv).

Single process so one detached launch keeps Ollama usage strictly sequential.

Usage (detached):  python run_2wiki.py  > results/2wiki_run.log
"""

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

import config
config.TAU = 0.40                        # published setting, with fallback
config.CHROMA_COLLECTION = "corpus_2wiki"  # SEPARATE collection

QUERIES = Path("data/queries_2wiki_100.json")
CACHE = Path("results/generation_cache_2wiki.csv")
SCORES = Path("results/scores_2wiki.csv")
MAIN = Path("results/results_2wiki.csv")
BATCH = 15
CONDITIONS = ["baseline", "reranker", "agentic"]

COLUMNS = [
    "question", "ground_truth",
    "baseline_answer", "baseline_contexts", "baseline_latency",
    "reranker_answer", "reranker_contexts", "reranker_latency",
    "agentic_answer", "agentic_contexts", "agentic_latency",
    "agentic_n_attempts", "agentic_confidence", "agentic_low_conf",
    "agentic_used_fallback",
]


def phase1_generate(queries):
    from retriever import Retriever
    from baseline_rag import run_baseline
    from reranker_rag import RerankerRAG
    from agentic_rag import run_agentic

    if CACHE.exists():
        done = set(pd.read_csv(CACHE)["question"].tolist())
    else:
        with open(CACHE, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(COLUMNS)
        done = set()

    todo = [q for q in queries if q["question"] not in done]
    print(f"[2wiki] tau={config.TAU} collection={config.CHROMA_COLLECTION}  "
          f"cached={len(done)} todo={len(todo)}/100", flush=True)
    if not todo:
        print("[2wiki] phase 1 already complete.", flush=True)
        return

    retriever = Retriever()
    reranker = RerankerRAG()
    t0 = time.time()
    for j, item in enumerate(todo):
        q = item["question"]
        gt = item.get("ground_truth", "")
        print(f"\n[2wiki] gen {j + 1}/{len(todo)}  {q[:90]}", flush=True)

        try:
            b = run_baseline(q, retriever)
        except Exception as exc:
            print(f"  [baseline] ERROR: {exc}", flush=True)
            b = {"answer": "", "contexts": [], "latency_s": float("nan")}
        try:
            r = reranker.run(q, retriever)
        except Exception as exc:
            print(f"  [reranker] ERROR: {exc}", flush=True)
            r = {"answer": "", "contexts": [], "latency_s": float("nan")}
        try:
            a = run_agentic(q, retriever)
        except Exception as exc:
            print(f"  [agentic]  ERROR: {exc}", flush=True)
            a = {"answer": "", "contexts": [], "latency_s": float("nan"),
                 "n_attempts": 0, "final_confidence": float("nan"),
                 "low_confidence": True, "used_fallback": False}

        row = [q, gt,
               b["answer"], json.dumps(b["contexts"]), b["latency_s"],
               r["answer"], json.dumps(r["contexts"]), r["latency_s"],
               a["answer"], json.dumps(a["contexts"]), a["latency_s"],
               a["n_attempts"], a["final_confidence"], a["low_confidence"],
               a["used_fallback"]]
        with open(CACHE, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(row)
        rate = (time.time() - t0) / (j + 1)
        print(f"  [2wiki] row appended. avg {rate:.1f}s/query, "
              f"ETA {rate * (len(todo) - j - 1) / 60:.0f} min", flush=True)
    print(f"\n[2wiki] phase 1 COMPLETE in {(time.time() - t0) / 60:.1f} min",
          flush=True)


def phase2_score():
    from evaluate import (_build_ragas_llm_and_embeddings, _compute_ragas,
                          _METRIC_KEYS)
    cache = pd.read_csv(CACHE)
    assert len(cache) == 100, f"cache has {len(cache)} rows, need 100"

    if SCORES.exists():
        sc = pd.read_csv(SCORES)
        done = set(zip(sc["condition"], sc["question"]))
        print(f"[2wiki] scoring resume: {len(sc)} scores on disk", flush=True)
    else:
        SCORES.write_text("question,condition," + ",".join(_METRIC_KEYS) + "\n",
                          encoding="utf-8")
        done = set()

    llm_w, emb_w = _build_ragas_llm_and_embeddings()
    if llm_w is None:
        print("[2wiki] ABORT: RAGAS LLM unavailable.", flush=True)
        sys.exit(1)

    t0 = time.time()
    for cond in CONDITIONS:
        todo = cache[~cache["question"].isin(
            {q for c, q in done if c == cond})].reset_index(drop=True)
        n_batches = (len(todo) + BATCH - 1) // BATCH
        print(f"\n[2wiki] scoring condition={cond}: {len(todo)} rows "
              f"({n_batches} batches of {BATCH})", flush=True)
        for bi in range(n_batches):
            sl = todo.iloc[bi * BATCH:(bi + 1) * BATCH]
            qs = sl["question"].tolist()
            ans = sl[f"{cond}_answer"].fillna("").tolist()
            ctxs = [json.loads(x) for x in sl[f"{cond}_contexts"]]
            gts = sl["ground_truth"].fillna("").tolist()
            print(f"[2wiki] {cond} batch {bi + 1}/{n_batches} "
                  f"(elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)
            per_df, _, _ = _compute_ragas(qs, ans, ctxs, gts, llm_w, emb_w,
                                          batch_size=BATCH)
            out = pd.DataFrame({"question": qs, "condition": cond})
            for m in _METRIC_KEYS:
                out[m] = per_df[m].values
            out.to_csv(SCORES, mode="a", header=False, index=False)
    print(f"\n[2wiki] phase 2 COMPLETE in {(time.time() - t0) / 60:.1f} min",
          flush=True)


def phase3_merge():
    from evaluate import _METRIC_KEYS
    cache = pd.read_csv(CACHE)
    sc = pd.read_csv(SCORES)
    merged = cache.copy()
    for cond in CONDITIONS:
        sub = sc[sc["condition"] == cond].drop_duplicates("question",
                                                          keep="last")
        for met in _METRIC_KEYS:
            merged[f"{cond}_{met}"] = merged["question"].map(
                sub.set_index("question")[met])
    assert len(merged) == 100

    agg = {"question": "** AGGREGATE MEAN **", "ground_truth": ""}
    for cond in CONDITIONS:
        for col in ([f"{cond}_latency"]
                    + [f"{cond}_{m}" for m in _METRIC_KEYS]):
            agg[col] = merged[col].mean(skipna=True)
    for col in ["agentic_n_attempts", "agentic_confidence"]:
        agg[col] = merged[col].mean(skipna=True)
    final = pd.concat([merged, pd.DataFrame([agg])], ignore_index=True)
    final.to_csv(MAIN, index=False)
    print(f"[2wiki] {MAIN} written: 100 scored rows + aggregate", flush=True)
    for cond in CONDITIONS:
        means = {m: merged[f'{cond}_{m}'].mean(skipna=True)
                 for m in _METRIC_KEYS}
        print(f"  {cond}: " + "  ".join(f"{m}={v:.4f}"
                                        for m, v in means.items()), flush=True)


def main():
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    assert len(queries) == 100
    phase1_generate(queries)
    phase2_score()
    phase3_merge()
    print("[2wiki] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
