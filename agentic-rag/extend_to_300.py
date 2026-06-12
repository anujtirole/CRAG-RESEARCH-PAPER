"""
extend_to_300.py — Stage 3 step 2: generate conditions for queries 150-299 ONLY.

- Loads data/queries_hotpot_300.json (300 queries; 0-149 already in cache).
- For each query NOT already present in results/generation_cache.csv, runs all
  three conditions (baseline, reranker, agentic-with-fallback; tau=0.40, the
  published setting) and APPENDS one row to the cache immediately afterwards.
- Checkpoint = the cache file itself: on restart, rows already present are
  skipped (matched by exact question text), so the process is kill-safe.
- Appends with csv.writer in the exact existing column order; never rewrites
  prior rows.

Run DETACHED:  output goes to results/extend300.log via shell redirection.
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
config.TAU = 0.40  # match the published n=150 runs (evaluate.py --tau 0.40)

from retriever import Retriever
from baseline_rag import run_baseline
from reranker_rag import RerankerRAG
from agentic_rag import run_agentic

QUERIES_300 = Path("data/queries_hotpot_300.json")
CACHE = Path("results/generation_cache.csv")

COLUMNS = [
    "question", "ground_truth",
    "baseline_answer", "baseline_contexts", "baseline_latency",
    "reranker_answer", "reranker_contexts", "reranker_latency",
    "agentic_answer", "agentic_contexts", "agentic_latency",
    "agentic_n_attempts", "agentic_confidence", "agentic_low_conf",
    "agentic_used_fallback",
]


def main():
    queries = json.loads(QUERIES_300.read_text(encoding="utf-8"))
    assert len(queries) == 300

    cache_df = pd.read_csv(CACHE)
    assert list(cache_df.columns) == COLUMNS, (
        f"cache column mismatch:\n{list(cache_df.columns)}\nvs\n{COLUMNS}")
    done = set(cache_df["question"].tolist())
    todo = [q for q in queries[150:] if q["question"] not in done]
    print(f"[extend300] cache rows: {len(cache_df)}  "
          f"todo: {len(todo)}/150 new queries  tau={config.TAU}", flush=True)
    if not todo:
        print("[extend300] nothing to do — all 150 new queries already cached.",
              flush=True)
        return

    retriever = Retriever()
    reranker = RerankerRAG()

    t_start = time.time()
    for j, item in enumerate(todo):
        q = item["question"]
        gt = item.get("ground_truth", "")
        print(f"\n[extend300] {j + 1}/{len(todo)}  {q[:90]}", flush=True)

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

        row = [
            q, gt,
            b["answer"], json.dumps(b["contexts"]), b["latency_s"],
            r["answer"], json.dumps(r["contexts"]), r["latency_s"],
            a["answer"], json.dumps(a["contexts"]), a["latency_s"],
            a["n_attempts"], a["final_confidence"], a["low_confidence"],
            a["used_fallback"],
        ]
        # Append-only checkpoint: one row per query, flushed before next query.
        with open(CACHE, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(row)

        elapsed = time.time() - t_start
        rate = elapsed / (j + 1)
        print(f"  [extend300] row appended ({len(cache_df) + j + 1} total). "
              f"avg {rate:.1f}s/query, ETA {rate * (len(todo) - j - 1) / 60:.0f} min",
              flush=True)

    print(f"\n[extend300] COMPLETE: all {len(todo)} new queries generated in "
          f"{(time.time() - t_start) / 60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
