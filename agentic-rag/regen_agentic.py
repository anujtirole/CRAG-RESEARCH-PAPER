"""
Regenerate ONLY the agentic condition (with best-effort fallback) into
results/generation_cache.csv.

- Loads data/queries_hotpot.json (150 queries) and the existing cache.
- Runs run_agentic fresh per query at τ=0.40 (the τ used by the main run).
- Overwrites ONLY: agentic_answer, agentic_contexts, agentic_latency,
  agentic_n_attempts, agentic_confidence, agentic_low_conf,
  agentic_used_fallback (new column).
- Baseline/reranker columns pass through byte-for-byte unchanged.
- Checkpoints the cache to disk after EVERY query; a progress file
  (results/regen_agentic_progress.txt) allows resuming after a crash.

Usage:  python regen_agentic.py
"""

import json
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd

import config
from agentic_rag import run_agentic
from retriever import Retriever

TAU = 0.40  # τ of the cached main run — must match for a valid comparison

CACHE_PATH    = config.RESULTS_DIR / "generation_cache.csv"
PROGRESS_PATH = config.RESULTS_DIR / "regen_agentic_progress.txt"

AGENTIC_COLS = [
    "agentic_answer", "agentic_contexts", "agentic_latency",
    "agentic_n_attempts", "agentic_confidence", "agentic_low_conf",
    "agentic_used_fallback",
]


def main() -> None:
    with open(config.QUERIES_FILE, encoding="utf-8") as fh:
        queries = json.load(fh)
    df = pd.read_csv(CACHE_PATH)
    assert len(df) == len(queries), (
        f"cache rows ({len(df)}) != queries ({len(queries)})"
    )
    for i in range(len(queries)):
        assert df["question"].iloc[i] == queries[i]["question"], (
            f"row {i}: cache question does not match queries_hotpot.json — aborting"
        )
    print(f"[regen] {len(queries)} queries aligned with cache; τ={TAU}")

    if "agentic_used_fallback" not in df.columns:
        df["agentic_used_fallback"] = pd.NA

    start_idx = 0
    if PROGRESS_PATH.exists():
        start_idx = int(PROGRESS_PATH.read_text().strip() or 0)
        print(f"[regen] resuming at row {start_idx}")

    retriever = Retriever()
    t_start = time.time()

    for i in range(start_idx, len(queries)):
        q = queries[i]["question"]
        print(f"\n[regen] {i + 1}/{len(queries)}  {q[:70]}", flush=True)
        a = run_agentic(q, retriever, tau=TAU)

        df.loc[i, "agentic_answer"]        = a["answer"]
        df.loc[i, "agentic_contexts"]      = json.dumps(a["contexts"])
        df.loc[i, "agentic_latency"]       = a["latency_s"]
        df.loc[i, "agentic_n_attempts"]    = a["n_attempts"]
        df.loc[i, "agentic_confidence"]    = a["final_confidence"]
        df.loc[i, "agentic_low_conf"]      = a["low_confidence"]
        df.loc[i, "agentic_used_fallback"] = a["used_fallback"]

        # Checkpoint after every query
        df.to_csv(CACHE_PATH, index=False)
        PROGRESS_PATH.write_text(str(i + 1))

        if a["used_fallback"]:
            print(f"[regen]   → fallback used")

    elapsed = time.time() - t_start
    n_fb = df["agentic_used_fallback"].map(
        lambda x: str(x).strip().lower() == "true").sum()
    n_refusal = df["agentic_answer"].astype(str).str.contains(
        "does not contain enough information", case=False).sum()
    print(f"\n[regen] DONE in {elapsed / 60:.1f} min")
    print(f"[regen] fallback used: {n_fb}/{len(df)}")
    print(f"[regen] remaining refusals: {n_refusal}/{len(df)}")


if __name__ == "__main__":
    main()
