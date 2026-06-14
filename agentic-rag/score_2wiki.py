"""
score_2wiki.py — Revision Chunk G (R3 part 3): SCORE + MERGE only.

Consumes results/generation_cache_2wiki.csv (100 rows, produced by Chunk F /
gen_2wiki.py) and runs RAGAS scoring for all three conditions with judge
config.LLM_MODEL (llama3.1:8b), batches of 15, checkpointed per batch to
results/scores_2wiki.csv (resume skips condition+question pairs already on
disk). Then merges wide per-query table + ** AGGREGATE MEAN ** row ->
results/results_2wiki.csv.

This is phase2_score + phase3_merge of run_2wiki.py ONLY. It does NOT run
generation (phase1) — the cache already has 100/100 rows. Importing run_2wiki
applies its module-level config (CHROMA_COLLECTION='corpus_2wiki', TAU=0.40),
which does not affect scoring but keeps a single source of truth.

Usage (detached):  python score_2wiki.py  > results/2wiki_score.log 2>&1
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd

import config
from run_2wiki import phase2_score, phase3_merge, CACHE


def main():
    print(f"[score_2wiki] CHUNK G score+merge  judge={config.LLM_MODEL}  "
          f"cache={CACHE}", flush=True)
    assert config.LLM_MODEL == "llama3.1:8b", \
        f"UNEXPECTED JUDGE: {config.LLM_MODEL!r} (expected llama3.1:8b)"
    n = len(pd.read_csv(CACHE))
    assert n == 100, f"cache has {n} rows, need 100 — run Chunk F first"

    phase2_score()
    phase3_merge()
    print("[score_2wiki] CHUNK G COMPLETE — scores_2wiki.csv + results_2wiki.csv",
          flush=True)


if __name__ == "__main__":
    main()
