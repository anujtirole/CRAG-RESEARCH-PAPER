"""
gen_2wiki.py — Revision Chunk F (R3 part 2): GENERATION ONLY.

Runs ONLY phase 1 (generation) of run_2wiki.py over the 100 2wiki queries:
baseline, reranker, and full SC-ARAG (tau=0.40, fallback enabled), retrieving
from the SEPARATE ChromaDB collection 'corpus_2wiki'. One row is appended per
query (kill-safe) -> results/generation_cache_2wiki.csv.

Importing run_2wiki applies its module-level config (TAU=0.40,
CHROMA_COLLECTION='corpus_2wiki'); we re-assert both here before launching so a
misconfigured run aborts instead of silently hitting the HotpotQA collection.

This script does NOT score and does NOT merge — scoring is Chunk G, a later
session. It stops as soon as generation reaches 100/100. Resuming is automatic:
phase1_generate skips any question already present in the cache.

Usage (detached):  python gen_2wiki.py  > results/2wiki_gen.log 2>&1
"""

import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import config
# phase1_generate + the config side-effects (TAU, collection) live in run_2wiki.
from run_2wiki import phase1_generate, QUERIES, CACHE


def main():
    print(f"[gen_2wiki] CHUNK F generation-only  tau={config.TAU}  "
          f"collection={config.CHROMA_COLLECTION}  cache={CACHE}", flush=True)
    assert config.CHROMA_COLLECTION == "corpus_2wiki", \
        f"WRONG COLLECTION: {config.CHROMA_COLLECTION!r} (must be corpus_2wiki)"
    assert abs(config.TAU - 0.40) < 1e-9, \
        f"WRONG TAU: {config.TAU!r} (must be 0.40)"

    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    assert len(queries) == 100, f"expected 100 queries, got {len(queries)}"

    phase1_generate(queries)

    print("[gen_2wiki] GENERATION COMPLETE — STOP. "
          "Scoring (Chunk G) is a separate session.", flush=True)


if __name__ == "__main__":
    main()
