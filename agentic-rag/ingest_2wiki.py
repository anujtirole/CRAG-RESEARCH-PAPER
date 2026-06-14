"""
ingest_2wiki.py — Revision Stage R3 step 2: index the 2wiki corpus.

Reuses ingest.build_index unchanged, but pointed at data/corpus_2wiki/ and a
SEPARATE ChromaDB collection 'corpus_2wiki' (same persistent chroma_db/ dir,
same chunking 640/128, same BGE-base embedder). The HotpotQA collection
'rag_corpus' is not touched.

Usage:  python ingest_2wiki.py [--reset]
"""

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import config
config.CORPUS_DIR = config.DATA_DIR / "corpus_2wiki"
config.CHROMA_COLLECTION = "corpus_2wiki"

from ingest import build_index

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    print(f"[ingest_2wiki] corpus={config.CORPUS_DIR}  "
          f"collection={config.CHROMA_COLLECTION}")
    build_index(reset=args.reset)
