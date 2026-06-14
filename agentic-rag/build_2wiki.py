"""
build_2wiki.py — Revision Stage R3 step 1: build the 2WikiMultiHopQA benchmark.

Source: xanhho/2WikiMultihopQA on HuggingFace — the dataset release by the
first author of the 2WikiMultiHopQA paper (Ho et al., COLING 2020), i.e. the
canonical data. Its legacy loading script is not runnable under datasets 4.x,
so dev.parquet is read DIRECTLY from the same repo (identical content; this is
not a substitute dataset). dev split = 12,576 rows; each row provides 10
context paragraphs as [title, sentences] — gold + distractors, the same
structure as HotpotQA distractor.

- Samples 100 questions with random.Random(42) over the dev split.
- Writes each sampled question's 10 paragraphs to data/corpus_2wiki/ using the
  identical title-hash filename scheme as build_datasets.py / build_300.py
  (dedup by title across questions).
- Writes queries + ground truths to data/queries_2wiki_100.json.
- Reports corpus doc count and verifies no empty contexts / answers.

ABORTS (rather than substituting silently) if the download or any structural
check fails. No Ollama usage.
"""

import hashlib
import json
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from huggingface_hub import hf_hub_download

DATA = Path("data")
CORPUS = DATA / "corpus_2wiki"
OUT = DATA / "queries_2wiki_100.json"
N_SAMPLE = 100
REPO = "xanhho/2WikiMultihopQA"


def main():
    print(f"Downloading dev.parquet from {REPO} (canonical author repo) ...")
    try:
        p = hf_hub_download(REPO, "dev.parquet", repo_type="dataset")
    except Exception as exc:
        print(f"ABORT: could not download canonical dataset: {exc}")
        sys.exit(1)
    df = pd.read_parquet(p)
    print(f"dev split rows: {len(df)}")
    required = {"question", "answer", "context", "type"}
    missing = required - set(df.columns)
    if missing:
        print(f"ABORT: dataset structure unexpected, missing columns {missing}")
        sys.exit(1)

    idxs = sorted(random.Random(42).sample(range(len(df)), N_SAMPLE))
    sample = df.iloc[idxs].reset_index(drop=True)
    print(f"sampled {N_SAMPLE} questions (seed=42), "
          f"dataset indices {idxs[:5]}...{idxs[-3:]}")

    # ── Structural verification BEFORE writing anything ──────────────────────
    n_empty_ctx = 0
    n_empty_ans = 0
    parsed_ctxs = []
    for _, r in sample.iterrows():
        ctx = json.loads(r["context"])
        if not ctx or any(not title.strip() or not sents
                          for title, sents in ctx):
            n_empty_ctx += 1
        if not str(r["answer"]).strip():
            n_empty_ans += 1
        parsed_ctxs.append(ctx)
    print(f"verification: empty/malformed contexts = {n_empty_ctx}, "
          f"empty answers = {n_empty_ans}")
    if n_empty_ctx or n_empty_ans:
        print("ABORT: empty contexts or answers in sample.")
        sys.exit(1)
    uq = sample["question"].nunique()
    assert uq == N_SAMPLE, f"only {uq} unique question texts"

    # ── Write corpus docs (dedup by title, same scheme as build_300.py) ──────
    CORPUS.mkdir(parents=True, exist_ok=True)
    n_new, n_already = 0, 0
    for ctx in parsed_ctxs:
        for title, sents in ctx:
            key = title.strip()
            fn = "doc_" + hashlib.md5(key.encode()).hexdigest()[:12] + ".txt"
            path = CORPUS / fn
            if path.exists():
                n_already += 1
                continue
            path.write_text(f"{title}\n\n" + " ".join(sents), encoding="utf-8")
            n_new += 1
    total_docs = len(list(CORPUS.glob("*.txt")))
    print(f"corpus_2wiki: wrote {n_new} new docs "
          f"({n_already} paragraph slots deduped by title); "
          f"total docs = {total_docs}")

    # ── Save queries file ─────────────────────────────────────────────────────
    items = [{"question": r["question"],
              "ground_truth": str(r["answer"]),
              "type": r["type"]}
             for _, r in sample.iterrows()]
    OUT.write_text(json.dumps(items, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"saved {len(items)} queries -> {OUT}")
    print("type distribution:",
          sample["type"].value_counts().to_dict())
    print("\nNEXT: python ingest_2wiki.py  (separate ChromaDB collection "
          "'corpus_2wiki'; HotpotQA collection untouched)")


if __name__ == "__main__":
    main()
