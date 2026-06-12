"""
build_300.py — Stage 3 step 1: create data/queries_hotpot_300.json.

- Loads HotpotQA distractor/validation from the OFFICIAL hotpotqa/hotpot_qa
  Parquet repo (same method as build_datasets.py).
- Queries 0–149: the existing data/queries_hotpot.json, order preserved.
  Verified item-by-item against the dataset.
- Queries 150–299: the next 150 dataset examples whose question text does not
  collide with the first 150 (or each other).
- Writes the NEW questions' supporting+distractor paragraphs into
  data/corpus_hotpot/ using the identical title-hash filename scheme, skipping
  files that already exist (same dedup-by-title rule as build_datasets.py).
- Verifies 300 unique question texts.

No Ollama usage. Safe to run alongside nothing else.
"""

import hashlib
import json
from pathlib import Path

from datasets import load_dataset

DATA = Path("data")
CORPUS = DATA / "corpus_hotpot"
EXISTING = DATA / "queries_hotpot.json"
OUT = DATA / "queries_hotpot_300.json"

existing = json.loads(EXISTING.read_text(encoding="utf-8"))
assert len(existing) == 150, f"expected 150 existing queries, got {len(existing)}"
existing_qs = [e["question"] for e in existing]

print("Loading HotpotQA (distractor, validation) — official Parquet repo ...")
ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
print(f"dataset rows: {len(ds)}")

# ── Verify the first 150 dataset questions match the existing file, in order ──
mismatches = 0
for i in range(150):
    if ds[i]["question"] != existing_qs[i]:
        mismatches += 1
        print(f"  MISMATCH at index {i}:")
        print(f"    dataset : {ds[i]['question'][:100]}")
        print(f"    existing: {existing_qs[i][:100]}")
print(f"first-150 order check: {150 - mismatches}/150 match"
      + ("" if mismatches == 0 else "  — ABORTING") )
assert mismatches == 0, "existing queries do not align with dataset order"

# ── Select the next 150 unique-by-text questions ──────────────────────────────
seen_qs = set(existing_qs)
new_items = []
new_indices = []
i = 150
while len(new_items) < 150 and i < len(ds):
    ex = ds[i]
    if ex["question"] not in seen_qs:
        seen_qs.add(ex["question"])
        new_items.append(ex)
        new_indices.append(i)
    else:
        print(f"  skipping dataset index {i}: duplicate question text")
    i += 1
assert len(new_items) == 150, f"only found {len(new_items)} new unique questions"
print(f"new queries: dataset indices {new_indices[0]}..{new_indices[-1]} "
      f"({len(new_items)} items, {i - 150 - 150} duplicates skipped)")

# ── Write the new questions' paragraphs into the corpus ───────────────────────
n_new_docs = 0
n_already = 0
for ex in new_items:
    titles = ex["context"]["title"]
    sentence_lists = ex["context"]["sentences"]
    for title, sents in zip(titles, sentence_lists):
        key = title.strip()
        fn = "doc_" + hashlib.md5(key.encode()).hexdigest()[:12] + ".txt"
        path = CORPUS / fn
        if path.exists():
            n_already += 1
            continue
        doc_text = f"{title}\n\n" + " ".join(sents)
        path.write_text(doc_text, encoding="utf-8")
        n_new_docs += 1

total_docs = len(list(CORPUS.glob("*.txt")))
print(f"corpus: wrote {n_new_docs} NEW documents "
      f"({n_already} paragraph slots already present by title)")
print(f"corpus total .txt docs now: {total_docs}")

# ── Build and save the combined 300-query file ────────────────────────────────
combined = list(existing)
for ex in new_items:
    combined.append({
        "question":     ex["question"],
        "ground_truth": ex["answer"],
        "level":        ex.get("level", ""),
        "type":         ex.get("type", ""),
    })

unique = len(set(item["question"] for item in combined))
print(f"combined queries: {len(combined)}  unique question texts: {unique}")
assert len(combined) == 300 and unique == 300, "uniqueness check FAILED"

OUT.write_text(json.dumps(combined, indent=2, ensure_ascii=False),
               encoding="utf-8")
print(f"saved -> {OUT}")
print("\nNEXT: corpus gained new docs -> ChromaDB re-ingest required "
      "(python ingest.py --reset) BEFORE generating queries 150-299.")
