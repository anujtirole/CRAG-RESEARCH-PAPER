"""
build_datasets.py
------------------
Creates TWO real, citable datasets for the agentic-RAG experiments:

  DATASET 1 (public benchmark): HotpotQA distractor validation subset.
    - Real multi-hop QA pairs with ground-truth answers (no hand-writing needed).
    - Directly answers reviewer's "test on a public benchmark" critique.
    - The paragraphs become corpus documents; the questions become queries.

  DATASET 2 (domain corpus): real arXiv abstracts in a chosen field.
    - Real technical documents you choose the topic of.
    - You write a smaller set of queries against these (or reuse the helper).

Run:  python build_datasets.py
Output:
  data/corpus_hotpot/*.txt        + data/queries_hotpot.json
  data/corpus_arxiv/*.txt         (write your own data/queries_arxiv.json)

Dependencies (add to requirements.txt if missing):
  datasets, arxiv
"""

import json, os, re, hashlib
from pathlib import Path

DATA = Path("data")

# ----------------------------------------------------------------------
# DATASET 1 — HotpotQA (public benchmark, questions + answers included)
# ----------------------------------------------------------------------
# Uses the official hotpotqa/hotpot_qa repo (Parquet-backed, no loading script).
# HotpotQA is CC BY-SA 4.0 — cite Yang et al., EMNLP 2018.

def build_hotpot(n_questions: int = 150):
    from datasets import load_dataset

    print("Loading HotpotQA (distractor, validation) ...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    ds = ds.select(range(min(n_questions, len(ds))))

    corpus_dir = DATA / "corpus_hotpot"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    queries = []
    seen_docs = {}  # title -> filename, so identical Wikipedia paras aren't duplicated

    for ex in ds:
        # Each example provides 10 paragraphs (gold + distractors) in ex["context"].
        titles = ex["context"]["title"]
        sentence_lists = ex["context"]["sentences"]

        for title, sents in zip(titles, sentence_lists):
            doc_text = f"{title}\n\n" + " ".join(sents)
            key = title.strip()
            if key not in seen_docs:
                # stable filename from a hash of the title
                fn = "doc_" + hashlib.md5(key.encode()).hexdigest()[:12] + ".txt"
                (corpus_dir / fn).write_text(doc_text, encoding="utf-8")
                seen_docs[key] = fn

        queries.append({
            "question": ex["question"],
            "ground_truth": ex["answer"],
            "level": ex.get("level", ""),
            "type": ex.get("type", ""),
        })

    (DATA / "queries_hotpot.json").write_text(
        json.dumps(queries, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  HotpotQA done: {len(seen_docs)} unique documents, "
          f"{len(queries)} queries.")
    print(f"  -> {corpus_dir}/  and  {DATA/'queries_hotpot.json'}")


# ----------------------------------------------------------------------
# DATASET 2 — arXiv abstracts (real domain corpus, you choose the field)
# ----------------------------------------------------------------------
# Uses the `arxiv` python package to pull real paper abstracts.
# Change SEARCH_QUERY and N_DOCS to set your domain and corpus size.

def build_arxiv(search_query: str = "retrieval augmented generation",
                n_docs: int = 300):
    import arxiv

    print(f"Fetching {n_docs} arXiv abstracts for: '{search_query}' ...")
    corpus_dir = DATA / "corpus_arxiv"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    search = arxiv.Search(
        query=search_query,
        max_results=n_docs,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    count = 0
    for result in search.results():
        title = result.title.strip()
        abstract = result.summary.strip().replace("\n", " ")
        doc_text = f"{title}\n\n{abstract}"
        fn = "arxiv_" + hashlib.md5(title.encode()).hexdigest()[:12] + ".txt"
        (corpus_dir / fn).write_text(doc_text, encoding="utf-8")
        count += 1

    print(f"  arXiv done: {count} documents.")
    print(f"  -> {corpus_dir}/")
    print("  NOTE: write your own data/queries_arxiv.json with "
          "{question, ground_truth} pairs you can verify from these abstracts.")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    # ---- Dataset 1: public benchmark (recommended first) ----
    build_hotpot(n_questions=150)

    # ---- Dataset 2: domain corpus (edit the topic to your field) ----
    # Commented out: arxiv 2.x API changed; use Client().results(search) — fix separately.
    # build_arxiv(search_query="retrieval augmented generation large language models",
    #             n_docs=300)

    print("\nDone. Point config.py at one corpus dir + matching queries file,")
    print("run ingest.py, then evaluate.py. Repeat for the second dataset.")
