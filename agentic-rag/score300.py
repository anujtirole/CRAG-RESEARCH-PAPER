"""
score300.py — Stage 3 step 3: RAGAS-score ONLY the new 150 cache rows.

- Waits until results/generation_cache.csv holds all 300 rows (aborts if not).
- Judge = llama3.1:8b via the same wrapper as evaluate.py (max_workers=1),
  local HuggingFace embeddings, batches of 15 with retry — reuses evaluate.py's
  _compute_ragas machinery directly.
- CHECKPOINT PER BATCH: every scored batch is appended immediately to
  results/score300_scores.csv as (question, condition, three metrics); on
  restart, (condition, question) pairs already present are skipped.
- When every condition is fully scored, merges the new 150 scored rows into
  results/results_main.csv (existing 150 rows untouched), recomputes the
  ** AGGREGATE MEAN ** row over all 300, and writes the file.
- Failure accounting accumulated per condition -> results/score300_failures.txt

Run DETACHED: output to results/score300.log via shell redirection.
"""

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
from evaluate import (_build_ragas_llm_and_embeddings, _compute_ragas,
                      _METRIC_KEYS)

CACHE = Path("results/generation_cache.csv")
MAIN = Path("results/results_main.csv")
SCORES = Path("results/score300_scores.csv")
FAILS = Path("results/score300_failures.txt")
BATCH = 15
CONDITIONS = ["baseline", "reranker", "agentic"]


def main():
    cache = pd.read_csv(CACHE)
    if len(cache) < 300:
        print(f"[score300] ABORT: cache has {len(cache)} rows, need 300. "
              "Run extend_to_300.py to completion first.", flush=True)
        sys.exit(1)

    main_df = pd.read_csv(MAIN)
    main_df = main_df[main_df["question"] != "** AGGREGATE MEAN **"].reset_index(drop=True)
    scored_qs = set(main_df["question"])
    new_rows = cache[~cache["question"].isin(scored_qs)].reset_index(drop=True)
    print(f"[score300] cache={len(cache)} rows; already scored={len(main_df)}; "
          f"to score={len(new_rows)}", flush=True)
    assert len(new_rows) == 150, f"expected 150 new rows, got {len(new_rows)}"

    # Load batch-level checkpoint
    if SCORES.exists():
        sc = pd.read_csv(SCORES)
        done = set(zip(sc["condition"], sc["question"]))
        print(f"[score300] resuming: {len(sc)} (condition,question) scores on disk",
              flush=True)
    else:
        SCORES.write_text("question,condition," + ",".join(_METRIC_KEYS) + "\n",
                          encoding="utf-8")
        done = set()

    llm_w, emb_w = _build_ragas_llm_and_embeddings()
    if llm_w is None:
        print("[score300] ABORT: RAGAS LLM unavailable.", flush=True)
        sys.exit(1)

    fail_acc = {c: {m: {"scored": 0, "failed": 0} for m in _METRIC_KEYS}
                for c in CONDITIONS}

    t0 = time.time()
    for cond in CONDITIONS:
        todo = new_rows[~new_rows["question"].isin(
            {q for c, q in done if c == cond})].reset_index(drop=True)
        n_batches = (len(todo) + BATCH - 1) // BATCH
        print(f"\n[score300] condition={cond}: {len(todo)} rows to score "
              f"({n_batches} batches of {BATCH})", flush=True)

        for bi in range(n_batches):
            sl = todo.iloc[bi * BATCH:(bi + 1) * BATCH]
            qs = sl["question"].tolist()
            ans = sl[f"{cond}_answer"].fillna("").tolist()
            ctxs = [json.loads(x) for x in sl[f"{cond}_contexts"]]
            gts = sl["ground_truth"].fillna("").tolist()

            print(f"[score300] {cond} batch {bi + 1}/{n_batches} "
                  f"(elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)
            per_df, _, _ = _compute_ragas(qs, ans, ctxs, gts, llm_w, emb_w,
                                          batch_size=BATCH)

            # Append checkpoint rows for this batch
            out = pd.DataFrame({"question": qs, "condition": cond})
            for m in _METRIC_KEYS:
                out[m] = per_df[m].values
                fail_acc[cond][m]["scored"] += int(per_df[m].notna().sum())
                fail_acc[cond][m]["failed"] += int(per_df[m].isna().sum())
            out.to_csv(SCORES, mode="a", header=False, index=False)

        # condition-level failure report (rewritten as we go)
        lines = [f"score300 failure accounting  (generated {time.strftime('%Y-%m-%d %H:%M:%S')})",
                 "covers ONLY the 150 new rows (queries 150-299)", "=" * 60]
        for c in CONDITIONS:
            lines.append(f"\n[{c}]")
            for m in _METRIC_KEYS:
                s = fail_acc[c][m]
                lines.append(f"  {m:<22} scored={s['scored']:>4}  failed={s['failed']:>4}")
        FAILS.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── Merge into results_main.csv ───────────────────────────────────────────
    print("\n[score300] all conditions scored — merging into results_main.csv",
          flush=True)
    sc = pd.read_csv(SCORES)
    new_scored = new_rows.copy()
    for cond in CONDITIONS:
        sub = sc[sc["condition"] == cond].drop_duplicates("question", keep="last")
        m = new_scored["question"].map(sub.set_index("question")[_METRIC_KEYS[0]])
        for met in _METRIC_KEYS:
            new_scored[f"{cond}_{met}"] = new_scored["question"].map(
                sub.set_index("question")[met])

    all_cols = list(main_df.columns)
    for c in new_scored.columns:
        if c not in all_cols:
            all_cols.append(c)
    combined = pd.concat([main_df.reindex(columns=all_cols),
                          new_scored.reindex(columns=all_cols)],
                         ignore_index=True)
    assert len(combined) == 300, f"combined has {len(combined)} rows"

    agg = {"question": "** AGGREGATE MEAN **", "ground_truth": ""}
    for cond in CONDITIONS:
        for col in [f"{cond}_latency", f"{cond}_faithfulness",
                    f"{cond}_answer_relevancy", f"{cond}_context_precision"]:
            if col in combined.columns:
                agg[col] = combined[col].mean(skipna=True)
    for col in ["agentic_n_attempts", "agentic_confidence"]:
        agg[col] = combined[col].mean(skipna=True)

    final = pd.concat([combined, pd.DataFrame([agg])], ignore_index=True)
    final.to_csv(MAIN, index=False)
    print(f"[score300] results_main.csv now holds {len(combined)} scored rows "
          f"+ aggregate. Total time {(time.time() - t0) / 60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()
