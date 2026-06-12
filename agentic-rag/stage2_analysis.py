"""
stage2_analysis.py — Stage 2 of the overnight pipeline (analysis only, no generation).

2A Holm-corrected significance (reads results/significance_holm.csv, produced
   by multiple_comparisons.py)
2B Merge statistics + verbatim dedup criterion from agentic_rag.py
2C RAGAS judge failure types per condition, from logs + results_main.csv NaNs
2D Three full worked examples selected from the cache
2E Reranker parameter count, predict() call sites, 20-rerank CPU timing

Output: printed AND saved to results/stage2_analysis.txt
"""

import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RES = ROOT / "results"
TAU = 0.40  # tau used for the published main runs

out_buf = io.StringIO()


def emit(*args):
    line = " ".join(str(a) for a in args)
    out_buf.write(line + "\n")
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode())


def section(title):
    emit("\n" + "=" * 78)
    emit(title)
    emit("=" * 78)


# ───────────────────────────── 2A ─────────────────────────────
section("2A — Holm-Bonferroni corrected significance (n=150)")
holm = pd.read_csv(RES / "significance_holm.csv")
emit(holm.to_string(index=False))

# ───────────────────────────── 2B ─────────────────────────────
section("2B — Merge statistics (agentic re-retrieval)")
cache = pd.read_csv(RES / "generation_cache.csv")
n_total = len(cache)
multi = cache[cache["agentic_n_attempts"] > 1]
emit(f"Agentic queries total:            {n_total}")
emit(f"Queries with n_attempts > 1:      {len(multi)} "
     f"({100*len(multi)/n_total:.1f}%)")
emit(f"n_attempts distribution:\n{cache['agentic_n_attempts'].value_counts().sort_index().to_string()}")

emit("\nVerbatim dedup/merge criterion from agentic_rag.py:")
src = (ROOT / "agentic_rag.py").read_text(encoding="utf-8").splitlines()
start = next(i for i, l in enumerate(src) if "MERGE-AND-DEDUPLICATE" in l)
end = next(i for i, l in enumerate(src) if "merged.append(c)" in l)
for i in range(start, end + 1):
    emit(f"  {i+1:4d} | {src[i]}")
emit("  -> Matching criterion: exact string equality of the full chunk text")
emit("     (c[\"text\"] not in seen). No fuzzy/embedding-based dedup.")
emit("\nAttempt-level chunk provenance: NOT STORED. The cache persists only the")
emit("final post-merge contexts (agentic_contexts); per-attempt retrieved sets")
emit("are not logged anywhere, so we cannot report how many final chunks came")
emit("from attempt 1 vs re-retrieval. This is a known logging gap.")

# ───────────────────────────── 2C ─────────────────────────────
section("2C — RAGAS judge failure types per condition")

def count_failures(text, label):
    n_parser = len(re.findall(r"OutputParserException", text))
    n_ragas_parser = len(re.findall(r"RagasOutputParserException", text))
    n_timeout = len(re.findall(r"[Tt]imeout|TimeoutError", text))
    emit(f"  {label:35s} parser_exceptions={n_parser:3d} "
         f"(of which RagasOutputParserException={n_ragas_parser}), "
         f"timeouts={n_timeout}")
    return n_parser, n_timeout

eval_log = (RES / "evaluate_phase3.log").read_text(encoding="utf-8", errors="replace")
marker = "Computing RAGAS metrics for condition: "
parts = eval_log.split(marker)
emit("evaluate_phase3.log (original n=150 scoring run, judge=llama3.1:8b):")
for part in parts[1:]:
    cond = part.split("…")[0].split("\n")[0].strip().rstrip(". …")
    count_failures(part, f"condition: {cond}")

rescore_log = (RES / "rescore_agentic.log").read_text(encoding="utf-8", errors="replace")
emit("rescore_agentic.log (agentic-only re-score after fallback change):")
count_failures(rescore_log, "condition: agentic (re-score)")

emit("\nNaN cells in results_main.csv (the authoritative per-query record —")
emit("a NaN here = that sample ultimately unscored after retries):")
main = pd.read_csv(RES / "results_main.csv")
# drop the "** AGGREGATE MEAN **" summary row appended by evaluate.py
main = main[main["question"] != "** AGGREGATE MEAN **"].reset_index(drop=True)
for cond in ["baseline", "reranker", "agentic"]:
    for met in ["faithfulness", "answer_relevancy", "context_precision"]:
        col = f"{cond}_{met}"
        if col in main.columns:
            emit(f"  {col:35s} NaN = {int(main[col].isna().sum()):3d} / {len(main)}")

emit("\nCurrent ragas_failure_report.txt (NOTE: covers AGENTIC ONLY — the")
emit("re-score run rewrote the file; per-condition snapshots were not kept):")
for line in (RES / "ragas_failure_report.txt").read_text(encoding="utf-8").splitlines():
    emit("  " + line)

emit("\nHONEST GAPS: (1) the logs do not tag each exception with the metric it")
emit("belonged to, so per-metric failure attribution comes only from NaN counts")
emit("above; (2) baseline/reranker failure counts for the original run survive")
emit("only inside evaluate_phase3.log (counted above), since the failure-report")
emit("file was overwritten; (3) zero judge timeouts occurred in either log —")
emit("all failures were output-parsing failures (invalid JSON from llama3.1:8b).")

# ───────────────────────────── 2D ─────────────────────────────
section("2D — Worked examples (full text, selected by inspection)")

# Parse per-attempt confidences from regen_agentic.log (source of current cache)
regen_log = (RES / "regen_agentic.log").read_text(encoding="utf-8", errors="replace")
attempt_confs = {}  # qidx (0-based) -> list of confidences
cur = None
for line in regen_log.splitlines():
    m = re.match(r"\[regen\] (\d+)/150", line)
    if m:
        cur = int(m.group(1)) - 1
        attempt_confs[cur] = []
    m = re.search(r"attempt \d/\d\s+confidence=([\d.]+)", line)
    if m and cur is not None:
        attempt_confs[cur].append(float(m.group(1)))


def show_row(idx, label):
    r = cache.iloc[idx]
    confs = attempt_confs.get(idx, [])
    emit(f"\n--- {label} (cache row {idx}) ---")
    emit(f"question:      {r['question']}")
    emit(f"ground_truth:  {r['ground_truth']}")
    emit(f"answer:        {r['agentic_answer']}")
    emit(f"n_attempts:    {r['agentic_n_attempts']}")
    emit(f"confidence:    final={r['agentic_confidence']}  per-attempt={confs}")
    emit(f"low_conf:      {r['agentic_low_conf']}")
    emit(f"used_fallback: {r['agentic_used_fallback']}")


def gt_in_answer(row):
    gt = str(row["ground_truth"]).strip().lower()
    ans = str(row["agentic_answer"]).lower()
    return gt in ans

# (a) re-retrieval success: >1 attempt, confidence rose, gate passed, answer correct
cand_a = []
for i in range(n_total):
    r = cache.iloc[i]
    confs = attempt_confs.get(i, [])
    if (r["agentic_n_attempts"] > 1 and not r["agentic_low_conf"]
            and len(confs) >= 2 and confs[-1] > confs[0]
            and len(str(r["ground_truth"])) > 3 and gt_in_answer(r)):
        cand_a.append(i)
emit(f"(a) re-retrieval successes found: {len(cand_a)} -> showing first")
if cand_a:
    show_row(cand_a[0], "(a) RE-RETRIEVAL SUCCESS: confidence rose, gate passed, answer correct")
else:
    emit("    NONE FOUND matching all criteria — reporting honestly.")

# (b) critic-misled: gate passed (low_conf=False) but answer wrong
cand_b = []
for i in range(n_total):
    r = cache.iloc[i]
    if (not r["agentic_low_conf"] and len(str(r["ground_truth"])) > 3
            and not gt_in_answer(r)
            and "does not contain enough information" not in str(r["agentic_answer"]).lower()):
        cand_b.append(i)
emit(f"\n(b) critic-misled candidates (gate passed, ground truth absent from "
     f"answer): {len(cand_b)} -> showing first")
if cand_b:
    show_row(cand_b[0], "(b) CRITIC-MISLED: gate passed, answer wrong")

# (c) fallback hedge
cand_c = [i for i in range(n_total)
          if cache.iloc[i]["agentic_used_fallback"]
          and "based on partial evidence" in str(cache.iloc[i]["agentic_answer"]).lower()]
emit(f"\n(c) used_fallback=True rows with hedge prefix: {len(cand_c)} -> showing first")
if cand_c:
    show_row(cand_c[0], '(c) FALLBACK HEDGE: "Based on partial evidence:"')

# ───────────────────────────── 2E ─────────────────────────────
section("2E — Reranker detail (BAAI/bge-reranker-base)")
from sentence_transformers import CrossEncoder  # noqa: E402
import config  # noqa: E402

ce = CrossEncoder(config.RERANKER_MODEL, device="cpu")
n_params = sum(p.numel() for p in ce.model.parameters())
emit(f"Parameter count: {n_params:,} ({n_params/1e6:.1f} M)")

emit("\npredict() call site — agentic_rag.py:")
for i, l in enumerate(src):
    if "ce_pairs" in l or "ce_scores" in l:
        emit(f"  {i+1:4d} | {l}")
rr_src = (ROOT / "reranker_rag.py").read_text(encoding="utf-8").splitlines()
emit("predict() call site — reranker_rag.py:")
for i, l in enumerate(rr_src):
    if ".predict(" in l or "pairs: List[tuple]" in l:
        emit(f"  {i+1:4d} | {l}")
emit("\nBatching: CrossEncoder.predict() batches internally with its default")
emit("batch_size=32. Both call sites pass the full pair list in one call");
emit(f"(TOP_K={config.TOP_K} pairs for reranker condition; up to ~{config.TOP_K}+{config.RERANK_TOP_N}")
emit("merged pairs in the agentic loop), i.e. a single internal batch per rerank.")

# Timing: 20 sample reranks using real cached (question, 10-candidate) sets
qs = cache["question"].tolist()[:20]
ctxs = [json.loads(cache.iloc[i]["baseline_contexts"]) for i in range(20)]
emit(f"\nTiming 20 sample reranks (pairs per rerank: "
     f"{sorted(set(len(c) for c in ctxs))} chunks, CPU)...")
ce.predict([(qs[0], t) for t in ctxs[0]])  # warm-up, excluded
times = []
for q, chunks in zip(qs, ctxs):
    t0 = time.perf_counter()
    ce.predict([(q, t) for t in chunks])
    times.append(time.perf_counter() - t0)
emit(f"mean wall-clock per rerank: {np.mean(times):.3f}s  "
     f"(std {np.std(times):.3f}s, min {min(times):.3f}s, max {max(times):.3f}s)")

# ───────────────────────────── save ─────────────────────────────
(RES / "stage2_analysis.txt").write_text(out_buf.getvalue(), encoding="utf-8")
print("\n[stage2] saved to results/stage2_analysis.txt")
