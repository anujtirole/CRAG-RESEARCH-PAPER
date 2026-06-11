"""
Evaluation harness — runs all three RAG conditions and computes RAGAS metrics.

Usage:
  python evaluate.py                          # full evaluation (all 3 conditions)
  python evaluate.py --score-only             # re-score ALL conditions from generation_cache.csv
  python evaluate.py --score-from-cache       # alias for --score-only (same behaviour)
  python evaluate.py --threshold-sweep        # Condition 3 only at τ = 0.5/0.6/0.7/0.8
  python evaluate.py --threshold-sweep \
         --sweep-from-cache                   # sweep RAGAS from cache; only re-generates
                                              # agentic at τ values not already in cache

Output files in results/:
  results_main.csv         — per-query results + aggregates for all conditions
  threshold_sweep.csv      — aggregate metrics per τ value

RAGAS metrics computed:
  Faithfulness        — is the answer grounded in the retrieved contexts?
  Answer Relevancy    — is the answer relevant to the question?
  Context Precision   — are relevant chunks ranked above irrelevant ones?
                        (requires ground_truth in queries.json)

IMPORTANT — context_precision zero-detection:
  A reranker_context_precision of 0.0 across all rows signals a silent RAGAS failure
  (RAGAS returned 0 instead of NaN).  --score-only and --score-from-cache both detect
  this and force a re-score of any column whose mean == 0.0 AND whose non-nan count == n.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Windows cmd/PowerShell default to cp1252; force UTF-8 so arrow/Greek chars print cleanly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
from tqdm import tqdm

import config
from retriever import Retriever
from baseline_rag import run_baseline
from reranker_rag import RerankerRAG
from agentic_rag import run_agentic

# ── RAGAS import (graceful degradation if not installed / wrong version) ───────

RAGAS_OK = False
_HAS_RUN_CONFIG = False
_RunConfig = None
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import Faithfulness, AnswerRelevancy

    # Context Precision import varies by RAGAS sub-version; try both names
    try:
        from ragas.metrics import ContextPrecision
    except ImportError:
        try:
            from ragas.metrics import LLMContextPrecisionWithReference as ContextPrecision
        except ImportError:
            ContextPrecision = None

    try:
        from ragas import EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        RAGAS_V2 = True
    except ImportError:
        from datasets import Dataset as HFDataset
        RAGAS_V2 = False

    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    try:
        from ragas.run_config import RunConfig as _RunConfig
        _HAS_RUN_CONFIG = True
    except ImportError:
        pass

    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama  # type: ignore

    try:
        from langchain_core.exceptions import OutputParserException as _OutputParserException
    except ImportError:
        _OutputParserException = Exception

    # Always load embeddings locally via sentence-transformers; never route through Ollama.
    try:
        # Preferred package
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    except Exception:
        try:
            # Another possible location in newer langchain versions
            from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
        except Exception:
            try:
                # Fallback for langchain_huggingface
                import importlib
                langchain_huggingface = importlib.import_module("langchain_huggingface")
                HuggingFaceEmbeddings = langchain_huggingface.HuggingFaceEmbeddings
            except Exception:
                # Final fallback: raise ImportError to be caught by outer try
                raise

    OutputParserException = _OutputParserException
    RAGAS_OK = True
    print("[evaluate] RAGAS loaded successfully.")
except ImportError as _e:
    print(f"[evaluate] WARNING: RAGAS not available ({_e}).")
    print("  Raw results will still be saved; RAGAS columns will be NaN.")
    print("  Install with: pip install ragas langchain-ollama langchain-community")
    OutputParserException = Exception  # keep name defined even when RAGAS is absent


# ── RAGAS setup ───────────────────────────────────────────────────────────────

_RAGAS_TIMEOUT = 300  # seconds per Ollama call during RAGAS scoring


def _build_ragas_llm_and_embeddings():
    if not RAGAS_OK:
        return None, None
    try:
        _chat_base = dict(
            model=config.LLM_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.0,
        )
        try:
            _chat = ChatOllama(**_chat_base, timeout=_RAGAS_TIMEOUT)
        except TypeError:
            _chat = ChatOllama(**_chat_base, request_timeout=_RAGAS_TIMEOUT)
        llm_wrapper = LangchainLLMWrapper(_chat)
        emb = HuggingFaceEmbeddings(
            model_name=config.EMBED_MODEL,
            model_kwargs={"device": "cpu"},
        )
        emb_wrapper = LangchainEmbeddingsWrapper(emb)
        return llm_wrapper, emb_wrapper
    except Exception as exc:
        print(f"[evaluate] WARNING: could not initialise RAGAS LLM/embeddings: {exc}")
        return None, None


def _make_run_config():
    if _HAS_RUN_CONFIG and _RunConfig is not None:
        return _RunConfig(max_workers=1)
    return None


def _ragas_batch_once(
    questions: List[str],
    answers: List[str],
    contexts_list: List[List[str]],
    ground_truths: List[str],
    metrics_to_run: list,
    llm_wrapper,
    emb_wrapper,
    run_config=None,
) -> pd.DataFrame:
    """Run ragas_evaluate on one batch; return per-sample scores as DataFrame."""
    rc_kwargs = {"run_config": run_config} if run_config is not None else {}
    if RAGAS_V2:
        samples = [
            SingleTurnSample(
                user_input=q,
                response=a,
                retrieved_contexts=ctxs,
                reference=gt if gt.strip() else None,
            )
            for q, a, ctxs, gt in zip(questions, answers, contexts_list, ground_truths)
        ]
        dataset = EvaluationDataset(samples=samples)
        result = ragas_evaluate(
            dataset=dataset,
            metrics=metrics_to_run,
            llm=llm_wrapper,
            embeddings=emb_wrapper,
            raise_exceptions=False,
            **rc_kwargs,
        )
    else:
        data = {
            "question":     questions,
            "answer":       answers,
            "contexts":     contexts_list,
            "ground_truth": ground_truths,
        }
        dataset = HFDataset.from_dict(data)
        result = ragas_evaluate(
            dataset,
            metrics=metrics_to_run,
            llm=llm_wrapper,
            embeddings=emb_wrapper,
            **rc_kwargs,
        )
    return result.to_pandas()


_METRIC_KEYS = ["faithfulness", "answer_relevancy", "context_precision"]

# RAGAS column names vary by version; map each canonical metric to its aliases.
_METRIC_ALIASES = {
    "faithfulness":      ["faithfulness"],
    "answer_relevancy":  ["answer_relevancy", "response_relevancy"],
    "context_precision": ["context_precision", "llm_context_precision_with_reference"],
}


def _normalize_per_sample(df_raw: Optional[pd.DataFrame], n_rows: int) -> pd.DataFrame:
    """
    Map a raw RAGAS per-sample result (or None for a failed batch) onto a
    DataFrame with exactly n_rows rows and the three canonical metric columns,
    NaN where a score is unavailable.
    """
    out = pd.DataFrame({m: [float("nan")] * n_rows for m in _METRIC_KEYS})
    if df_raw is None:
        return out
    for metric, aliases in _METRIC_ALIASES.items():
        for alias in aliases:
            if alias in df_raw.columns:
                vals = pd.to_numeric(df_raw[alias], errors="coerce").tolist()
                if len(vals) == n_rows:
                    out[metric] = vals
                break
    return out


def _compute_ragas(
    questions: List[str],
    answers: List[str],
    contexts_list: List[List[str]],
    ground_truths: List[str],
    llm_wrapper,
    emb_wrapper,
    batch_size: int = 15,
    max_retries: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    """
    Compute RAGAS metrics for one condition.

    Splits into batches of `batch_size` and retries each batch up to `max_retries`
    times on OutputParserException before skipping it.  Failed batches contribute
    NaN rows so per-sample alignment with the input order is always preserved.

    Returns:
        per_sample : DataFrame, len(questions) rows × the three canonical metric
                     columns; NaN where scoring failed or the metric was not run.
        means      : dict of per-metric means over the scored rows (may be NaN).
        stats      : failure accounting — n_rows, batches_total, batches_failed,
                     cp_attempted, and per-metric scored/failed counts.
    """
    nan = float("nan")
    n = len(questions)
    per_sample_empty = _normalize_per_sample(None, n)
    empty_means = {m: nan for m in _METRIC_KEYS}

    metrics_to_run = [Faithfulness(), AnswerRelevancy()] if RAGAS_OK else []
    has_ground_truth = any(gt.strip() for gt in ground_truths)
    cp_attempted = RAGAS_OK and ContextPrecision is not None and has_ground_truth
    if cp_attempted:
        metrics_to_run.append(ContextPrecision())

    base_stats: Dict[str, Any] = {
        "n_rows": n,
        "batches_total": (n + batch_size - 1) // batch_size,
        "batches_failed": 0,
        "cp_attempted": cp_attempted,
    }

    if not RAGAS_OK or llm_wrapper is None:
        base_stats["batches_failed"] = base_stats["batches_total"]
        for m in _METRIC_KEYS:
            base_stats[f"{m}_scored"] = 0
            base_stats[f"{m}_failed"] = n
        return per_sample_empty, empty_means, base_stats

    run_cfg = _make_run_config()
    batch_dfs: List[pd.DataFrame] = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_num = start // batch_size + 1
        b_total = base_stats["batches_total"]
        print(f"  [ragas] batch {b_num}/{b_total} (rows {start}–{end - 1})", flush=True)

        bq  = questions[start:end]
        ba  = answers[start:end]
        bc  = contexts_list[start:end]
        bgt = ground_truths[start:end]

        df_batch: Optional[pd.DataFrame] = None
        for attempt in range(max_retries + 1):
            try:
                df_batch = _ragas_batch_once(bq, ba, bc, bgt, metrics_to_run, llm_wrapper, emb_wrapper, run_config=run_cfg)
                break
            except OutputParserException as exc:
                if attempt < max_retries:
                    print(f"  [ragas] OutputParserException on batch {b_num}, retry {attempt + 1}/{max_retries}: {exc}")
                    time.sleep(3)
                else:
                    print(f"  [ragas] OutputParserException on batch {b_num} after {max_retries} retries — skipping batch.")
            except Exception as exc:
                print(f"  [ragas] ERROR on batch {b_num} (attempt {attempt + 1}): {exc}")
                traceback.print_exc()
                if attempt < max_retries:
                    time.sleep(3)
                else:
                    print(f"  [ragas] Giving up on batch {b_num}.")

        if df_batch is None:
            base_stats["batches_failed"] += 1
        # Failed batches become NaN rows so row alignment is preserved.
        batch_dfs.append(_normalize_per_sample(df_batch, end - start))

    per_sample = pd.concat(batch_dfs, ignore_index=True)

    means: Dict[str, float] = {}
    for m in _METRIC_KEYS:
        scored = int(per_sample[m].notna().sum())
        base_stats[f"{m}_scored"] = scored
        base_stats[f"{m}_failed"] = n - scored
        means[m] = float(per_sample[m].mean(skipna=True)) if scored else nan

    return per_sample, means, base_stats


# ── RAGAS failure accounting ──────────────────────────────────────────────────

_FAILURE_STATS: Dict[str, Dict[str, Any]] = {}


def _record_failure_stats(label: str, stats: Dict[str, Any]) -> None:
    """
    Store failure stats for `label` (a condition or sweep point) and rewrite
    results/ragas_failure_report.txt with everything recorded so far, so a
    partial run still leaves a complete report on disk.
    """
    _FAILURE_STATS[label] = stats
    path = config.RESULTS_DIR / "ragas_failure_report.txt"
    lines = [
        "RAGAS failure report",
        f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "Counts of per-query RAGAS evaluations that succeeded vs. failed",
        "(failure = batch error after retries, per-sample timeout/parse error,",
        " or RAGAS returning NaN for that sample).",
        "=" * 64,
    ]
    for cond, s in _FAILURE_STATS.items():
        lines.append(f"\n[{cond}]  rows={s['n_rows']}  "
                     f"batches_failed={s['batches_failed']}/{s['batches_total']}")
        for m in _METRIC_KEYS:
            if m == "context_precision" and not s.get("cp_attempted", True):
                lines.append(f"  {m:<22} not computed (no ground truth or metric unavailable)")
                continue
            lines.append(f"  {m:<22} scored={s[f'{m}_scored']:>4}   failed={s[f'{m}_failed']:>4}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [ragas] failure report updated → {path}")


# ── Suspicious-zero detection ─────────────────────────────────────────────────

def _detect_zero_failure_columns(df: pd.DataFrame, conditions: List[str]) -> List[str]:
    """
    Return list of '<condition>_context_precision' column names whose values are
    ALL exactly 0.0 (and non-null) across every row — a known RAGAS silent-failure
    pattern where the metric returns 0 instead of NaN when parsing fails.

    faithfulness and answer_relevancy are NOT included: a genuine 0 there is
    plausible.  context_precision returning 0.0 for every single sample is not.
    """
    suspect: List[str] = []
    n = len(df)
    for cond in conditions:
        col = f"{cond}_context_precision"
        if col not in df.columns:
            continue
        non_null = df[col].dropna()
        if len(non_null) == n and (non_null == 0.0).all():
            suspect.append(col)
    return suspect


# ── Per-query runner ──────────────────────────────────────────────────────────

def _run_all_conditions(
    queries: List[Dict],
    retriever: Retriever,
    reranker: RerankerRAG,
) -> pd.DataFrame:
    rows = []
    for item in tqdm(queries, desc="Queries", unit="q"):
        q  = item["question"]
        gt = item.get("ground_truth", "")

        try:
            b = run_baseline(q, retriever)
        except Exception as exc:
            print(f"  [baseline] ERROR on '{q[:50]}': {exc}")
            b = {"answer": "", "contexts": [], "latency_s": float("nan")}

        try:
            r = reranker.run(q, retriever)
        except Exception as exc:
            print(f"  [reranker] ERROR on '{q[:50]}': {exc}")
            r = {"answer": "", "contexts": [], "latency_s": float("nan")}

        try:
            a = run_agentic(q, retriever)
        except Exception as exc:
            print(f"  [agentic]  ERROR on '{q[:50]}': {exc}")
            a = {
                "answer": "", "contexts": [], "latency_s": float("nan"),
                "n_attempts": 0, "final_confidence": float("nan"),
                "low_confidence": True,
            }

        rows.append({
            "question":       q,
            "ground_truth":   gt,
            "baseline_answer":   b["answer"],
            "baseline_contexts": json.dumps(b["contexts"]),
            "baseline_latency":  b["latency_s"],
            "reranker_answer":   r["answer"],
            "reranker_contexts": json.dumps(r["contexts"]),
            "reranker_latency":  r["latency_s"],
            "agentic_answer":       a["answer"],
            "agentic_contexts":     json.dumps(a["contexts"]),
            "agentic_latency":      a["latency_s"],
            "agentic_n_attempts":   a["n_attempts"],
            "agentic_confidence":   a["final_confidence"],
            "agentic_low_conf":     a["low_confidence"],
        })

    return pd.DataFrame(rows)


def _run_threshold_sweep(
    queries: List[Dict],
    retriever: Retriever,
    tau_values: List[float] = (0.5, 0.6, 0.7, 0.8),
) -> Tuple[pd.DataFrame, Dict[float, Dict]]:
    rows: List[Dict] = []
    per_tau_data: Dict[float, Dict] = {}

    for tau in tau_values:
        print(f"\n{'─' * 60}")
        print(f"Threshold sweep: τ = {tau}")
        print(f"{'─' * 60}")
        latencies:     List[float] = []
        n_attempts_l:  List[int]   = []
        confidences:   List[float] = []
        answers:       List[str]   = []
        contexts_list: List[List[str]] = []
        questions:     List[str]   = []
        gt_list:       List[str]   = []
        low_conf_count = 0

        for item in tqdm(queries, desc=f"τ={tau}", unit="q"):
            q  = item["question"]
            gt = item.get("ground_truth", "")
            try:
                a = run_agentic(q, retriever, tau=tau)
            except Exception as exc:
                print(f"  ERROR at τ={tau} on '{q[:50]}': {exc}")
                a = {
                    "answer": "", "contexts": [], "latency_s": float("nan"),
                    "n_attempts": config.N_MAX, "final_confidence": float("nan"),
                    "low_confidence": True,
                }
            latencies.append(a["latency_s"])
            n_attempts_l.append(a["n_attempts"])
            confidences.append(a["final_confidence"])
            answers.append(a["answer"])
            contexts_list.append(a["contexts"])
            questions.append(q)
            gt_list.append(gt)
            if a["low_confidence"]:
                low_conf_count += 1

        rows.append({
            "tau":             tau,
            "mean_latency_s":  _safe_mean(latencies),
            "mean_n_attempts": _safe_mean(n_attempts_l),
            "mean_confidence": _safe_mean(confidences),
            "low_conf_count":  low_conf_count,
        })
        per_tau_data[tau] = {
            "questions":     questions,
            "answers":       answers,
            "contexts_list": contexts_list,
            "ground_truths": gt_list,
        }

    return pd.DataFrame(rows), per_tau_data


# ── Utilities ─────────────────────────────────────────────────────────────────

def _safe_mean(values: list) -> float:
    valid = [v for v in values if v == v]   # filter NaN
    return sum(valid) / len(valid) if valid else float("nan")


def _print_summary(df_main: pd.DataFrame, ragas_results: Dict[str, Dict]) -> None:
    conditions = ["baseline", "reranker", "agentic"]
    col_w = 18

    header = f"{'Metric':<28}" + "".join(f"{c.capitalize():>{col_w}}" for c in conditions)
    print("\n" + "=" * len(header))
    print("RESULTS SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    def _row(label: str, values: List) -> str:
        return f"{label:<28}" + "".join(f"{str(v):>{col_w}}" for v in values)

    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        vals = []
        for c in conditions:
            v = ragas_results.get(c, {}).get(metric, float("nan"))
            vals.append(f"{v:.4f}" if v == v else "   N/A")
        print(_row(metric, vals))

    lats = [f"{df_main[f'{c}_latency'].mean():.2f}s" for c in conditions]
    print(_row("mean_latency", lats))

    print("-" * len(header))
    print(f"Agentic — mean attempts  : {df_main['agentic_n_attempts'].mean():.2f} / {config.N_MAX}")
    print(f"Agentic — mean confidence: {df_main['agentic_confidence'].mean():.4f}  (τ={config.TAU})")
    lc = df_main["agentic_low_conf"].sum()
    print(f"Agentic — low-conf flags : {lc} / {len(df_main)} queries")
    print("=" * len(header))


def _print_sweep_summary(df: pd.DataFrame, n_queries: int) -> None:
    metrics = ["faithfulness", "answer_relevancy", "context_precision",
               "mean_latency_s", "mean_n_attempts", "low_conf_count"]
    labels  = ["faithfulness", "answer_relevancy", "context_precision",
               "mean_latency_s", "mean_attempts", f"low_conf / {n_queries}"]
    col_w = 14

    taus = df["tau"].tolist()
    header = f"{'metric':<24}" + "".join(f"tau={t:<{col_w - 4}}" for t in taus)
    sep    = "─" * len(header)
    print(f"\n{sep}")
    print("THRESHOLD SWEEP SUMMARY  (agentic condition only)")
    print(sep)
    print(header)
    print(sep)
    for col, lbl in zip(metrics, labels):
        if col not in df.columns:
            continue
        vals = df[col].tolist()
        def _fmt(v: Any) -> str:
            if isinstance(v, float) and v != v:
                return "N/A"
            if col == "mean_latency_s":
                return f"{v:.1f}s"
            if col == "mean_n_attempts":
                return f"{v:.2f}"
            if col == "low_conf_count":
                return str(int(v))
            return f"{v:.4f}"
        print(f"{lbl:<24}" + "".join(f"{_fmt(v):<{col_w}}" for v in vals))
    print(sep)


# ── score-from-cache core ─────────────────────────────────────────────────────

def _score_from_cache(
    cache_path: Path,
    results_path: Path,
    llm_w,
    emb_w,
    conditions_to_score: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Load generation_cache.csv (answers/contexts) and results_main.csv (existing
    partial RAGAS cols).  Compute RAGAS metrics only for conditions that are
    missing metrics or have a silent-zero failure on context_precision.

    Detects silent-zero failures (context_precision == 0.0 for ALL rows) and
    forces those columns to be re-scored even if they already exist.

    Returns (df_with_ragas_cols, ragas_results_dict).
    """
    # Load generation cache — answers and contexts live here
    df_gen = pd.read_csv(cache_path)
    if "agentic_low_conf" in df_gen.columns:
        df_gen["agentic_low_conf"] = df_gen["agentic_low_conf"].map(
            lambda x: str(x).strip().lower() == "true"
        )

    # Load existing RAGAS results if available (may contain partial scores)
    if results_path.exists():
        df_existing = pd.read_csv(results_path)
        # Drop aggregate row if present so lengths match
        df_existing = df_existing[
            df_existing["question"] != "** AGGREGATE MEAN **"
        ].reset_index(drop=True)
    else:
        df_existing = df_gen.copy()

    all_conditions = ["baseline", "reranker", "agentic"]
    if conditions_to_score is None:
        conditions_to_score = all_conditions

    nan = float("nan")
    ragas_results: Dict[str, Dict] = {}
    q_list  = df_gen["question"].tolist()
    gt_list = df_gen["ground_truth"].tolist()

    # Detect silent-zero failures from the existing results file
    suspect_cols = _detect_zero_failure_columns(df_existing, all_conditions)
    if suspect_cols:
        print(f"\n[score-from-cache] Detected silent-zero RAGAS failure in: {suspect_cols}")
        print("  These columns will be re-scored regardless of prior values.")

    # Pre-populate ragas_results from df_existing (not from cache)
    for condition in all_conditions:
        existing: Dict[str, float] = {}
        for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
            col = f"{condition}_{metric}"
            suspect = col in suspect_cols
            if col in df_existing.columns and not suspect:
                val = df_existing[col].mean(skipna=True)
                if val == val:   # not NaN
                    existing[metric] = val
        ragas_results[condition] = existing

    # Start building output df from gen cache; merge in non-suspect RAGAS cols
    df = df_gen.copy()
    if len(df_existing) == len(df):
        for condition in all_conditions:
            for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
                col = f"{condition}_{metric}"
                if col in df_existing.columns and col not in suspect_cols:
                    df[col] = df_existing[col].values

    for condition in conditions_to_score:
        missing_metrics = [
            m for m in ["faithfulness", "answer_relevancy", "context_precision"]
            if m not in ragas_results.get(condition, {})
            or f"{condition}_{m}" in suspect_cols
        ]
        if not missing_metrics:
            print(f"[score-from-cache] {condition}: all metrics already present — skipping.")
            continue

        print(f"\nComputing RAGAS metrics for condition: {condition}  "
              f"(missing/re-scoring: {missing_metrics})…")
        a_list   = df_gen[f"{condition}_answer"].tolist()
        ctx_list = [json.loads(x) for x in df_gen[f"{condition}_contexts"].tolist()]

        per_df, means, stats = _compute_ragas(q_list, a_list, ctx_list, gt_list, llm_w, emb_w)
        ragas_results[condition].update(means)
        _record_failure_stats(condition, stats)

        # Write PER-QUERY scores back to df (one value per row, NOT broadcast means)
        for m in _METRIC_KEYS:
            df[f"{condition}_{m}"] = per_df[m].values

        # Partial save after each condition
        df.to_csv(results_path, index=False)
        print(f"  Partial results written → {results_path}")

    # Ensure all metric columns exist (fill NaN for any that were never scored)
    for condition in all_conditions:
        for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
            col = f"{condition}_{metric}"
            if col not in df.columns:
                df[col] = nan

    return df, ragas_results


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agentic RAG pipeline")
    parser.add_argument(
        "--threshold-sweep", action="store_true",
        help="Run Condition 3 only at τ values",
    )
    parser.add_argument(
        "--tau-values", nargs="+", type=float, default=[0.40, 0.50, 0.60],
        help="τ values for threshold sweep (default: 0.40 0.50 0.60)",
    )
    parser.add_argument(
        "--sweep-from-cache", action="store_true",
        help=(
            "Use with --threshold-sweep: skip re-generation for τ=0.40 "
            "(already in generation_cache.csv); only generate for new τ values."
        ),
    )
    parser.add_argument(
        "--score-only", "--score-from-cache", action="store_true",
        dest="score_only",
        help=(
            "Skip generation entirely. Load results/generation_cache.csv, "
            "score missing or suspect-zero conditions with RAGAS, write results_main.csv."
        ),
    )
    parser.add_argument(
        "--max-queries", type=int, default=None,
        help="Truncate query list to N (useful for smoke-testing)",
    )
    parser.add_argument(
        "--tau", type=float, default=None,
        help="Override config.TAU for the main 3-condition run",
    )
    args = parser.parse_args()

    if args.tau is not None:
        config.TAU = args.tau
        print(f"[evaluate] config.TAU overridden to {config.TAU}")

    # ── Load queries ──────────────────────────────────────────────────────────
    if not config.QUERIES_FILE.exists():
        print(f"ERROR: {config.QUERIES_FILE} not found.")
        sys.exit(1)

    with open(config.QUERIES_FILE, encoding="utf-8") as fh:
        queries = json.load(fh)
    if args.max_queries is not None:
        queries = queries[: args.max_queries]
    print(f"Loaded {len(queries)} queries from {config.QUERIES_FILE}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = config.RESULTS_DIR / "generation_cache.csv"
    out_path   = config.RESULTS_DIR / "results_main.csv"

    # ── score-only / score-from-cache mode ───────────────────────────────────
    if args.score_only:
        if not cache_path.exists():
            print(f"ERROR: --score-only requested but {cache_path} does not exist.")
            print("Run without --score-only first to generate and cache results.")
            sys.exit(1)
        print(f"\nLoading generation cache from {cache_path}…")

        llm_w, emb_w = _build_ragas_llm_and_embeddings()
        df, ragas_results = _score_from_cache(cache_path, out_path, llm_w, emb_w)

        # Add aggregate row and save
        agg: Dict[str, Any] = {"question": "** AGGREGATE MEAN **", "ground_truth": ""}
        for condition in ["baseline", "reranker", "agentic"]:
            for col in [f"{condition}_latency", f"{condition}_faithfulness",
                        f"{condition}_answer_relevancy", f"{condition}_context_precision"]:
                if col in df.columns:
                    agg[col] = df[col].mean(skipna=True)
        for col in ["agentic_n_attempts", "agentic_confidence"]:
            if col in df.columns:
                agg[col] = df[col].mean(skipna=True)

        df_agg = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
        df_agg.to_csv(out_path, index=False)

        _print_summary(df, ragas_results)
        print(f"\nFull results → {out_path}")
        return

    # ── Threshold sweep mode ──────────────────────────────────────────────────
    if args.threshold_sweep:
        retriever = Retriever()
        llm_w, emb_w = _build_ragas_llm_and_embeddings()

        if args.sweep_from_cache and cache_path.exists():
            # τ=0.40 is already in the cache; read those answers directly.
            # For other τ values, re-generate agentic answers.
            print(f"\nThreshold sweep — reading τ=0.40 from cache, generating others.")
            cache_df = pd.read_csv(cache_path)
            cached_tau = 0.40

            tau_values_to_generate = [t for t in args.tau_values if t != cached_tau]
            tau_values_from_cache  = [t for t in args.tau_values if t == cached_tau]

            # Build per_tau_data for cached τ
            per_tau_data: Dict[float, Dict] = {}
            for tau in tau_values_from_cache:
                per_tau_data[tau] = {
                    "questions":     cache_df["question"].tolist(),
                    "answers":       cache_df["agentic_answer"].tolist(),
                    "contexts_list": [json.loads(x) for x in cache_df["agentic_contexts"]],
                    "ground_truths": cache_df["ground_truth"].tolist(),
                }
                print(f"  τ={tau}: loaded {len(cache_df)} rows from cache.")

            # Generate for remaining τ values
            if tau_values_to_generate:
                print(f"\nGenerating agentic answers for τ={tau_values_to_generate}…")
                _, new_data = _run_threshold_sweep(
                    queries, retriever, tau_values=tau_values_to_generate
                )
                per_tau_data.update(new_data)

            # Build agg rows (latency/attempt stats from cache for τ=0.40; computed for others)
            agg_rows: List[Dict] = []
            for tau in args.tau_values:
                if tau in tau_values_from_cache:
                    agg_rows.append({
                        "tau":             tau,
                        "mean_latency_s":  cache_df["agentic_latency"].mean(),
                        "mean_n_attempts": cache_df["agentic_n_attempts"].mean(),
                        "mean_confidence": cache_df["agentic_confidence"].mean(),
                        "low_conf_count":  cache_df["agentic_low_conf"].map(
                            lambda x: str(x).strip().lower() == "true"
                        ).sum(),
                    })
                else:
                    d = per_tau_data[tau]
                    agg_rows.append({
                        "tau":             tau,
                        "mean_latency_s":  float("nan"),
                        "mean_n_attempts": float("nan"),
                        "mean_confidence": float("nan"),
                        "low_conf_count":  0,
                    })
            df_sweep = pd.DataFrame(agg_rows)
        else:
            print(f"\nThreshold sweep over τ = {args.tau_values}")
            df_sweep, per_tau_data = _run_threshold_sweep(
                queries, retriever, tau_values=args.tau_values
            )

        # Score with RAGAS
        ragas_rows = []
        for tau in args.tau_values:
            if RAGAS_OK and llm_w is not None:
                print(f"\nComputing RAGAS metrics for τ={tau}…")
                data = per_tau_data[tau]
                _, r_metrics, sweep_stats = _compute_ragas(
                    data["questions"], data["answers"],
                    data["contexts_list"], data["ground_truths"],
                    llm_w, emb_w,
                )
                _record_failure_stats(f"agentic_tau={tau}", sweep_stats)
            else:
                r_metrics = {"faithfulness": float("nan"), "answer_relevancy": float("nan"),
                             "context_precision": float("nan")}
            ragas_rows.append(r_metrics)

        df_ragas  = pd.DataFrame(ragas_rows)
        df_sweep  = pd.concat([df_sweep.reset_index(drop=True), df_ragas], axis=1)

        out_sweep = config.RESULTS_DIR / "threshold_sweep.csv"
        df_sweep.to_csv(out_sweep, index=False)
        print(f"\nThreshold sweep saved → {out_sweep}")
        _print_sweep_summary(df_sweep, n_queries=len(queries))
        return

    # ── Full evaluation (generation + scoring) ────────────────────────────────
    reranker  = RerankerRAG()
    retriever = Retriever()

    print(f"\nRunning all 3 conditions on {len(queries)} queries…")
    df = _run_all_conditions(queries, retriever, reranker)

    df.to_csv(cache_path, index=False)
    print(f"\nGeneration cache saved → {cache_path}")

    llm_w, emb_w = _build_ragas_llm_and_embeddings()
    ragas_results: Dict[str, Dict] = {}
    gt_list = df["ground_truth"].tolist()

    for condition in ["baseline", "reranker", "agentic"]:
        print(f"\nComputing RAGAS metrics for condition: {condition}…")
        q_list   = df["question"].tolist()
        a_list   = df[f"{condition}_answer"].tolist()
        ctx_list = [json.loads(x) for x in df[f"{condition}_contexts"].tolist()]
        per_df, means, stats = _compute_ragas(q_list, a_list, ctx_list, gt_list, llm_w, emb_w)
        ragas_results[condition] = means
        _record_failure_stats(condition, stats)
        # PER-QUERY scores (one value per row, NOT broadcast means)
        for m in _METRIC_KEYS:
            df[f"{condition}_{m}"] = per_df[m].values

        df.to_csv(out_path, index=False)
        print(f"  Partial results written → {out_path}")

    agg: Dict[str, Any] = {"question": "** AGGREGATE MEAN **", "ground_truth": ""}
    for condition in ["baseline", "reranker", "agentic"]:
        for col in [f"{condition}_latency", f"{condition}_faithfulness",
                    f"{condition}_answer_relevancy", f"{condition}_context_precision"]:
            agg[col] = df[col].mean(skipna=True)
    for col in ["agentic_n_attempts", "agentic_confidence"]:
        agg[col] = df[col].mean(skipna=True)

    df_agg = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)
    df_agg.to_csv(out_path, index=False)

    _print_summary(df, ragas_results)
    print(f"\nFull results → {out_path}")


if __name__ == "__main__":
    main()