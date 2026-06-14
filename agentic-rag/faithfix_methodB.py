"""
faithfix_methodB.py — Revision R3 faithfulness coverage-bias fix, METHOD B.

Robust re-judge of faithfulness for ONLY the non-refusal NaN rows
(results/faithfix_pending.json) that ragas' strict parser choked on. No answer
regeneration — we re-score faithfulness on the existing cached answer+contexts.

Faithfulness is recomputed with the same two-step logic ragas uses
(statement decomposition -> per-statement NLI verdict against the context) but
with a TOLERANT parser: try strict json.loads, then fall back to regex
extraction of the array / 0-1 verdicts. Judge = config.LLM_MODEL (llama3.1:8b),
temperature 0, one Ollama process.

faithfulness = (#statements supported by context) / (#statements). A row is
"recovered" if >=1 statement was extracted; otherwise it stays NaN.

Writes results/faithfix_methodB.csv:
  question, condition, n_statements, n_supported, faith_B, recovered, note
Prints per-condition recovered count + mean faith_B over recovered rows.

Usage:  python faithfix_methodB.py  > results/faithfix_methodB.log 2>&1
"""
import json
import re
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import config

try:
    from langchain_ollama import ChatOllama
except Exception:
    from langchain_community.chat_models import ChatOllama  # type: ignore

PENDING = "results/faithfix_pending.json"
OUT = "results/faithfix_methodB.csv"

_STMT_PROMPT = (
    "Break the ANSWER into a list of fully self-contained atomic statements "
    "(resolve pronouns, one claim each). Use the QUESTION only for context.\n"
    "Return ONLY a JSON array of strings, nothing else.\n\n"
    "QUESTION: {q}\nANSWER: {a}\n\nJSON array of statements:"
)
_VERDICT_PROMPT = (
    "You are given a CONTEXT and a numbered list of STATEMENTS. For each "
    "statement decide if it can be directly inferred/supported by the CONTEXT.\n"
    "Return ONLY a JSON array of integers (1 = supported by context, "
    "0 = not supported), one per statement, in the same order. Nothing else.\n\n"
    "CONTEXT:\n{ctx}\n\nSTATEMENTS:\n{stmts}\n\nJSON array of 0/1:"
)


def _content(resp):
    return resp.content if hasattr(resp, "content") else str(resp)


def parse_statements(text):
    """Tolerant: JSON array of strings -> list; fallbacks to regex/lines."""
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            out = [str(s).strip() for s in arr if str(s).strip()]
            if out:
                return out
        except Exception:
            pass
        # regex-extract quoted strings inside the bracketed block
        qs = re.findall(r'"([^"]{3,})"', m.group(0))
        if qs:
            return [s.strip() for s in qs if s.strip()]
    # last resort: split into sentences
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 8]


def parse_verdicts(text, n):
    """Tolerant: JSON int array -> list[int]; fallback regex 0/1 tokens."""
    m = re.search(r"\[[^\]]*\]", text, re.S)
    block = m.group(0) if m else text
    try:
        arr = json.loads(block)
        vs = [1 if int(x) >= 1 else 0 for x in arr]
        if vs:
            return vs[:n]
    except Exception:
        pass
    toks = re.findall(r"(?<![0-9])[01](?![0-9])", block)
    return [int(t) for t in toks][:n]


def main():
    pending = json.load(open(PENDING, encoding="utf-8"))
    print(f"[methodB] judge={config.LLM_MODEL}  pending rows={len(pending)}",
          flush=True)
    chat = ChatOllama(model=config.LLM_MODEL,
                      base_url=config.OLLAMA_BASE_URL, temperature=0.0)

    rows = []
    for i, r in enumerate(pending):
        q, c, a = r["question"], r["condition"], r["answer"]
        ctx = "\n\n".join(r["contexts"]) if r["contexts"] else ""
        print(f"\n[methodB] {i+1}/{len(pending)} {c} | {q[:70]}", flush=True)

        stmt_out = _content(chat.invoke(
            _STMT_PROMPT.format(q=q, a=a)))
        stmts = parse_statements(stmt_out)
        if not stmts:
            print("  no statements extracted -> still NaN", flush=True)
            rows.append({"question": q, "condition": c, "n_statements": 0,
                         "n_supported": 0, "faith_B": float("nan"),
                         "recovered": False, "note": "no_statements"})
            continue
        numbered = "\n".join(f"{j+1}. {s}" for j, s in enumerate(stmts))
        verd_out = _content(chat.invoke(
            _VERDICT_PROMPT.format(ctx=ctx, stmts=numbered)))
        verdicts = parse_verdicts(verd_out, len(stmts))
        if not verdicts:
            print(f"  {len(stmts)} stmts but no verdicts parsed -> still NaN",
                  flush=True)
            rows.append({"question": q, "condition": c,
                         "n_statements": len(stmts), "n_supported": 0,
                         "faith_B": float("nan"), "recovered": False,
                         "note": "no_verdicts"})
            continue
        # align lengths (tolerant): score over the verdicts we got
        n_used = min(len(stmts), len(verdicts))
        n_sup = sum(verdicts[:n_used])
        faith = n_sup / n_used
        print(f"  statements={len(stmts)} verdicts={len(verdicts)} "
              f"used={n_used} supported={n_sup} faith_B={faith:.4f}", flush=True)
        rows.append({"question": q, "condition": c, "n_statements": len(stmts),
                     "n_supported": n_sup, "faith_B": round(faith, 6),
                     "recovered": True,
                     "note": f"used{n_used}of{len(stmts)}"})

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("\nMETHOD B — tolerant re-parse results")
    for c in ["baseline", "reranker", "agentic"]:
        sub = out[out.condition == c]
        rec = sub[sub.recovered]
        mean = rec["faith_B"].mean() if len(rec) else float("nan")
        print(f"  {c:9s} pending={len(sub)} recovered={len(rec)} "
              f"still_NaN={len(sub)-len(rec)} mean_faith_B={mean:.4f}")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
