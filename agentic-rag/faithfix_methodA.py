"""
faithfix_methodA.py — Revision R3 faithfulness coverage-bias fix, METHOD A.

Pure computation on results/scores_2wiki.csv + results/generation_cache_2wiki.csv.
For each condition's 100 rows, find faithfulness == NaN. If the corresponding
answer is a REFUSAL (same heuristic family used in Chunk F), impute
faithfulness = 1.0 (a refusal makes no unsupported claims). If NaN for a
non-refusal reason (genuine parser failure on a substantive answer), leave it
NaN and mark it 'pending' for Method B re-parse.

Writes:
  results/faithfix_methodA.csv   per-row: question, condition, faith_orig,
                                 refusal, faith_A, status
  results/faithfix_pending.json  the non-refusal NaN rows (question, condition,
                                 answer, contexts, ground_truth) for Method B.
Prints per-condition Method-A means and the imputed / still-NaN counts.
"""
import json
import re
import sys

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CONDITIONS = ["baseline", "reranker", "agentic"]

_REF = re.compile(
    r"(does not|do not|doesn't|don't) contain (enough|sufficient)"
    r"|not contain (enough|sufficient)"
    r"|(don't|do not|cannot|can't|could not|couldn't) (know|answer|find|determine)"
    r"|no (relevant|sufficient|enough) (information|context|evidence)"
    r"|not (enough|sufficient) (information|context|evidence)"
    r"|unable to (answer|determine|find)"
    r"|cannot be (determined|answered)"
    r"|insufficient (information|context|evidence)"
    r"|i'm sorry"
    r"|there is no (information|mention|indication)", re.I)


def is_refusal(x):
    s = "" if pd.isna(x) else str(x).strip()
    return s == "" or bool(_REF.search(s))


def main():
    sc = pd.read_csv("results/scores_2wiki.csv")
    cache = pd.read_csv("results/generation_cache_2wiki.csv")

    ans, ctx, gt = {}, {}, {}
    for _, r in cache.iterrows():
        gt[r["question"]] = r.get("ground_truth", "")
        for c in CONDITIONS:
            ans[(r["question"], c)] = r[f"{c}_answer"]
            try:
                ctx[(r["question"], c)] = json.loads(r[f"{c}_contexts"])
            except Exception:
                ctx[(r["question"], c)] = []

    rows, pending = [], []
    for _, r in sc.iterrows():
        q, c, f = r["question"], r["condition"], r["faithfulness"]
        a = ans.get((q, c))
        ref = is_refusal(a)
        if pd.notna(f):
            faith_a, status = float(f), "scored"
        elif ref:
            faith_a, status = 1.0, "imputed"
        else:
            faith_a, status = float("nan"), "pending"
            pending.append({"question": q, "condition": c,
                            "answer": "" if pd.isna(a) else str(a),
                            "contexts": ctx.get((q, c), []),
                            "ground_truth": gt.get(q, "")})
        rows.append({"question": q, "condition": c, "faith_orig": f,
                     "refusal": ref, "faith_A": faith_a, "status": status})

    out = pd.DataFrame(rows)
    out.to_csv("results/faithfix_methodA.csv", index=False)
    with open("results/faithfix_pending.json", "w", encoding="utf-8") as fh:
        json.dump(pending, fh, ensure_ascii=False, indent=2)

    print("METHOD A — refusal imputation (faithfulness)")
    print(f"{'cond':9s} {'orig_scored':>11s} {'imputed':>8s} "
          f"{'still_NaN':>9s} {'meanA(recovered)':>17s} {'n_A':>4s}")
    for c in CONDITIONS:
        sub = out[out.condition == c]
        n_orig = (sub.status == "scored").sum()
        n_imp = (sub.status == "imputed").sum()
        n_pend = (sub.status == "pending").sum()
        mean_a = sub["faith_A"].mean(skipna=True)
        n_a = sub["faith_A"].notna().sum()
        print(f"{c:9s} {n_orig:11d} {n_imp:8d} {n_pend:9d} "
              f"{mean_a:17.4f} {n_a:4d}")
    print(f"\nPending non-refusal NaN rows for Method B: {len(pending)} "
          f"-> results/faithfix_pending.json")


if __name__ == "__main__":
    main()
