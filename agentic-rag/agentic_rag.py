"""
Condition 3 — Confidence-gated re-retrieval loop (proposed method).

Algorithm (from paper):
    q0 = original query;  q = q0;  D = ∅;  n = 0
    repeat:
        D  = rerank(q0, dedup(D ∪ retrieve(q, k)))[:top_n]   # MERGE, not replace
        C  = mean(critic.score(q0, d) for d in D)
        if C >= τ:
            return generate(q0, D)
        else:
            q  = reformulate(q, D)
            n += 1
    until n >= N_MAX
    return generate(q0, D) with low_confidence=True

Key invariants:
  • Re-retrieval MERGES with prior chunks (dedup by text) so good chunks from
    attempt 1 survive; the merged set is reranked and trimmed to top-N.
  • The reformulated query is used ONLY for retrieval. Reranking, critic
    confidence, and final generation always use the ORIGINAL question — that is
    what the user asked and what the answer is evaluated against.

Logged per query: n_attempts, final_confidence, low_confidence flag, latency.
"""

import time
from typing import Dict, List, Optional

import requests
from sentence_transformers import CrossEncoder

import config
import critic as critic_module
from retriever import Retriever

# Cross-encoder loaded once on first use (CPU; ~400 MB RAM, no VRAM impact).
_reranker: Optional[CrossEncoder] = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"[agentic] Loading cross-encoder: {config.RERANKER_MODEL} (CPU)")
        _reranker = CrossEncoder(config.RERANKER_MODEL, device="cpu")
    return _reranker


_REFORMULATION_PROMPT = """\
You are a search query optimizer. The query below was used to retrieve passages \
from a knowledge base, but none of the retrieved passages were relevant enough.

Your task: rewrite the query so it is more likely to retrieve relevant information.

Apply these strategies as appropriate:
- Expand abbreviations and acronyms to their full form
- Add domain-specific terminology that an expert would use
- Make implicit background assumptions explicit
- If the query is multi-part, focus on its most specific sub-question
- Add synonyms or closely related concepts that might appear in relevant documents

Original query: {query}

Retrieved context (insufficient relevance — first 200 chars of each chunk):
{context_preview}

Write ONLY the reformulated query. One sentence, no preamble, no quotes."""


def _call_ollama(prompt: str) -> str:
    url = f"{config.OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": config.LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": dict(config.GENERATION_OPTIONS),
    }
    for attempt in range(2):
        try:
            resp = requests.post(url, json=payload, timeout=config.OLLAMA_TIMEOUT)
            resp.raise_for_status()
            return resp.json()["response"]
        except requests.exceptions.Timeout:
            if attempt == 0:
                print("  [agentic] Ollama timeout; retrying once…")
            else:
                raise TimeoutError("Ollama generation timed out on retry")
        except Exception as exc:
            raise RuntimeError(f"Ollama call failed: {exc}") from exc
    return ""  # unreachable


def _reformulate_query(original_query: str, chunks: List[Dict]) -> str:
    """Use the LLM to rewrite a query that failed to retrieve confident context."""
    preview = "\n".join(f"- {c['text'][:200]}…" for c in chunks[:3])
    prompt = _REFORMULATION_PROMPT.format(
        query=original_query,
        context_preview=preview,
    )
    try:
        new_query = _call_ollama(prompt).strip().strip('"\'')
        # Fall back to original if the model returns something too short/empty
        return new_query if len(new_query) > 10 else original_query
    except Exception:
        return original_query


def run_agentic(
    query: str,
    retriever: Retriever,
    tau: Optional[float] = None,
) -> Dict:
    """
    Run the confidence-gated re-retrieval loop.

    Args:
        query:     The user's original question.
        retriever: Initialised Retriever instance.
        tau:       Confidence threshold override (defaults to config.TAU).

    Returns dict with keys:
        answer            : str
        contexts          : list[str]   — chunks used for final generation
        latency_s         : float
        n_attempts        : int         — total retrieval calls made (1 = no reformulation)
        final_confidence  : float       — mean critic score on the final retrieval
        low_confidence    : bool        — True if max attempts exhausted without exceeding τ
        used_fallback     : bool        — True if the best-effort fallback replaced a
                                          low-confidence refusal
    """
    if tau is None:
        tau = config.TAU

    t0 = time.perf_counter()

    current_query  = query
    final_chunks:  List[Dict] = []
    final_confidence = 0.0
    low_confidence   = False
    n = 0  # counts reformulations completed

    while True:
        # ── Retrieve TOP_K candidates (reformulated query is used HERE only) ──
        retrieved = retriever.retrieve(current_query, k=config.TOP_K)

        # ── MERGE-AND-DEDUPLICATE with chunks kept from previous attempts ────
        seen: set = set()
        merged: List[Dict] = []
        for c in final_chunks + retrieved:
            if c["text"] not in seen:
                seen.add(c["text"])
                merged.append(c)

        # ── Rerank merged set against the ORIGINAL question, keep top-N ──────
        reranker = _get_reranker()
        ce_pairs = [(query, c["text"]) for c in merged]
        ce_scores = reranker.predict(ce_pairs)
        ranked = sorted(zip(ce_scores, merged), key=lambda x: x[0], reverse=True)
        chunks = [c for _, c in ranked[: config.RERANK_TOP_N]]

        # ── Critic confidence against the ORIGINAL question ──────────────────
        scores = critic_module.score_contexts_batch(
            query, [c["text"] for c in chunks]
        )
        confidence = sum(scores) / len(scores) if scores else 0.0

        final_chunks     = chunks
        final_confidence = confidence

        print(
            f"  [agentic] attempt {n + 1}/{config.N_MAX}  "
            f"confidence={confidence:.3f}  tau={tau:.2f}"
        )

        # ── Confidence gate ───────────────────────────────────────────────────
        if confidence >= tau:
            break   # satisfied — exit loop and generate

        # ── Reformulate & retry ───────────────────────────────────────────────
        new_query = _reformulate_query(current_query, chunks)
        print(f"  [agentic] reformulated → '{new_query[:80]}'")
        current_query = new_query
        n += 1

        if n >= config.N_MAX:
            low_confidence = True
            print(
                f"  [agentic] max attempts ({config.N_MAX}) reached; "
                "generating with low-confidence flag"
            )
            break

    # ── Generate answer from final context, addressing the ORIGINAL question ──
    # (current_query may have drifted via reformulation; the user's question —
    #  and the one RAGAS scores against — is `query`.)
    context_text = "\n\n---\n\n".join(c["text"] for c in final_chunks)
    prompt = config.GENERATION_PROMPT.format(context=context_text, question=query)
    answer = _call_ollama(prompt)

    # ── Best-effort fallback: ONE extra call when the loop ended low-confidence
    # AND the model refused. Replaces the refusal with the best-supported answer.
    used_fallback = False
    if low_confidence and "does not contain enough information" in answer.lower():
        print("  [agentic] low-confidence refusal — attempting best-effort fallback")
        fb_prompt = config.FALLBACK_PROMPT.format(context=context_text, question=query)
        try:
            fb_answer = _call_ollama(fb_prompt).strip()
            if fb_answer:
                answer = fb_answer
                used_fallback = True
        except Exception as exc:
            print(f"  [agentic] fallback generation failed ({exc}); keeping refusal")

    return {
        "answer":           answer.strip(),
        "contexts":         [c["text"] for c in final_chunks],
        "latency_s":        time.perf_counter() - t0,
        "n_attempts":       n + 1,           # total retrieval calls
        "final_confidence": final_confidence,
        "low_confidence":   low_confidence,
        "used_fallback":    used_fallback,
    }
