"""
Condition 1 — Plain single-pass RAG.

Pipeline: embed query → retrieve top-k → format prompt → generate with llama3.1:8b
No reranking, no critic, no iterative refinement.
"""

import time
from typing import Dict

import requests

import config
from retriever import Retriever


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
                print("  [baseline] Ollama timeout; retrying once…")
            else:
                raise TimeoutError("Ollama generation timed out on retry")
        except Exception as exc:
            raise RuntimeError(f"Ollama call failed: {exc}") from exc
    return ""  # unreachable


def run_baseline(query: str, retriever: Retriever) -> Dict:
    """
    Returns:
        answer     : str
        contexts   : list[str]  — the raw retrieved chunks
        latency_s  : float
    """
    t0 = time.perf_counter()

    chunks = retriever.retrieve(query, k=config.TOP_K)
    context_text = "\n\n---\n\n".join(c["text"] for c in chunks)
    prompt = config.GENERATION_PROMPT.format(context=context_text, question=query)
    answer = _call_ollama(prompt)

    return {
        "answer":    answer.strip(),
        "contexts":  [c["text"] for c in chunks],
        "latency_s": time.perf_counter() - t0,
    }
