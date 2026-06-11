"""
Condition 2 — RAG + BGE cross-encoder reranking.

Pipeline: embed query → retrieve top-k → rerank with BAAI/bge-reranker-base → generate

This is the traditional-reranker baseline. It tests whether a cheap, non-LLM
reranker achieves parity with the more expensive LLM critic in Condition 3.
The cross-encoder runs on CPU to avoid competing with Ollama for VRAM.
"""

import time
from typing import Dict, List

import requests
from sentence_transformers import CrossEncoder

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
                print("  [reranker] Ollama timeout; retrying once…")
            else:
                raise TimeoutError("Ollama generation timed out on retry")
        except Exception as exc:
            raise RuntimeError(f"Ollama call failed: {exc}") from exc
    return ""  # unreachable


class RerankerRAG:
    """Wraps the BGE cross-encoder; load once and reuse across queries."""

    def __init__(self) -> None:
        print(f"[reranker] Loading cross-encoder: {config.RERANKER_MODEL} (CPU)")
        self.cross_encoder = CrossEncoder(config.RERANKER_MODEL, device="cpu")

    def run(self, query: str, retriever: Retriever) -> Dict:
        """
        Returns:
            answer     : str
            contexts   : list[str]  — reranked chunks (best first)
            latency_s  : float
        """
        t0 = time.perf_counter()

        # Retrieve initial candidate set
        chunks = retriever.retrieve(query, k=config.TOP_K)

        # Cross-encoder reranking
        pairs: List[tuple] = [(query, c["text"]) for c in chunks]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        reranked = [c for _, c in ranked]

        context_text = "\n\n---\n\n".join(c["text"] for c in reranked)
        prompt = config.GENERATION_PROMPT.format(context=context_text, question=query)
        answer = _call_ollama(prompt)

        return {
            "answer":    answer.strip(),
            "contexts":  [c["text"] for c in reranked],
            "latency_s": time.perf_counter() - t0,
        }
