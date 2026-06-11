"""
Central configuration. Change any hyperparameter here; all modules import from this file.
"""

import os
# Force PyTorch-only mode in HuggingFace transformers before any model library loads.
# Prevents the Keras-3 / tf-keras conflict when TensorFlow is also installed.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR    = PROJECT_ROOT / "data"
CORPUS_DIR  = DATA_DIR / "corpus_hotpot"
QUERIES_FILE = DATA_DIR / "queries_hotpot.json"
RESULTS_DIR = PROJECT_ROOT / "results"
CHROMA_DIR  = PROJECT_ROOT / "chroma_db"

# ── Models ──────────────────────────────────────────────────────────────────
EMBED_MODEL    = "BAAI/bge-base-en-v1.5"
LLM_MODEL      = "llama3.1:8b"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT  = 180   # seconds; long generations need patience

# ── Generation (SHARED across all three conditions) ──────────────────────────
# The prompt and sampling options below are the single source of truth for
# baseline, reranker, and agentic generation. They MUST stay identical across
# conditions or the three-way comparison is invalid.
GENERATION_PROMPT = """\
You are a helpful assistant. Answer the question using ONLY the information \
in the provided context.

Rules:
- Answer with a complete, specific, self-contained sentence that directly \
answers the question. Do not reply with just a bare entity, number, or \
yes/no — restate the relevant fact (e.g. "The capital of France is Paris.").
- Use only information from the context. Do not add outside knowledge.
- If the context does not contain enough information to answer the question, \
say: "The provided context does not contain enough information to answer \
this question."

Context:
{context}

Question: {question}

Answer:"""

# ── Agentic-only best-effort fallback ────────────────────────────────────────
# Used ONLY by agentic_rag.py, and only when the loop ends low-confidence AND
# the standard prompt refused. Baseline/reranker never use this prompt, so
# their generation behaviour is unchanged.
FALLBACK_PROMPT = """\
You are a helpful assistant. The context below may only partially cover the \
question. Your job is to identify the single answer that is BEST SUPPORTED by \
the context and state it.

Rules:
- Reply with EXACTLY ONE complete, specific, self-contained sentence that \
directly answers the question, prefixed with "Based on partial evidence: ".
- Use only information from the context. Do not add outside knowledge.
- Do NOT refuse. Do NOT say the context is insufficient. Always commit to the \
best-supported answer available in the context.

Context:
{context}

Question: {question}

Answer:"""

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# num_ctx=8192 covers the largest context (TOP_K=10 chunks x 640 tokens ≈ 6400
# tokens) plus prompt and answer; Ollama's default (2048–4096) silently
# truncates the prompt from the front otherwise.
GENERATION_OPTIONS = {
    "temperature": 0.1,
    "seed":        RANDOM_SEED,
    "num_ctx":     8192,
    "num_predict": 384,
}

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K         = 10   # candidates retrieved; reranked and trimmed to RERANK_TOP_N
RERANK_TOP_N  = 5    # chunks passed to critic and generator after cross-encoder reranking
CHUNK_SIZE    = 640   # tokens (cl100k_base) — larger for more self-contained chunks
CHUNK_OVERLAP = 128   # tokens — higher overlap preserves cross-boundary context

# ── Agentic RAG ──────────────────────────────────────────────────────────────
TAU   = 0.70   # confidence threshold; sweep with --threshold-sweep
N_MAX = 2      # max re-retrieval attempts; 3rd attempt never raised confidence in calibration

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "rag_corpus"
