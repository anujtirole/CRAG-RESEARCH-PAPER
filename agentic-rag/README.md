# SC-ARAG: Self-Correcting Agentic Retrieval-Augmented Generation

SC-ARAG is an empirical study of confidence-gated, self-correcting retrieval-augmented
generation on consumer hardware. A chain-of-thought LLM critic scores retrieved context;
when mean confidence falls below a threshold τ, the system reformulates the query,
re-retrieves, merges and deduplicates evidence, and re-generates — with a single
best-effort fallback generation ("Based on partial evidence: …") when the loop ends
low-confidence and the standard prompt refused. The pipeline is evaluated on the
HotpotQA distractor validation set against two controls (plain RAG and cross-encoder
reranked RAG) using per-query RAGAS metrics (Faithfulness, Answer Relevancy, Context
Precision), Wilcoxon signed-rank tests with bootstrap confidence intervals, and
Holm-Bonferroni multiple-comparison correction. Everything runs locally: llama3.1:8b
via Ollama on an 8 GB consumer GPU, with embeddings and reranking on CPU.

## Hardware / software requirements

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 4060 Laptop (8 GB VRAM) — any 8 GB+ CUDA GPU works |
| Generator / judge LLM | `llama3.1:8b` via [Ollama](https://ollama.com) (GPU) |
| Embedding model | `BAAI/bge-base-en-v1.5` (CPU, sentence-transformers) |
| Reranker | `BAAI/bge-reranker-base` cross-encoder (CPU) |
| Vector DB | ChromaDB (persistent, local, collection `rag_corpus`) |
| Python | 3.12 |
| OS | Windows 11 (any OS with Python + Ollama works) |

## Reproduction steps

```powershell
# 0. Environment
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# config.py sets USE_TF=0 / USE_TORCH=1 automatically (PyTorch-only HF mode);
# no other environment variables are required.

ollama serve            # if not already running
ollama pull llama3.1:8b

# 1. Build the dataset (HotpotQA distractor validation subset; CC BY-SA 4.0)
python build_datasets.py

# 2. Ingest the corpus into ChromaDB (tiktoken chunking -> BGE embeddings)
python ingest.py

# 3. Validate the critic (calibration check: 20 relevant + 20 irrelevant pairs)
python critic.py --calibration-check

# 4. Run all three conditions + per-query RAGAS scoring
#    NOTE: published results use tau = 0.40 (not the config.py default).
python evaluate.py --tau 0.40

# 5. Statistical tests (Wilcoxon signed-rank + bootstrap CIs)
python significance_test.py

# 6. Holm-Bonferroni multiple-comparison correction
python multiple_comparisons.py
```

Outputs land in `results/`: `generation_cache.csv` (per-query answers/contexts/latencies),
`results_main.csv` (per-query RAGAS scores), `significance.csv`, `significance_holm.csv`,
and `ragas_failure_report.txt` (judge failure accounting).

## Prompts

All prompts are version-controlled in code:

- **Generation prompt** (identical across all three conditions) and sampling options:
  `config.py` → `GENERATION_PROMPT`, `GENERATION_OPTIONS`
- **Best-effort fallback prompt** (agentic condition only): `config.py` → `FALLBACK_PROMPT`
- **Critic prompt** (justify-first, score-last chain-of-thought): `critic.py`
- **Query reformulation prompt**: `agentic_rag.py`

## Key hyperparameters (`config.py`)

| Variable | Value | Description |
|----------|-------|-------------|
| `TAU` | 0.40 in published runs (`--tau 0.40`) | Confidence gate for re-retrieval |
| `N_MAX` | 2 | Max re-retrieval attempts |
| `TOP_K` | 10 | Candidates retrieved per query |
| `RERANK_TOP_N` | 5 | Chunks kept after cross-encoder reranking |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 640 / 128 | Tokens (cl100k_base) |
| `RANDOM_SEED` | 42 | Fixed everywhere |

## Project layout

```
config.py             central hyperparameters + shared prompts
build_datasets.py     HotpotQA distractor validation -> corpus + queries
ingest.py             chunking -> BGE embeddings -> ChromaDB
retriever.py          embedding + ChromaDB query wrapper
critic.py             LLM confidence critic (+ --calibration-check)
baseline_rag.py       Condition 1: plain RAG
reranker_rag.py       Condition 2: cross-encoder reranked RAG
agentic_rag.py        Condition 3: confidence-gated re-retrieval + fallback
evaluate.py           runs all conditions, per-query RAGAS, failure report
significance_test.py  Wilcoxon signed-rank + bootstrap CIs
multiple_comparisons.py  Holm-Bonferroni correction
data/                 corpus_hotpot/ + queries_hotpot*.json
results/              per-query CSVs, significance tables, failure reports, archives
```

## License

- **Code**: MIT (see `LICENSE`).
- **Data**: the HotpotQA-derived corpus and queries in `data/` are
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), following
  HotpotQA (Yang et al., EMNLP 2018).

## Citation

```bibtex
@misc{tirole2026scarag,
  title  = {SC-ARAG: Self-Correcting Agentic Retrieval-Augmented Generation
            on Consumer Hardware},
  author = {Tirole, Anuj},
  year   = {2026},
  note   = {Manuscript in preparation. Code: https://github.com/<user>/sc-arag}
}
```
