# SC-ARAG — Self-Correcting Agentic RAG on Consumer Hardware

SC-ARAG is an empirical study of confidence-gated, self-correcting
retrieval-augmented generation that runs entirely on consumer hardware. A
chain-of-thought LLM critic scores retrieved context; when mean confidence
falls below a threshold τ, the system reformulates the query, re-retrieves,
merges and deduplicates evidence, and re-generates — with a single
best-effort fallback generation ("Based on partial evidence: …") when the
loop ends low-confidence and the standard prompt refused. The pipeline is
evaluated on the HotpotQA distractor validation set against two controls
(plain RAG and cross-encoder reranked RAG) using per-query RAGAS metrics
(Faithfulness, Answer Relevancy, Context Precision), Wilcoxon signed-rank
tests with bootstrap confidence intervals, and Holm-Bonferroni
multiple-comparison correction. All code, data, per-query results, and
statistical outputs live in [`agentic-rag/`](agentic-rag/).

## Hardware requirements

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 4060 (8 GB VRAM) — any 8 GB+ CUDA GPU works |
| Generator / judge LLM | `llama3.1:8b` via [Ollama](https://ollama.com) (GPU) |
| Embedding model | `BAAI/bge-base-en-v1.5` (CPU) |
| Reranker | `BAAI/bge-reranker-base` cross-encoder (CPU) |
| Vector DB | ChromaDB (persistent, local) |
| Python | 3.12 |

## Reproduction steps

All commands run from inside `agentic-rag/`:

```powershell
cd agentic-rag

# 0. Environment
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# config.py sets USE_TF=0 / USE_TORCH=1 automatically (PyTorch-only HF mode);
# no other environment variables are required.

ollama serve            # if not already running
ollama pull llama3.1:8b

# 1. Build the dataset (HotpotQA distractor validation subset)
python build_datasets.py

# 2. Ingest the corpus into ChromaDB
python ingest.py

# 3. Run all three conditions + per-query RAGAS scoring
#    NOTE: published results use tau = 0.40 (not the config.py default).
python evaluate.py --tau 0.40

# 4. Statistical tests (Wilcoxon signed-rank + bootstrap CIs)
python significance_test.py

# 5. Holm-Bonferroni multiple-comparison correction
python multiple_comparisons.py
```

Outputs land in `agentic-rag/results/`. Prompts are version-controlled in
`agentic-rag/config.py` (generation + fallback), `agentic-rag/critic.py`
(critic), and `agentic-rag/agentic_rag.py` (query reformulation).

## License

- **Code**: MIT (see `LICENSE`).
- **Data**: the HotpotQA-derived corpus and queries in `agentic-rag/data/`
  are [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/),
  following HotpotQA (Yang et al., EMNLP 2018).
