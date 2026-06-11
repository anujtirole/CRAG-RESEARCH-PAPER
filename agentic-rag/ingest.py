"""
Build the ChromaDB index from documents in data/corpus/.

Supported formats: .txt  .md  .pdf
Chunking:          token-accurate sliding window via tiktoken (cl100k_base)
Embedding:         BAAI/bge-base-en-v1.5 on CPU
Output:            Persistent ChromaDB collection + printed corpus statistics

Usage:
  python ingest.py
  python ingest.py --reset    # force full re-index even if collection exists
"""

import sys
import uuid
import argparse
from pathlib import Path
from typing import List, Tuple

import tiktoken
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader

import config

TOKENIZER = tiktoken.get_encoding("cl100k_base")
BATCH_SIZE = 64   # chunks embedded per SentenceTransformer call


# ── Document loading ──────────────────────────────────────────────────────────

def _load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


# ── Token-accurate chunking ───────────────────────────────────────────────────

def _chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[str]:
    """Sliding-window chunking on the token stream. Never splits mid-token."""
    tokens = TOKENIZER.encode(text)
    if not tokens:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(TOKENIZER.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


# ── Index builder ─────────────────────────────────────────────────────────────

def build_index(reset: bool = False) -> None:
    # Ensure output directories exist
    config.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    doc_paths = sorted(
        p for p in config.CORPUS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in (".txt", ".md", ".pdf")
    )
    if not doc_paths:
        print(f"No documents found in {config.CORPUS_DIR}")
        print("Place your .txt / .md / .pdf files there, then re-run ingest.py")
        sys.exit(1)

    print(f"Found {len(doc_paths)} document(s) in {config.CORPUS_DIR}")

    # ── Load embedding model ──────────────────────────────────────────────────
    print(f"Loading embedding model: {config.EMBED_MODEL} (CPU)")
    embed_model = SentenceTransformer(config.EMBED_MODEL, device="cpu")

    # ── ChromaDB setup ────────────────────────────────────────────────────────
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    existing = [c.name for c in client.list_collections()]
    if config.CHROMA_COLLECTION in existing:
        if reset:
            client.delete_collection(config.CHROMA_COLLECTION)
            print(f"Deleted existing collection '{config.CHROMA_COLLECTION}' (--reset)")
        else:
            print(
                f"Collection '{config.CHROMA_COLLECTION}' already exists. "
                "Pass --reset to rebuild from scratch."
            )
            sys.exit(0)

    collection = client.create_collection(
        config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "l2"},
    )

    # ── Process documents ─────────────────────────────────────────────────────
    all_chunks: List[Tuple[str, dict]] = []   # (text, metadata)
    total_tokens = 0
    total_docs_loaded = 0

    for path in doc_paths:
        try:
            text = _load_document(path)
        except Exception as exc:
            print(f"  WARNING: could not load '{path.name}': {exc}")
            continue

        doc_tokens = len(TOKENIZER.encode(text))
        total_tokens += doc_tokens
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append((chunk, {
                "source":        path.name,
                "chunk_index":   i,
                "total_chunks":  len(chunks),
            }))
        total_docs_loaded += 1
        print(f"  {path.name}: {doc_tokens:,} tokens -> {len(chunks)} chunks")

    if not all_chunks:
        print("No chunks produced. Check document content.")
        sys.exit(1)

    # ── Print corpus statistics (report these in the paper) ──────────────────
    print(f"\n{'=' * 60}")
    print("CORPUS STATISTICS (report verbatim in paper):")
    print(f"  Documents loaded : {total_docs_loaded}")
    print(f"  Total chunks     : {len(all_chunks)}")
    print(f"  Total tokens     : {total_tokens:,}  (cl100k_base tokenizer)")
    print(f"  Chunk size       : {config.CHUNK_SIZE} tokens")
    print(f"  Chunk overlap    : {config.CHUNK_OVERLAP} tokens")
    print(f"{'=' * 60}\n")

    # ── Embed and store ───────────────────────────────────────────────────────
    print(f"Embedding {len(all_chunks)} chunks (batch_size={BATCH_SIZE})…")
    for batch_start in tqdm(range(0, len(all_chunks), BATCH_SIZE), unit="batch"):
        batch = all_chunks[batch_start : batch_start + BATCH_SIZE]
        texts  = [item[0] for item in batch]
        metas  = [item[1] for item in batch]
        ids    = [str(uuid.uuid4()) for _ in batch]

        embeddings = embed_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        ).tolist()

        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metas,
            ids=ids,
        )

    print(f"\nIndex complete. '{config.CHROMA_COLLECTION}' has {collection.count()} chunks.")
    print(f"Path: {config.CHROMA_DIR}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB index from data/corpus/")
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete existing collection and rebuild from scratch",
    )
    args = parser.parse_args()
    build_index(reset=args.reset)
