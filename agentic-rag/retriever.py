"""
Embedding + vector-search wrapper around BGE-small and ChromaDB.
Embedding model runs on CPU to preserve all 8 GB VRAM for Ollama.
"""

from typing import List, Dict

from sentence_transformers import SentenceTransformer
import chromadb

import config


class Retriever:
    """Thin wrapper that embeds queries and searches the ChromaDB collection."""

    def __init__(self) -> None:
        print(f"[retriever] Loading embedding model: {config.EMBED_MODEL} (CPU)")
        self.embed_model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = self.client.get_collection(config.CHROMA_COLLECTION)
        print(f"[retriever] ChromaDB collection '{config.CHROMA_COLLECTION}' "
              f"loaded — {self.collection.count()} chunks")

    def embed(self, text: str) -> List[float]:
        return self.embed_model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def retrieve(self, query: str, k: int = config.TOP_K) -> List[Dict]:
        """
        Return the top-k chunks most similar to `query`.

        Each element: {"text": str, "metadata": dict, "distance": float}
        Distance is L2 on unit-normalised vectors (lower = more similar).
        """
        embedding = self.embed(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]
