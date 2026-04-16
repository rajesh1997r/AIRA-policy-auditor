"""
Clause embedder using sentence-transformers.

Embeds clauses with all-MiniLM-L6-v2 (384-dim), caches to disk.
"""

import os
import logging
from typing import List, Dict

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import to avoid import errors when sentence_transformers is not installed
_model_cache: dict = {}


class ClauseEmbedder:
    """
    Embeds policy clauses using sentence-transformers.

    Model: all-MiniLM-L6-v2 — 384-dim, fast, good semantic similarity.
    Embeddings are L2-normalized (unit sphere), so dot product == cosine similarity.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    EMBED_DIM = 384

    def __init__(self, cache_dir: str = "data/processed"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._model = None

    def _load_model(self):
        """Lazy-load model on first use."""
        if self._model is None:
            if self.MODEL_NAME not in _model_cache:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading model {self.MODEL_NAME}...")
                _model_cache[self.MODEL_NAME] = SentenceTransformer(self.MODEL_NAME)
            self._model = _model_cache[self.MODEL_NAME]

    def _cache_path(self, doc_name: str) -> str:
        return os.path.join(self.cache_dir, f"{doc_name}_embeddings.npy")

    def embed_clauses(
        self, clauses: List[Dict], doc_name: str = "default", force: bool = False
    ) -> np.ndarray:
        """
        Embed a list of clause dicts.

        Args:
            clauses: List of dicts with at least a 'text' key.
            doc_name: Used to name the cache file.
            force: If True, re-embed even if cache exists.

        Returns:
            np.ndarray of shape (N, 384), dtype float32.
        """
        cache_path = self._cache_path(doc_name)

        if not force and os.path.exists(cache_path):
            cached = np.load(cache_path)
            if cached.shape[0] == len(clauses):
                logger.info(f"Loaded {len(clauses)} embeddings from cache: {cache_path}")
                return cached.astype(np.float32)
            logger.warning(
                f"Cache size mismatch ({cached.shape[0]} vs {len(clauses)}), re-embedding."
            )

        self._load_model()
        texts = [c["text"] for c in clauses]
        logger.info(f"Embedding {len(texts)} clauses for '{doc_name}'...")

        embeddings = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # unit sphere: dot product == cosine sim
            convert_to_numpy=True,
        ).astype(np.float32)

        np.save(cache_path, embeddings)
        logger.info(f"Saved embeddings to {cache_path}")
        return embeddings

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single string.

        Returns:
            np.ndarray of shape (384,), dtype float32.
        """
        self._load_model()
        emb = self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb[0].astype(np.float32)


def embed_clauses(
    clauses: List[Dict], doc_name: str = "default", cache_dir: str = "data/processed"
) -> np.ndarray:
    """Convenience function. Returns (N, 384) float32 array."""
    embedder = ClauseEmbedder(cache_dir=cache_dir)
    return embedder.embed_clauses(clauses, doc_name=doc_name)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    sample_clauses = [
        {"text": "Students may use AI tools for research purposes with appropriate attribution.", "id": "C001"},
        {"text": "Submitting AI-generated text as your own work is strictly prohibited.", "id": "C002"},
        {"text": "All submitted work must reflect the student's own original thinking.", "id": "C003"},
    ]

    embedder = ClauseEmbedder(cache_dir="/tmp/aira_test_embeddings")
    embs = embedder.embed_clauses(sample_clauses, doc_name="test")
    print(f"Shape: {embs.shape}")  # (3, 384)
    print(f"Dtype: {embs.dtype}")  # float32

    # Cosine similarity (== dot product since normalized)
    sim_01 = float(np.dot(embs[0], embs[1]))
    sim_02 = float(np.dot(embs[0], embs[2]))
    sim_12 = float(np.dot(embs[1], embs[2]))
    print(f"Cosine sim(C001, C002): {sim_01:.4f}")  # permission vs prohibition — low
    print(f"Cosine sim(C001, C003): {sim_02:.4f}")
    print(f"Cosine sim(C002, C003): {sim_12:.4f}")  # both restrictive — higher

    # Single text
    v = embedder.embed_text("AI use is allowed with disclosure.")
    print(f"Single embed shape: {v.shape}")  # (384,)
