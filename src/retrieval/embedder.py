"""
Phase 6 — Embedding Model Wrapper.

Uses sentence-transformers (all-MiniLM-L6-v2) locally.
Embeddings are cached to disk to avoid recomputing on every eval run.

Decision log:
  - all-MiniLM-L6-v2 chosen for speed/quality tradeoff on CPU (384-dim,
    ~60ms per 100 sentences on a laptop). Alternatives like mpnet-base-v2
    are better quality but 3x slower — not needed at this scale.
  - Cache is a numpy .npy file keyed by (model_name, content_hash). If the
    training data changes, delete data/cache/embeddings_*.npy to rebuild.
"""
from __future__ import annotations

import logging

import numpy as np

from src.config.settings import settings

log = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None  # module-level cache for CLI / test use


def _get_model():
    """
    Return the SentenceTransformer model.

    When running inside Streamlit: delegates to the st.cache_resource wrapper
    in src.ui.cache so the model is loaded once per server process (not once
    per button click).  Falls back to a plain module-level singleton for CLI
    and test runs where Streamlit is not present.
    """
    # Try the Streamlit cache first — zero overhead on a cache hit.
    try:
        import streamlit as st
        # st.runtime.exists() is True only when a Streamlit server is running.
        if st.runtime.exists():
            from src.ui.cache import get_cached_embedder
            return get_cached_embedder()
    except Exception:
        pass  # not in Streamlit, or cache import failed — fall through

    # CLI / pytest path: use plain module-level singleton
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model %s...", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(texts: list[str], batch_size: int = 128,
          show_progress: bool = False) -> np.ndarray:
    """Embed a list of texts. Returns (N, 384) float32 array."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _get_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity via dot product
    ).astype(np.float32)


def embed_cached(texts: list[str], cache_key: str,
                 show_progress: bool = True) -> np.ndarray:
    """Embed with disk caching. cache_key is used in the filename."""
    cache_path = settings.cache_dir / f"embeddings_{cache_key}.npy"
    if cache_path.exists():
        log.info("Loading embeddings from cache: %s", cache_path)
        return np.load(str(cache_path))

    log.info("Computing embeddings for %d texts (key=%s)...", len(texts), cache_key)
    vecs = embed(texts, show_progress=show_progress)
    np.save(str(cache_path), vecs)
    log.info("Saved embeddings -> %s", cache_path)
    return vecs
