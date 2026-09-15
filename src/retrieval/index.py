"""
Phase 6 — Vector Index & Evidence Retrieval.

Uses a simple cosine-similarity search over numpy arrays (no FAISS dependency).
At the scale of 3k–5k training threads, brute-force dot-product is fast enough
(~10ms for top-5 over 3k 384-dim vectors).

Decision log:
  - FAISS is optional — at 3k vectors, numpy matmul is <10ms, FAISS adds
    installation complexity for no measurable speedup. Only switch to FAISS
    if indexing 100k+ vectors.
  - Index is built from TRAINING split only. Never val or test — would be
    retrieval leakage.
  - Each indexed item is the customer message text + thread_id + brand reply.
    The brand's last reply is the "evidence" (how they actually resolved it).
  - Similarity threshold from config.yaml (retrieval.min_similarity = 0.35).
    Below this, evidence is considered "thin" and influences escalation.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config.settings import settings
from src.retrieval.embedder import embed, embed_cached

log = logging.getLogger(__name__)

TOP_K = 5
MIN_SIMILARITY = 0.35
MAX_INDEX_SIZE = 5000  # cap to keep embedding time fast on CPU


@dataclass
class EvidenceItem:
    thread_id: int
    customer_message: str
    brand_reply: str
    similarity: float
    relevance_reason: str


class RetrievalResult:
    def __init__(self, items: list[EvidenceItem], query: str):
        self.items = items
        self.query = query

    @property
    def top_similarity(self) -> float:
        return self.items[0].similarity if self.items else 0.0

    @property
    def has_sufficient_evidence(self) -> bool:
        return self.top_similarity >= MIN_SIMILARITY and len(self.items) >= 2

    def to_dict(self) -> list[dict]:
        return [
            {
                "thread_id": e.thread_id,
                "customer_message": e.customer_message,
                "brand_reply": e.brand_reply,
                "similarity": round(e.similarity, 4),
                "relevance_reason": e.relevance_reason,
            }
            for e in self.items
        ]


class VectorIndex:
    """
    In-memory cosine similarity index over customer messages.
    Build once from the training split; persist to disk for fast reload.
    """

    def __init__(self):
        self.vectors: Optional[np.ndarray] = None   # (N, 384)
        self.metadata: list[dict] = []
        self._built = False

    def build(self, brand: str, force: bool = False) -> None:
        """Build index from training split. Caches to disk."""
        index_path = settings.cache_dir / f"retrieval_index_{brand}.pkl"

        if index_path.exists() and not force:
            log.info("Loading retrieval index from %s", index_path)
            print("[index] Loading index from cache...")
            with open(index_path, "rb") as f:
                data = pickle.load(f)
            self.vectors = data["vectors"]
            self.metadata = data["metadata"]
            self._built = True
            print(f"[index] Loaded {len(self.metadata):,} items")
            return

        print(f"[index] Building retrieval index for {brand}...")
        train_path = settings.cache_dir / f"{brand}_train.jsonl"
        if not train_path.exists():
            raise FileNotFoundError("Run data pipeline first.")

        records: list[dict] = []
        with open(train_path, encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                # Get first customer message + last brand reply
                cust_msg = next(
                    (m["text"] for m in t["messages"]
                     if m["role"] == "customer" and m["text"].strip()),
                    None
                )
                brand_reply = next(
                    (m["text"] for m in reversed(t["messages"])
                     if m["role"] == "brand" and m["text"].strip()),
                    None
                )
                if cust_msg and brand_reply:
                    records.append({
                        "thread_id": t["thread_id"],
                        "customer_message": cust_msg,
                        "brand_reply": brand_reply,
                    })

        # Cap index size for speed
        if len(records) > MAX_INDEX_SIZE:
            import random
            rng = random.Random(settings.sample_seed)
            records = rng.sample(records, MAX_INDEX_SIZE)

        print(f"[index] Embedding {len(records):,} training examples...")
        texts = [r["customer_message"] for r in records]
        vecs = embed_cached(texts, f"{brand}_train_{len(records)}",
                            show_progress=True)

        self.vectors = vecs
        self.metadata = records
        self._built = True

        with open(index_path, "wb") as f:
            pickle.dump({"vectors": vecs, "metadata": records}, f)
        print(f"[index] Index saved -> {index_path} ({len(records):,} items)")

    def search(self, query: str, top_k: int = TOP_K) -> RetrievalResult:
        """Return top-k most similar training examples for the query."""
        if not self._built:
            raise RuntimeError("Index not built. Call build() first.")

        q_vec = embed([query])  # (1, 384)
        # Dot product = cosine similarity (vectors are normalized)
        sims = (self.vectors @ q_vec.T).squeeze()  # (N,)
        top_idx = np.argsort(sims)[::-1][:top_k]

        items = []
        for idx in top_idx:
            sim = float(sims[idx])
            if sim < 0:
                continue
            meta = self.metadata[idx]
            # Simple relevance reason based on similarity score
            if sim >= 0.80:
                reason = "Very high semantic similarity — near-identical issue."
            elif sim >= 0.60:
                reason = "High similarity — same intent category, similar language."
            elif sim >= 0.40:
                reason = "Moderate similarity — related topic, different phrasing."
            else:
                reason = "Low similarity — weak evidence; treat with caution."

            items.append(EvidenceItem(
                thread_id=meta["thread_id"],
                customer_message=meta["customer_message"],
                brand_reply=meta["brand_reply"],
                similarity=sim,
                relevance_reason=reason,
            ))

        return RetrievalResult(items=items, query=query)


# Module-level singleton for CLI / pytest use
_index: Optional[VectorIndex] = None


def get_index(brand: str = "AmazonHelp", force: bool = False) -> VectorIndex:
    """
    Return a built VectorIndex.

    When running inside Streamlit: delegates to the st.cache_resource wrapper
    in src.ui.cache so the index (including the 5k-vector numpy array) is
    loaded once per server process — not once per button click.
    Falls back to a plain module-level singleton for CLI and test runs.
    """
    if not force:
        try:
            import streamlit as st
            if st.runtime.exists():
                from src.ui.cache import get_cached_index
                return get_cached_index(brand)
        except Exception:
            pass  # not in Streamlit — fall through

    # CLI / pytest path
    global _index
    if _index is None or force:
        _index = VectorIndex()
        _index.build(brand, force=force)
    return _index
