"""
Streamlit resource cache for expensive one-time loads.

These functions are decorated with st.cache_resource so Streamlit keeps
the loaded objects alive across re-runs for the lifetime of the server
process. Without this, every button click re-loads the embedding model
(~40 seconds on CPU).

Usage in pages:
    from src.ui.cache import get_cached_index, get_cached_embedder
    index = get_cached_index()          # loads once per session
    model = get_cached_embedder()       # loads once per session
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of where Streamlit runs from
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st


@st.cache_resource(show_spinner="Loading retrieval index…")
def get_cached_index(brand: str = "AmazonHelp"):
    """
    Load (or build) the vector retrieval index once per server process.
    Subsequent calls return the same in-memory VectorIndex object instantly.
    """
    from src.retrieval.index import VectorIndex
    idx = VectorIndex()
    idx.build(brand)
    return idx


@st.cache_resource(show_spinner="Loading embedding model…")
def get_cached_embedder():
    """
    Load the sentence-transformer model once per server process.
    Returns the raw SentenceTransformer object.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")
