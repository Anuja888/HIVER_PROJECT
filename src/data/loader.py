"""
Phase 2 — Data Ingestion & Brand Filtering.

Streams the CSV in chunks, never loading all 3M rows at once.
Filters to rows that belong to conversations touching the chosen brand.

Decision log:
- Chunk size 100k: balances memory use vs. I/O overhead.
- response_tweet_id kept as str (can be comma-sep list like "5,7").
- in_response_to_tweet_id coerced to nullable Int64 after load.
- Only rows where author_id == brand OR in_response_to_tweet_id links
  into a brand tweet are kept (two-pass approach: first pass collects
  brand tweet IDs, second pass pulls all connected rows).
"""
from __future__ import annotations

import logging
from typing import Iterator

import pandas as pd
from tqdm import tqdm

from src.config.settings import settings

log = logging.getLogger(__name__)

CHUNK_SIZE = 100_000
CSV_TOTAL_ROWS = 3_003_124  # known row count (excl. header)

RAW_DTYPES = {
    "tweet_id": "int64",
    "author_id": "str",
    "inbound": "bool",
    "created_at": "str",
    "text": "str",
    "response_tweet_id": "str",  # comma-sep list possible
}


def _iter_csv_chunks() -> Iterator[pd.DataFrame]:
    """Yield raw chunks from the full CSV."""
    reader = pd.read_csv(
        settings.raw_data_file,
        chunksize=CHUNK_SIZE,
        dtype=RAW_DTYPES,
        low_memory=False,
    )
    for chunk in reader:
        chunk["in_response_to_tweet_id"] = pd.to_numeric(
            chunk["in_response_to_tweet_id"], errors="coerce"
        ).astype("Int64")
        yield chunk


def load_brand_threads(brand: str, *, verbose: bool = True) -> pd.DataFrame:
    """
    Load ALL rows that are part of conversations involving `brand`.

    Two-pass strategy:
      Pass 1: collect all tweet_ids authored by `brand`.
      Pass 2: keep any row where author_id==brand OR
              in_response_to_tweet_id is a brand tweet ID.

    This captures: brand outbound messages + direct customer replies to them.
    Combined with thread reconstruction, full conversation trees are recovered.

    Returns a DataFrame with the raw columns plus `in_response_to_tweet_id`
    as nullable Int64.
    """
    cache_path = settings.cache_dir / f"brand_{brand}_raw.parquet"
    if cache_path.exists():
        if verbose:
            log.info("Loading brand rows from cache: %s", cache_path)
            print(f"[loader] Cache hit: {cache_path}")
        return pd.read_parquet(cache_path)

    if verbose:
        print(f"[loader] Pass 1: collecting {brand} tweet IDs...")

    # Pass 1 — fast: only read author_id + tweet_id columns
    brand_ids: set[int] = set()
    for chunk in tqdm(_iter_csv_chunks(), desc="  pass1", unit="chunk",
                      disable=not verbose):
        mask = chunk["author_id"] == brand
        brand_ids.update(chunk.loc[mask, "tweet_id"].astype(int).tolist())

    if verbose:
        print(f"[loader]   -> {len(brand_ids):,} brand tweet IDs found")
        print("[loader] Pass 2: pulling all connected rows...")

    # Pass 2 — keep rows authored by brand OR replying to brand
    parts: list[pd.DataFrame] = []
    for chunk in tqdm(_iter_csv_chunks(), desc="  pass2", unit="chunk",
                      disable=not verbose):
        authored = chunk["author_id"] == brand
        reply_to_brand = chunk["in_response_to_tweet_id"].isin(brand_ids)
        mask = authored | reply_to_brand
        if mask.any():
            parts.append(chunk[mask].copy())

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(subset=["tweet_id"])

    if verbose:
        print(f"[loader]   -> {len(df):,} rows kept")

    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    if verbose:
        print(f"[loader] Cached -> {cache_path}")
    return df
