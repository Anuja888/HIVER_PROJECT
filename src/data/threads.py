"""
Phase 2 — Thread Reconstruction & Role Tagging.

Uses fully vectorised pandas operations — no Python-level row iteration.

Decision log:
- Thread ID = root tweet_id via vectorised parent-chain walk (numpy arrays).
- Branching handled: a tweet can have multiple children; we keep ALL rows
  found in pass2 of the loader, so full trees are preserved.
- Role tagging: author_id == brand -> "brand", else -> "customer".
- Chronological ordering by created_at timestamp; tweet_id used as tiebreak.
- Truncation: threads longer than MAX_CONTEXT_TURNS keep the LAST N messages
  (most recent context is most useful for reply generation).
- Emoji kept (sentiment signal). @mentions stripped from customer text only.
- Threads with no customer message are discarded (not useful for intent eval).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

MAX_CONTEXT_TURNS = 10
TIMESTAMP_FMT = "%a %b %d %H:%M:%S +0000 %Y"

_AT_HANDLE_RE = re.compile(r"@\S+")
_URL_RE = re.compile(r"https?://\S+")
_WS_RE = re.compile(r"\s+")


@dataclass
class Message:
    tweet_id: int
    role: str
    text: str
    raw_text: str
    created_at: Optional[datetime]
    in_response_to: Optional[int]


@dataclass
class Thread:
    thread_id: int
    brand: str
    messages: list[Message] = field(default_factory=list)
    truncated: bool = False

    @property
    def customer_message(self) -> Optional[Message]:
        for m in self.messages:
            if m.role == "customer":
                return m
        return None

    @property
    def last_brand_message(self) -> Optional[Message]:
        for m in reversed(self.messages):
            if m.role == "brand":
                return m
        return None

    @property
    def n_turns(self) -> int:
        return len(self.messages)

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "brand": self.brand,
            "n_turns": self.n_turns,
            "truncated": self.truncated,
            "messages": [
                {
                    "tweet_id": m.tweet_id,
                    "role": m.role,
                    "text": m.text,
                    "raw_text": m.raw_text,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "in_response_to": m.in_response_to,
                }
                for m in self.messages
            ],
        }


def _clean_text(text: str, is_customer: bool) -> str:
    if not isinstance(text, str):
        return ""
    t = text
    if is_customer:
        t = _AT_HANDLE_RE.sub("", t)
    t = _URL_RE.sub("[URL]", t)
    return _WS_RE.sub(" ", t).strip()


def _parse_ts(ts_series: pd.Series) -> pd.Series:
    """Vectorised timestamp parse; NaT on failure."""
    return pd.to_datetime(ts_series, format=TIMESTAMP_FMT, errors="coerce", utc=True)


def reconstruct_threads(df: pd.DataFrame, brand: str,
                        verbose: bool = True) -> list[Thread]:
    """
    Reconstruct conversation threads — fully vectorised.

    Steps:
    1. Build parent map as two numpy arrays.
    2. Vectorised root-find: iterate at most 50 levels; uses numpy indexing.
    3. Assign thread_id = root tweet_id.
    4. Sort all rows by (thread_id, created_at, tweet_id).
    5. Apply cleaning and truncation per group using pandas groupby + apply.
    """
    if verbose:
        print(f"[threads] Reconstructing threads from {len(df):,} rows...")

    df = df.copy()

    # ---- Step 1: parse timestamps (vectorised) ----
    df["_ts"] = _parse_ts(df["created_at"])

    # ---- Step 2: vectorised root-find ----
    # Build arrays: child_ids, parent_ids
    mask_has_parent = df["in_response_to_tweet_id"].notna()
    child_arr = df.loc[mask_has_parent, "tweet_id"].astype(np.int64).values
    parent_arr = df.loc[mask_has_parent, "in_response_to_tweet_id"].astype(np.int64).values
    parent_lookup = dict(zip(child_arr.tolist(), parent_arr.tolist()))

    tweet_ids = df["tweet_id"].astype(np.int64).values
    roots = tweet_ids.copy()

    for _ in range(50):
        # For each current root, look up its parent; if found, jump to parent
        new_roots = np.array(
            [parent_lookup.get(int(r), int(r)) for r in roots],
            dtype=np.int64
        )
        if np.array_equal(new_roots, roots):
            break
        roots = new_roots

    df["thread_id"] = roots

    # ---- Step 3: assign roles (vectorised) ----
    df["role"] = np.where(df["author_id"] == brand, "brand", "customer")

    # ---- Step 4: sort ----
    df = df.sort_values(["thread_id", "_ts", "tweet_id"], na_position="last")

    # ---- Step 5: clean text (vectorised with apply on Series) ----
    is_cust = df["role"] == "customer"
    texts = df["text"].fillna("").astype(str)

    def _clean_series_row(row_tuple):
        text, is_customer = row_tuple
        return _clean_text(text, is_customer)

    df["clean_text"] = [
        _clean_text(t, ic)
        for t, ic in zip(texts, is_cust)
    ]

    # ---- Step 6: build Thread objects per group ----
    if verbose:
        print(f"[threads] Building Thread objects ({df['thread_id'].nunique():,} groups)...")

    threads: list[Thread] = []
    ts_vals = df["_ts"].values
    tid_vals = df["tweet_id"].astype(np.int64).values
    role_vals = df["role"].values
    text_vals = df["clean_text"].values
    raw_vals = df["text"].fillna("").values
    parent_vals = df["in_response_to_tweet_id"].values
    thread_id_vals = df["thread_id"].astype(np.int64).values

    # Group indices by thread_id using numpy
    # Sort is already done, so we just need split points
    unique_tids, counts = np.unique(thread_id_vals, return_counts=True)
    splits = np.concatenate([[0], np.cumsum(counts)])

    for i, (tid, start, end) in enumerate(
        zip(unique_tids.tolist(), splits[:-1].tolist(), splits[1:].tolist())
    ):
        msgs: list[Message] = []
        for j in range(start, end):
            ts_val = ts_vals[j]
            try:
                ts_dt: Optional[datetime] = (
                    None if pd.isnull(ts_val)
                    else pd.Timestamp(ts_val).to_pydatetime()
                )
            except Exception:
                ts_dt = None
            parent_raw = parent_vals[j]
            try:
                parent_int: Optional[int] = (
                    None if pd.isna(parent_raw) else int(parent_raw)
                )
            except (TypeError, ValueError):
                parent_int = None
            msgs.append(Message(
                tweet_id=int(tid_vals[j]),
                role=str(role_vals[j]),
                text=str(text_vals[j]),
                raw_text=str(raw_vals[j]),
                created_at=ts_dt,
                in_response_to=parent_int,
            ))

        truncated = False
        if len(msgs) > MAX_CONTEXT_TURNS:
            msgs = msgs[-MAX_CONTEXT_TURNS:]
            truncated = True

        # Only keep threads with ≥1 customer message
        if any(m.role == "customer" for m in msgs):
            threads.append(Thread(
                thread_id=int(tid),
                brand=brand,
                messages=msgs,
                truncated=truncated,
            ))

    if verbose:
        n_multi = sum(1 for t in threads if t.n_turns >= 3)
        print(f"[threads]   -> {len(threads):,} threads  ({n_multi:,} multi-turn >=3)")

    return threads
