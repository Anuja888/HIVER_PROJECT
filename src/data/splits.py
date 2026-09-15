"""
Phase 2 — Train / Val / Test Splitting & Sampling.

Decision log:
- Split is CONVERSATION-level (not message-level). A full thread lives
  entirely in one split. This prevents leakage where the model sees a
  customer query in training and the same query in the test set in a
  different message.
- Stratification: threads are stratified by n_turns bucket (1, 2, 3-5,
  6+) to ensure the splits have similar length distributions. This is a
  weak proxy for intent diversity at split time (before intents are
  labelled).
- Sizes: 70% train / 10% val / 20% test (from config.yaml).
- Random seed: settings.sample_seed (42) for full reproducibility.
- Only threads with at least ONE customer message are kept (threads with
  only brand messages are uninformative for intent classification).
- Subsample: the downstream pipeline uses a bounded subsample of the
  training split for the retrieval index (RETRIEVAL_TRAIN_SIZE) to
  keep indexing time <2 minutes on a laptop.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split

from src.config.settings import settings
from src.data.threads import Thread

log = logging.getLogger(__name__)

# Max threads to index for retrieval (avoids multi-minute embedding runs)
RETRIEVAL_TRAIN_SIZE = 3_000


def _turns_bucket(n: int) -> str:
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if n <= 5:
        return "3-5"
    return "6+"


def split_threads(
    threads: list[Thread],
    train_frac: float | None = None,
    val_frac: float | None = None,
    seed: int | None = None,
) -> tuple[list[Thread], list[Thread], list[Thread]]:
    """
    Split threads into (train, val, test) using conversation-level splits.
    Returns (train, val, test) lists.
    """
    train_frac = train_frac or settings.train_fraction
    val_frac = val_frac or settings.val_fraction
    seed = seed or settings.sample_seed

    # Keep only threads with at least one customer message
    threads = [t for t in threads if t.customer_message is not None]

    # Stratify by turns bucket
    buckets = [_turns_bucket(t.n_turns) for t in threads]

    # First split off test
    test_frac = 1.0 - train_frac - val_frac
    idx = list(range(len(threads)))
    try:
        idx_trainval, idx_test = train_test_split(
            idx, test_size=test_frac, stratify=buckets, random_state=seed
        )
        buckets_trainval = [buckets[i] for i in idx_trainval]
        val_of_trainval = val_frac / (train_frac + val_frac)
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=val_of_trainval,
            stratify=buckets_trainval, random_state=seed,
        )
    except ValueError:
        # Fallback if stratification fails (rare classes)
        idx_trainval, idx_test = train_test_split(
            idx, test_size=test_frac, random_state=seed
        )
        val_of_trainval = val_frac / (train_frac + val_frac)
        idx_train, idx_val = train_test_split(
            idx_trainval, test_size=val_of_trainval, random_state=seed
        )

    train = [threads[i] for i in idx_train]
    val   = [threads[i] for i in idx_val]
    test  = [threads[i] for i in idx_test]

    return train, val, test


def save_splits(
    train: list[Thread],
    val: list[Thread],
    test: list[Thread],
    brand: str,
) -> dict[str, Path]:
    """Serialize split threads to JSONL files in the cache directory."""
    paths: dict[str, Path] = {}
    for split_name, split_threads in [("train", train), ("val", val), ("test", test)]:
        path = settings.cache_dir / f"{brand}_{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for t in split_threads:
                f.write(json.dumps(t.to_dict()) + "\n")
        paths[split_name] = path
        log.info("Saved %d threads to %s", len(split_threads), path)
    return paths


def load_splits(brand: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Load split JSONL files. Returns raw dicts (not Thread objects)."""
    result = []
    for split_name in ("train", "val", "test"):
        path = settings.cache_dir / f"{brand}_{split_name}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Split file not found: {path}. Run the data pipeline first."
            )
        with open(path, encoding="utf-8") as f:
            result.append([json.loads(line) for line in f])
    return tuple(result)  # type: ignore[return-value]
