"""
Phase 2 — Full Data Pipeline Orchestrator.

Runs all data steps in order:
  1. Load brand-connected rows (cached)
  2. Reconstruct threads
  3. Split (conversation-level)
  4. Cache splits as JSONL
  5. Write quality + stats reports

Cache invalidation: delete data/cache/*.parquet and data/cache/*.jsonl
to force a full re-run.
"""
from __future__ import annotations

import logging

from src.config.settings import settings
from src.data.loader import load_brand_threads
from src.data.splits import load_splits, save_splits, split_threads
from src.data.stats import write_data_quality_report, write_data_stats_report
from src.data.threads import reconstruct_threads

log = logging.getLogger(__name__)


def run_pipeline(brand: str | None = None, force: bool = False) -> dict:
    """
    Run the full data pipeline for the given brand.

    Args:
        brand: Brand handle (e.g. 'AmazonHelp'). Reads from
               reports/brand_selection/chosen_brand.txt if None.
        force: If True, re-run even if caches exist.

    Returns:
        dict with keys: brand, train, val, test (lists of thread dicts)
    """
    if brand is None:
        chosen_path = settings.reports_dir / "brand_selection" / "chosen_brand.txt"
        if chosen_path.exists():
            brand = chosen_path.read_text(encoding="utf-8").strip()
        else:
            brand = "AmazonHelp"
            log.warning("chosen_brand.txt not found; defaulting to %s", brand)

    print(f"\n{'='*60}")
    print(f"Data Pipeline — brand: {brand}")
    print(f"{'='*60}\n")

    # Check if splits already cached
    train_cache = settings.cache_dir / f"{brand}_train.jsonl"
    if train_cache.exists() and not force:
        print("[pipeline] Split caches exist — loading from disk (use force=True to rebuild)")
        train, val, test = load_splits(brand)
        print(f"[pipeline] Loaded: train={len(train):,}  val={len(val):,}  test={len(test):,}")
        return {"brand": brand, "train": train, "val": val, "test": test}

    # Step 1 — Load
    df = load_brand_threads(brand)

    # Step 2 — Reconstruct threads
    threads = reconstruct_threads(df, brand)

    # Step 3 — Quality report (on raw df)
    write_data_quality_report(df, brand)

    # Step 4 — Split
    train_threads, val_threads, test_threads = split_threads(threads)
    print(f"[pipeline] Split: train={len(train_threads):,}  "
          f"val={len(val_threads):,}  test={len(test_threads):,}")

    # Step 5 — Cache splits
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    save_splits(train_threads, val_threads, test_threads, brand)

    # Step 6 — Stats report
    write_data_stats_report(train_threads, val_threads, test_threads, brand)

    # Convert Thread objects to dicts for return
    train = [t.to_dict() for t in train_threads]
    val   = [t.to_dict() for t in val_threads]
    test  = [t.to_dict() for t in test_threads]

    print("\n[pipeline] Done")
    return {"brand": brand, "train": train, "val": val, "test": test}
