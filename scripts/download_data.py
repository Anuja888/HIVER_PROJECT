#!/usr/bin/env python
"""
Phase 1 — Step 1: Download the raw Twitter Customer Support dataset from Kaggle.

Dataset: thoughtvector/customer-support-on-twitter (~3M tweets)
The full CSV is ~250MB. We download it once into data/raw/ and never commit it.

Usage:
    python scripts/download_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.config.settings import settings  # noqa: E402

KAGGLE_DATASET = "thoughtvector/customer-support-on-twitter"
TARGET = settings.raw_data_file


def download() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(f"[download] Already present: {TARGET} ({TARGET.stat().st_size} bytes)")
        return

    username = settings.kaggle_username
    key = settings.kaggle_key
    if not username or not key:
        print("[download] No Kaggle credentials in .env — attempting anonymous download...")
        # Kaggle requires auth even for public datasets; fall back to direct URL.
        # The dataset is mirrored at a Google Drive link on the Kaggle page.
        url = "https://drive.google.com/uc?export=download&id=1d56J-HAIBd9PY3pphzz2PrjGm4JFtEOx"
        print(f"[download] Trying mirror: {url}")
        import urllib.request
        urllib.request.urlretrieve(url, str(TARGET))
        print(f"[download] Saved to {TARGET}")
        return

    print(f"[download] Using Kaggle API as user {username}...")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(TARGET.parent), unzip=True, quiet=False)
    # Find the downloaded CSV
    candidates = list(TARGET.parent.glob("*.csv"))
    if candidates:
        actual = candidates[0]
        if actual != TARGET:
            actual.rename(TARGET)
        print(f"[download] Done: {TARGET} ({TARGET.stat().st_size} bytes)")
    else:
        print("[download] ERROR: No CSV found after download.")


if __name__ == "__main__":
    download()