"""
Pytest configuration.

Skips integration tests that require cached data if the cache doesn't exist,
so the test suite works on a clean clone (unit tests still run).
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Force mock mode for all tests — no API cost
os.environ.setdefault("LLM_PROVIDER", "mock")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if data cache is missing."""
    from src.config.settings import settings
    cache_missing = not (settings.cache_dir / "AmazonHelp_train.jsonl").exists()
    index_missing = not (settings.cache_dir / "retrieval_index_AmazonHelp.pkl").exists()

    for item in items:
        if "integration" in str(item.fspath):
            if cache_missing:
                item.add_marker(pytest.mark.skip(
                    reason="Data cache missing — run 'python run.py data-pipeline' first"
                ))
            elif index_missing:
                item.add_marker(pytest.mark.skip(
                    reason="Retrieval index missing — run the pipeline once to build it"
                ))
