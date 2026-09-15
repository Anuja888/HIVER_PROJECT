#!/usr/bin/env python
"""Phase 2 — Run the full data pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.pipeline import run_pipeline

if __name__ == "__main__":
    result = run_pipeline()
    print(f"\nBrand: {result['brand']}")
    print(f"Train threads: {len(result['train']):,}")
    print(f"Val threads:   {len(result['val']):,}")
    print(f"Test threads:  {len(result['test']):,}")
