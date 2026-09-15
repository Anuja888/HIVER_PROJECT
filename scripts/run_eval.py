#!/usr/bin/env python
"""Phase 9 — Run evaluation harness."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.harness import run_evaluation

if __name__ == "__main__":
    results = run_evaluation()
    n = results.get("n_gold", 0)
    if n > 0:
        im = results["intent_metrics"]
        em = results["escalation_metrics"]
        print("\nSummary:")
        print(f"  Intent accuracy:     {im['accuracy']:.3f}")
        print(f"  Intent macro-F1:     {im['macro_f1']:.3f}")
        print(f"  Escalation accuracy: {em['accuracy']:.3f}")
        print(f"  False-auto-handle:   {em['false_auto_handle_rate']:.3f}")
        print("\nFull report: reports/evaluation.md")
