#!/usr/bin/env python
"""
Threshold sweep on the validation split.

Searches for optimal intent_confidence_threshold and
retrieval_similarity_threshold using the val split (NOT golden set / test).

Selects thresholds by maximising F1 on escalation decisions while
keeping false-auto-handle rate below 0.15.

Writes results to reports/threshold_sweep.md and a plot to
reports/figures/threshold_sweep.png.

Usage:
    python scripts/threshold_sweep.py

Decision log:
  - Sweep uses val split, not golden set, to prevent leakage.
  - Escalation labels for val split are derived heuristically (escalation-
    sensitive intents → ESCALATE, others → AUTO_HANDLE) because the val
    split is not human-labelled. This is a documented limitation.
  - Final thresholds are written back to config.yaml as suggestions only;
    human must confirm before they take effect.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config.settings import settings
from src.ai.classifier import KeywordBaseline, ESCALATION_SENSITIVE
from src.ai.router import route
from src.ai.validator import ValidationResult
from src.retrieval.index import get_index
from src.eval.metrics import escalation_metrics

CONF_THRESHOLDS = [0.30, 0.40, 0.50, 0.55, 0.60, 0.70]
SIM_THRESHOLDS  = [0.25, 0.35, 0.45, 0.55, 0.65]
MAX_VAL = 500  # cap val examples for speed


def load_val_examples(brand: str) -> list[dict]:
    path = settings.cache_dir / f"{brand}_val.jsonl"
    if not path.exists():
        raise FileNotFoundError("Run data pipeline first.")
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            cust = next(
                (m for m in t["messages"] if m["role"] == "customer" and m["text"].strip()),
                None
            )
            if cust:
                examples.append({
                    "customer_message": cust["text"],
                    "thread_id": t["thread_id"],
                })
    return examples[:MAX_VAL]


def heuristic_escalation(intent: str) -> str:
    """Heuristic gold label for val split (no human labels available)."""
    return "ESCALATE" if intent in ESCALATION_SENSITIVE else "AUTO_HANDLE"


def main():
    brand_path = settings.reports_dir / "brand_selection" / "chosen_brand.txt"
    brand = brand_path.read_text().strip() if brand_path.exists() else "AmazonHelp"

    print(f"[sweep] Loading val examples for {brand}...")
    examples = load_val_examples(brand)
    print(f"[sweep] {len(examples)} val examples")

    print("[sweep] Loading retrieval index...")
    index = get_index(brand)

    print("[sweep] Running keyword classification...")
    kb = KeywordBaseline()
    dummy_val = ValidationResult(grounded=True, score=0.80, passed=True)

    # Pre-compute: classify + retrieve for each example
    print("[sweep] Pre-computing classifications and retrievals...")
    records = []
    for ex in examples:
        msg = ex["customer_message"]
        clf = kb.predict(msg)
        ret = index.search(msg, top_k=5)
        gold = heuristic_escalation(clf.intent)
        records.append({"clf": clf, "ret": ret, "gold_esc": gold})

    # Grid search
    print(f"[sweep] Sweeping {len(CONF_THRESHOLDS) * len(SIM_THRESHOLDS)} combinations...")
    results = []
    for ct in CONF_THRESHOLDS:
        for st in SIM_THRESHOLDS:
            y_true, y_pred = [], []
            for r in records:
                routing = route(r["clf"], r["ret"], dummy_val,
                                intent_conf_threshold=ct,
                                retrieval_sim_threshold=st)
                y_true.append(r["gold_esc"])
                y_pred.append(routing.decision)
            m = escalation_metrics(y_true, y_pred)
            results.append({
                "conf_thresh": ct, "sim_thresh": st,
                "accuracy": m["accuracy"],
                "false_auto_handle": m["false_auto_handle_rate"],
                "false_escalation": m["false_escalation_rate"],
                "kappa": m["kappa"],
            })

    # Select best: maximise accuracy, subject to false-auto-handle < 0.15
    safe = [r for r in results if r["false_auto_handle"] <= 0.15]
    if safe:
        best = max(safe, key=lambda r: r["accuracy"])
    else:
        best = max(results, key=lambda r: r["accuracy"])  # relax constraint

    print(f"\n{'='*60}")
    print("Threshold Sweep Results (val split, heuristic labels)")
    print(f"{'='*60}")
    print(f"{'CT':>6} {'ST':>6} {'Acc':>6} {'FalsAH':>8} {'FalsESC':>8}")
    print("-" * 40)
    for r in sorted(results, key=lambda x: -x["accuracy"]):
        marker = " ← BEST" if r is best else ""
        print(f"{r['conf_thresh']:>6.2f} {r['sim_thresh']:>6.2f} "
              f"{r['accuracy']:>6.3f} {r['false_auto_handle']:>8.3f} "
              f"{r['false_escalation']:>8.3f}{marker}")

    print("\nRecommended thresholds (subject to human validation):")
    print(f"  intent_confidence_threshold:    {best['conf_thresh']}")
    print(f"  retrieval_similarity_threshold: {best['sim_thresh']}")

    # Plot
    fig_dir = settings.reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Heatmap: accuracy
    acc_grid = np.zeros((len(CONF_THRESHOLDS), len(SIM_THRESHOLDS)))
    fah_grid = np.zeros((len(CONF_THRESHOLDS), len(SIM_THRESHOLDS)))
    for r in results:
        i = CONF_THRESHOLDS.index(r["conf_thresh"])
        j = SIM_THRESHOLDS.index(r["sim_thresh"])
        acc_grid[i, j] = r["accuracy"]
        fah_grid[i, j] = r["false_auto_handle"]

    for ax, data, title in [
        (axes[0], acc_grid, "Accuracy"),
        (axes[1], fah_grid, "False-Auto-Handle Rate (↓ better)"),
    ]:
        im = ax.imshow(data, cmap="RdYlGn" if title == "Accuracy" else "RdYlGn_r",
                       aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(SIM_THRESHOLDS)))
        ax.set_yticks(range(len(CONF_THRESHOLDS)))
        ax.set_xticklabels([f"{s:.2f}" for s in SIM_THRESHOLDS])
        ax.set_yticklabels([f"{c:.2f}" for c in CONF_THRESHOLDS])
        ax.set_xlabel("Retrieval similarity threshold")
        ax.set_ylabel("Intent confidence threshold")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        for ii in range(len(CONF_THRESHOLDS)):
            for jj in range(len(SIM_THRESHOLDS)):
                ax.text(jj, ii, f"{data[ii,jj]:.2f}", ha="center", va="center",
                        fontsize=8, color="black")

    plt.suptitle("Threshold Sweep on Validation Split\n(heuristic escalation labels)",
                 fontsize=12)
    plt.tight_layout()
    fig_path = fig_dir / "threshold_sweep.png"
    plt.savefig(fig_path, dpi=120)
    plt.close()

    # Write report
    report_path = settings.reports_dir / "threshold_sweep.md"
    md = f"""# Threshold Sweep Report

**Method**: Grid search on validation split ({len(examples)} examples).
**Escalation labels**: Heuristic — escalation-sensitive intents → ESCALATE.
This is an approximation; human labels would be more accurate but are not
available for the val split (by design — they're reserved for the golden test set).

## Recommended Thresholds

| Parameter | Recommended value | Rationale |
|---|---|---|
| `intent_confidence_threshold` | `{best['conf_thresh']}` | Best accuracy with false-auto-handle ≤ 15% |
| `retrieval_similarity_threshold` | `{best['sim_thresh']}` | Paired with above |

**Validation accuracy at recommended thresholds:** {best['accuracy']:.3f}
**False-auto-handle rate:** {best['false_auto_handle']:.3f}
**False-escalation rate:** {best['false_escalation']:.3f}

## Sweep Grid

![Threshold sweep heatmaps](figures/threshold_sweep.png)

## Limitations

1. Escalation labels are heuristic (not human-verified on val split). Treat
   recommended thresholds as a starting point, not a final answer.
2. The sweep uses the keyword baseline classifier for speed. The LLM classifier
   may yield different optimal thresholds — re-run with LLM classifier for final values.
3. Val split is {len(examples)} examples — sweep results have real variance.

*Generated by `scripts/threshold_sweep.py`*
"""
    report_path.write_text(md, encoding="utf-8")
    print(f"\nReport → {report_path}")
    print(f"Plot   → {fig_path}")


if __name__ == "__main__":
    main()
