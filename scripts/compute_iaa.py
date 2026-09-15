#!/usr/bin/env python
"""
Inter-Annotator Agreement (IAA) computation.

Purpose: Measure consistency of the golden set labels.

Procedure:
  1. Take a random 25-example subset of the labelled golden set.
  2. Re-label those 25 examples (second pass — same person a day later,
     or a second annotator) and save to data/gold/iaa_second_pass.jsonl.
  3. Run this script to compute Cohen's kappa between the two label sets.

Usage:
  python scripts/compute_iaa.py

If a second-pass file doesn't exist, this script prints instructions for
creating one and exits cleanly — it never fabricates agreement numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.config.settings import settings
from src.data.golden import load_golden_set

IAA_SAMPLE_SIZE = 25
IAA_SECOND_PASS = settings.gold_dir / "iaa_second_pass.jsonl"
IAA_SUBSET_FILE = settings.gold_dir / "iaa_subset.jsonl"


def sample_iaa_subset() -> list[dict]:
    """Sample 25 examples for second-pass labelling."""
    golden = load_golden_set()
    if not golden:
        print("No labelled examples found. Complete labelling first.")
        return []
    rng = np.random.default_rng(settings.sample_seed + 99)  # different seed
    n = min(IAA_SAMPLE_SIZE, len(golden))
    idx = rng.choice(len(golden), size=n, replace=False)
    return [golden[i] for i in sorted(idx)]


def compute_kappa(labels_a: list[str], labels_b: list[str]) -> dict:
    """Compute Cohen's kappa between two label lists."""
    from sklearn.metrics import cohen_kappa_score
    assert len(labels_a) == len(labels_b), "Label lists must be same length"
    kappa = cohen_kappa_score(labels_a, labels_b)
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    return {
        "kappa": round(kappa, 4),
        "percent_agreement": round(agree / len(labels_a) * 100, 1),
        "n_pairs": len(labels_a),
        "disagreements": [
            {"first": a, "second": b}
            for a, b in zip(labels_a, labels_b) if a != b
        ],
        "interpretation": (
            "Substantial agreement" if kappa >= 0.6 else
            "Moderate agreement — review ambiguous intents" if kappa >= 0.4 else
            "Fair agreement — taxonomy may need refinement" if kappa >= 0.2 else
            "Slight agreement — significant labelling inconsistency detected"
        ),
    }


def main() -> None:
    print(f"\n{'='*60}")
    print("Inter-Annotator Agreement Computation")
    print(f"{'='*60}\n")

    # Step 1: Create/show IAA subset for second-pass labelling
    if not IAA_SUBSET_FILE.exists():
        subset = sample_iaa_subset()
        if not subset:
            sys.exit(1)
        settings.gold_dir.mkdir(parents=True, exist_ok=True)
        with open(IAA_SUBSET_FILE, "w", encoding="utf-8") as f:
            for ex in subset:
                # Strip existing labels so second annotator is blind
                blind = {
                    "thread_id": ex["thread_id"],
                    "customer_message": ex["customer_message"],
                    "thread_context": ex.get("thread_context", []),
                }
                f.write(json.dumps(blind) + "\n")
        print(f"✓ IAA subset ({len(subset)} examples) written to: {IAA_SUBSET_FILE}")
        print()
        print("NEXT STEPS:")
        print("  1. Wait at least 24 hours (or ask a second person)")
        print("  2. Open the Label Mode UI and label the examples in:")
        print(f"     {IAA_SUBSET_FILE}")
        print(f"  3. Save second-pass labels to: {IAA_SECOND_PASS}")
        print("     Each record needs: thread_id, gold_intent, labelled_by='human_2'")
        print("  4. Re-run this script")
        return

    # Step 2: Load both label sets
    golden = load_golden_set()
    golden_by_id = {ex["thread_id"]: ex["gold_intent"] for ex in golden}

    if not IAA_SECOND_PASS.exists():
        print(f"Second-pass labels not found: {IAA_SECOND_PASS}")
        print("Complete the second labelling pass first (see instructions above).")
        return

    with open(IAA_SECOND_PASS, encoding="utf-8") as f:
        second_pass = [json.loads(line) for line in f]

    # Align labels by thread_id
    labels_a, labels_b = [], []
    for rec in second_pass:
        tid = rec["thread_id"]
        if tid in golden_by_id and rec.get("gold_intent"):
            labels_a.append(golden_by_id[tid])
            labels_b.append(rec["gold_intent"])

    if len(labels_a) < 5:
        print(f"Only {len(labels_a)} matching pairs found. Need at least 5.")
        return

    # Step 3: Compute kappa
    result = compute_kappa(labels_a, labels_b)

    print(f"IAA Results (N={result['n_pairs']} pairs):")
    print(f"  Cohen's kappa:       {result['kappa']:.4f}")
    print(f"  Percent agreement:   {result['percent_agreement']:.1f}%")
    print(f"  Interpretation:      {result['interpretation']}")
    print()
    if result["disagreements"]:
        print("Disagreements (first → second):")
        for d in result["disagreements"][:10]:
            print(f"  {d['first']} → {d['second']}")

    # Write result
    report_path = settings.reports_dir / "iaa_report.md"
    md = f"""# Inter-Annotator Agreement Report

**N pairs:** {result['n_pairs']}
**Cohen's kappa:** {result['kappa']:.4f}
**Percent agreement:** {result['percent_agreement']:.1f}%
**Interpretation:** {result['interpretation']}

## Disagreements
{chr(10).join(f"- `{d['first']}` → `{d['second']}`" for d in result['disagreements'])}

## Implications
{"Good agreement — taxonomy is well-defined, golden labels are trustworthy." if result['kappa'] >= 0.6 else "Moderate agreement — flag ambiguous intents and review taxonomy definitions before reporting headline numbers." if result['kappa'] >= 0.4 else "⚠️ Low agreement — headline accuracy numbers may be overstated due to inconsistent labels. Consider consolidating ambiguous intents."}

*Generated by scripts/compute_iaa.py*
"""
    report_path.write_text(md, encoding="utf-8")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
