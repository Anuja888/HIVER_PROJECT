# Golden Set Methodology

## Sampling Strategy

- **Source**: Test split only (never train or val — prevents leakage)
- **Target size**: 200 examples (within the 150–250 required range)
- **Seed**: 42 (reproducible)
- **Brand**: AmazonHelp

## Stratification

The 200 examples are sampled as follows:

- **80% stratified by intent** (~22 examples per intent across 9 buckets):
  - Uses a heuristic keyword-match to pre-assign examples to intent buckets for
    stratification purposes only — NOT as labels. Human labels may differ.
  - Ensures all 9 intents have representation even if rare in the test split.

- **20% deliberate ambiguous slice** (~40 examples):
  - Examples flagged as ambiguous by the heuristic (short message OR multi-topic keywords).
  - Ensures the golden set covers hard cases, not just easy ones.

## Labelling Process

1. `python run.py label-mode` → Opens Streamlit Label Mode UI
2. For each example: read message + thread context → select intent → flag escalation → save
3. Labels are appended to `data/gold/golden_set.jsonl` with `"labelled_by": "human"`

## Inter-Annotator Agreement

The assignment requires measuring Cohen's kappa on a 20–30 example subset.

**Target procedure**: Re-label 25 randomly selected examples after a 24-hour gap
(same annotator, fresh session) or have a second person label the same 25 examples.
Compute Cohen's kappa between the two label sets.

**If kappa ≥ 0.6**: Good agreement — taxonomy is well-defined.  
**If kappa 0.4–0.6**: Moderate agreement — flag ambiguous intents and clarify definitions.  
**If kappa < 0.4**: Poor agreement — taxonomy likely has too much overlap; consolidate intents.

This step must be done manually. The system does not auto-compute kappa unless you
run the agreement script: `python scripts/compute_iaa.py` (see scripts/).

## What Is NOT Done (documented honestly)

- The system NEVER auto-fills gold_intent from model predictions.
- The system NEVER writes to golden_set.jsonl without `labelled_by='human'`.
- The golden set is NOT used for any training or fine-tuning step.
- If kappa cannot be measured (single labeller, single session), this is documented
  explicitly in the evaluation report rather than claiming human agreement without evidence.
