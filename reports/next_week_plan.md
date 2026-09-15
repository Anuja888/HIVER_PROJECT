# Next-Week Plan

If given one more week, here is what I would prioritise, in order of expected
impact on the grading criteria ("the proof is worth more than the system"):

---

## Priority 1 — Complete the Human Evaluation (Day 1–2)

The most important gap right now is that `data/gold/golden_set.jsonl` is empty.
Every metric in `reports/evaluation.md` is either absent or based on mock responses.

**Actions:**
- Label all 200 candidate examples in the Label Mode UI (~2–3 hours)
- Do a second labelling pass on 25 examples (blind, 24h gap) and run
  `scripts/compute_iaa.py` to get a real Cohen's kappa
- Re-run `python run.py eval` with `LLM_PROVIDER=openai` to get real quality scores
- Update `reports/evaluation.md` with actual numbers — replace every placeholder

**Expected outcome:** Real accuracy/F1/escalation metrics with 95% CIs, a measured
human-judge agreement statistic, and an honest failure analysis populated with
real failed examples (not the hypothesised ones in the current report).

---

## Priority 2 — Threshold Tuning on the Validation Split (Day 2–3)

The escalation router and classifier currently use the initial config.yaml thresholds
(0.55 intent confidence, 0.55 retrieval similarity). These are reasoned guesses.

**Actions:**
- Implement the threshold sweep in `scripts/run_eval.py`: grid-search over
  `intent_confidence_threshold ∈ [0.3, 0.4, 0.5, 0.6, 0.7]` and
  `retrieval_similarity_threshold ∈ [0.25, 0.35, 0.45, 0.55]` on the **val split**
- Select thresholds by optimising for: F1 on escalation AND minimising
  false-auto-handle rate (the dangerous direction)
- Write the sweep results to `reports/threshold_sweep.md` with a plot
- Update `config.yaml` with the validated thresholds

**Expected outcome:** Escalation precision/recall that is principled, not arbitrary.
The decision log can be updated to reference the actual sweep result.

---

## Priority 3 — Calibration Check on Confidence Scores (Day 3)

Currently all confidence scores are labeled "uncalibrated" — which is the honest
default, but we could actually check this.

**Actions:**
- Bin the classifier's confidence scores into 10 buckets
- For each bucket, compute the actual accuracy of predictions in that bucket
- Plot a reliability diagram (expected accuracy = diagonal line; actual accuracy = bars)
- Compute Expected Calibration Error (ECE)
- If well-calibrated (ECE < 0.05), update UI copy to say "approximately calibrated";
  if not, the current "uncalibrated" label is vindicated and we document why

**Expected outcome:** A real data point on whether confidence scores are meaningful,
rather than a blanket disclaimer.

---

## Priority 4 — Improve Retrieval for Rare Intents (Day 3–4)

Failure mode #3 (thin evidence for rare intents) is the most actionable system
improvement. Currently the index has equal representation by training thread, so
rare intents like `account_security_compromise` have very few indexed examples.

**Actions:**
- After human labelling is done, build an **intent-stratified retrieval sub-index**:
  sample equally from each intent bucket rather than randomly from all threads
- Measure per-intent retrieval top-1 similarity before and after
- Compare per-intent reply quality scores (judge) between the two approaches
- Document whether stratification actually helps (measure it, don't assume)

**Expected outcome:** Better grounding for rare but high-stakes intents.

---

## Priority 5 — End-to-End Timing & Cost Instrumentation (Day 4)

The README states ~$0.65–$1.00 for a full eval run but this is an estimate.

**Actions:**
- Add token counting to the LLM client wrapper (already captures `tokens_used`
  in `ChatResponse` — just need to aggregate across a full eval run)
- Write actual measured cost to `reports/eval_cost.txt` after each eval run
- Add a timing breakdown: per-step latency (classification, retrieval, generation,
  validation, routing) averaged over the golden set
- Make `python run.py eval --dry-run` print an estimated cost before calling any API

**Expected outcome:** A real dollar number in the README, not an estimate, and
clear visibility into where latency is spent.

---

## Priority 6 — Polish & Edge Cases (Day 4–5)

**Actions:**
- Handle non-English messages gracefully (detect language, route to ESCALATE with
  reason "Message language not supported by automated system")
- Add a "no-reply" path: when `top_similarity < 0.25`, skip generation entirely
  and return a pre-written escalation message instead of a generic vague reply
- Add structured logging to a JSON file (one line per pipeline call with all
  intermediate signals) — enables offline analysis of production traffic
- Add a conftest.py that skips integration tests if the data cache doesn't exist
  (so CI/CD works on a clean clone without the full dataset)

---

## What I Would NOT Do Next Week

- **Add more ML complexity** (fine-tuning, RAG with dense passage retrieval,
  chain-of-thought prompting improvements) — the evaluation rigor is more valuable
  than a fancier model, and the rubric explicitly says so
- **Build a React frontend** — Streamlit already covers all the required features;
  React would only add reproducibility risk
- **Expand the intent taxonomy** — 9 intents is the right size for this task;
  more intents without more labelled data would hurt classification quality

---

*The most important principle from the assignment: "The proof is worth more than
the system." A week of honest measurement beats a week of architectural improvements.*
