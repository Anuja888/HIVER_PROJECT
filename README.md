# Hiver AI Support Agent — [Brand Name, e.g. AmazonHelp]

An AI support agent that classifies incoming customer messages, drafts a reply grounded in how the brand has historically resolved similar issues, and decides whether the message can be auto-handled or should be escalated to a human — with a stated, legible reason.

Built for the Hiver SDE Intern take-home assignment. **The proof matters more than the system** — see [`reports/evaluation.md`](reports/evaluation.md) and the honesty sections below before judging this on the demo alone.

---

## Quickstart (reproduces headline results in under 15 minutes)

```bash
git clone <repo-url>
cd <repo>
cp .env.example .env        # optional — leave blank to run in mock mode, zero API cost
make setup                  # installs deps, downloads/caches the data subsample
make eval                   # runs the full evaluation harness against the golden set
make run                    # launches the Streamlit app at http://localhost:8501
```

- `make setup` downloads and caches a fixed, reproducible subsample of the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset — not the full ~3M rows.
- `make eval` regenerates every number in this README and in `reports/evaluation.md` from scratch, deterministically (fixed seeds throughout).
- Runs in **mock mode** by default (no API key needed) for zero-cost reproduction. Set `LLM_API_KEY` in `.env` to use a real model; see [Cost](#cost) below.

Total wall-clock time on a laptop CPU: **[FILL IN — measured, e.g. ~9 min]**.

---

## What this system does

```
Customer message + thread context
      → intent classification
      → historical evidence retrieval (this brand's past resolutions only)
      → resolution pattern extraction
      → grounded reply generation
      → grounding / hallucination check
      → auto-handle vs. escalate decision, with a stated reason
      → structured output + evidence trace + logging
```

Every reply is generated **conditioned on retrieved historical evidence**, not general LLM knowledge. The system never asserts historical patterns as current company policy — see [Grounding](#grounding--evidence) below.

---

## Why this brand

Brand selection was data-driven, not arbitrary. `scripts/select_brand.py` scores every brand account in the dataset on conversation volume, multi-turn thread availability, topic diversity, noise rate, and resolution richness (full methodology and weights in [`reports/brand_selection.md`](reports/brand_selection.md)).

**Chosen brand:** `[FILL IN, e.g. AmazonHelp]` — score breakdown and runner-ups in the linked report.

---

## Intent taxonomy

`[N]` intents, derived from clustering real customer messages for this brand — not Banking77's 77 categories. Full definitions, examples, and ambiguity notes in [`docs/intent_taxonomy.md`](docs/intent_taxonomy.md).

| Intent | Description | Escalation-sensitive? |
|---|---|---|
| `[FILL IN]` | `[FILL IN]` | Yes / No |
| ... | ... | ... |

---

## Golden evaluation set

- **[N, 150–250] examples**, hand-labelled by a human (not model-generated) — stored separately at `data/gold/golden_set.jsonl`, each record tagged `labelled_by: human`.
- Sampling: stratified across intents, with a deliberate slice of ambiguous and noisy examples. Full methodology in [`docs/golden_set_methodology.md`](docs/golden_set_methodology.md).
- Inter-annotator agreement on a [~20–30 example] subset: `[FILL IN, e.g. Cohen's κ = 0.71]`. If a second labelling pass wasn't feasible, this is stated explicitly rather than omitted.

---

## Results

### Intent classification

| Metric | Trivial baseline | Simple baseline | Full system |
|---|---|---|---|
| Accuracy | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` |
| Macro F1 | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` |

Per-intent breakdown and confusion matrix: [`reports/evaluation.md`](reports/evaluation.md).

### Reply quality (LLM-as-judge, 1–5 scale per rubric dimension)

| Dimension | Trivial | Simple | Full system |
|---|---|---|---|
| Relevance | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` |
| Grounding | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` |
| Helpfulness | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` |
| Hallucination rate | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` |

**Judge–human agreement:** `[FILL IN, e.g. weighted κ = 0.58 on a 40-example human-rated subset]` — see [Judge reliability](#judge-reliability) for what this number does and doesn't license.

### Escalation decision

| Metric | Trivial | Simple | Full system |
|---|---|---|---|
| Decision accuracy | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` |
| False-auto-handle rate | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` |
| False-escalation rate | `[FILL IN]` | `[FILL IN]` | `[FILL IN]` |

False-auto-handle is the dangerous direction — flagged prominently in [`reports/evaluation.md`](reports/evaluation.md), not buried in an average.

---

## What is misleading about my headline number?

The honest caveats behind the top-line accuracy/quality figures above — read before quoting them:

- `[FILL IN — e.g. class imbalance: N% of the golden set is one intent, so a naive classifier already scores high]`
- `[FILL IN — e.g. small-sample noise: with only N examples, the 95% CI on accuracy is ±X points]`
- `[FILL IN — e.g. judge–human agreement is only moderate, so judge-based reply scores should be read as directional, not precise]`
- `[FILL IN — e.g. retrieval quality varies significantly by intent; average similarity hides intents where evidence is consistently thin]`
- `[FILL IN — e.g. historical data reflects past policy, not necessarily current policy — the system is honest about this in its output, but the eval set can't fully verify current-policy correctness]`

Full writeup: [`reports/evaluation.md`](reports/evaluation.md), section "Headline number caveats."

---

## Baselines

Two required baselines, evaluated on the exact same golden set as the full system — no cherry-picking:

- **Trivial**: majority-class intent classifier, one generic fallback reply, fixed escalation policy.
- **Simple**: TF-IDF/keyword intent classification, nearest-neighbor historical-reply retrieval (no generation), threshold-based escalation rule.

Implementation: `src/baselines/`.

---

## Grounding & evidence

Every generated reply carries a structured evidence trace: which historical tweets were retrieved, why, what resolution pattern was extracted, and whether evidence was judged sufficient. A grounding validator flags any claim in the generated reply not traceable to retrieved evidence before it's shown as auto-handle-eligible.

The system is deliberately worded to distinguish *"historical pattern suggests..."* from *"this is current policy"* — it never asserts the latter from historical data alone.

Confidence and system-confidence scores shown in the UI are **uncalibrated model signals, not probabilities**, unless a calibration check is reported here: `[FILL IN — "not calibrated" or link to calibration report]`.

---

## Judge reliability

The LLM-as-judge rubric (relevance, correctness, grounding, helpfulness, completeness, tone, hallucination) is defined in `src/eval/judge_rubric.py`. Its agreement with human ratings on a `[N]`-example subset is `[FILL IN statistic]`. This number is reported as measured, including if it's mediocre — see the headline-number caveats above for what that implies about trusting judge scores elsewhere in this report.

---

## Failure analysis

Top 5 real failure modes, pulled from actual golden-set evaluation runs (not invented), each with input, expected vs. actual behavior, root-cause hypothesis, impact, and possible fix: [`reports/failure_analysis.md`](reports/failure_analysis.md).

---

## Decision log

10–15 non-obvious calls made during this build, with one-line rationale each (brand-scoring weights, conversation-level split, truncation length, threshold-selection method, taxonomy size, judge rubric choices, etc.): [`reports/decision_log.md`](reports/decision_log.md).

---

## Architecture

```
src/
  data/           # ingestion, cleaning, thread reconstruction, splitting, caching
  classifier/     # rule baseline + LLM few-shot intent classifier
  retriever/      # embedding index over historical brand resolutions
  generator/      # grounded reply generation
  validator/      # grounding / hallucination checks
  router/         # escalation decision logic
  baselines/      # trivial + simple baselines
  eval/           # evaluation harness, LLM-judge, metrics
app/              # Streamlit UI: Inbox, Evaluation, Analytics, Label Mode
scripts/          # brand selection, one-off analysis scripts
data/gold/        # human-labelled golden set (never model-populated)
reports/          # generated reports — brand selection, data stats, evaluation,
                   # failure analysis, decision log
docs/             # taxonomy, golden-set methodology, architecture notes
```

Full diagram and module responsibilities: [`docs/architecture.md`](docs/architecture.md).

---

## Running the app

```bash
make run
```

Opens the Streamlit app with four pages:

- **Inbox** — paste a customer message, see intent, confidence, evidence trace, drafted reply, escalation decision and reason, and accept/edit/reject the reply.
- **Evaluation Dashboard** — full metrics from the last `make eval` run.
- **Analytics & Failure Explorer** — intent distribution, escalation analysis, failure explorer, human override log, filterable by intent/decision/failure type.
- **Label Mode** — the tool used to build the golden set.

---

## Human feedback loop

Accept / edit / reject / correct-intent / override-escalation actions in the Inbox are persisted to a structured feedback log (`data/feedback.db`). **This is logged for future use — the system does not currently retrain or update from this data.**

---

## Testing

```bash
make test        # unit + integration tests
make eval-test   # evaluation-harness self-test on fixture examples
```

---

## Cost

Full `make eval` run against a real LLM API: approximately `[FILL IN, e.g. $0.30–$0.80]` depending on provider/model. Running with `LLM_API_KEY` unset uses deterministic mock mode at zero cost — recommended for grading/reproduction.

---

## Citations

Anything borrowed (code, prompts, libraries beyond standard package usage) is cited here and inline in code comments:

- `[FILL IN as used, e.g. "escalation-threshold sweep approach adapted from <source>"]`

---

## What I'd do with one more week

`[FILL IN — e.g. calibrate confidence scores, expand golden set to 400+ examples, add a second human labeller for full inter-annotator agreement, test a second brand for generalization, add retraining loop from feedback log]`

---

## Known limitations

- Golden set size (150–250 examples) limits statistical precision on rare intents — see headline-number caveats.
- Historical resolutions reflect past brand behavior, not verified current policy.
- LLM-as-judge agreement with humans is `[FILL IN]` — treat judge-based scores as directional.
- Mock mode is deterministic but not a substitute for live-model evaluation numbers reported above.
