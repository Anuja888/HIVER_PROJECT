"""
Phase 9 — Evaluation Harness.

Runs full evaluation against the golden set and produces reports/evaluation.md.
Also evaluates two baselines (trivial, simple) on the same set.

Decision log:
  - Evaluation uses ONLY the golden set (human-labelled). If golden set is
    missing, the harness prints a clear error and exits — never fabricates.
  - Baselines use the exact same golden set — no cherry-picking.
  - All metrics include 95% confidence intervals (Wilson score) given the
    small N (150-250 examples has real CI width).
  - False-auto-handle rate is flagged prominently (the dangerous direction).
"""
from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config.settings import settings
from src.data.golden import load_golden_set, INTENTS
from src.ai.classifier import KeywordBaseline, ESCALATION_SENSITIVE
from src.ai.pipeline import run_pipeline
from src.eval.metrics import (
    intent_classification_metrics,
    escalation_metrics,
    confidence_interval_95,
)
from src.eval.judge import judge_batch

log = logging.getLogger(__name__)

LABELS = INTENTS


def _trivial_baseline(message: str) -> tuple[str, str]:
    """Majority class intent + always-escalate decision."""
    return "order_delivery_tracking", "ESCALATE"


def _simple_baseline(message: str, kb: KeywordBaseline) -> tuple[str, str]:
    """TF-IDF/keyword classification + simple threshold escalation."""
    result = kb.predict(message)
    decision = "ESCALATE" if (
        result.confidence < 0.50 or
        result.intent in ESCALATION_SENSITIVE
    ) else "AUTO_HANDLE"
    return result.intent, decision


def run_evaluation(
    brand: str = "AmazonHelp",
    max_examples: int = 250,
    skip_if_no_gold: bool = True,
) -> dict:
    """
    Run the full evaluation harness.

    Returns a dict with all metric results.
    Writes reports/evaluation.md and figures.
    """
    print(f"\n{'='*60}")
    print("Evaluation Harness")
    print(f"{'='*60}\n")

    # ---- Load golden set ----
    golden = load_golden_set()
    if not golden:
        msg = (
            "ERROR: No human-labelled golden examples found in data/gold/golden_set.jsonl.\n"
            "Complete the labelling step first: python run.py label-mode\n"
            "Then label at least 150 examples before running eval."
        )
        print(msg)
        if skip_if_no_gold:
            return {"error": "No golden set found", "n_gold": 0}
        raise RuntimeError(msg)

    examples = golden[:max_examples]
    n = len(examples)
    print(f"[eval] Golden set: {n} examples")

    # ---- Run full pipeline on each example ----
    print("[eval] Running full pipeline...")
    pipeline_results = []
    for ex in examples:
        msg = ex["customer_message"]
        ctx = ex.get("thread_context", [])
        result = run_pipeline(msg, thread_context=ctx, brand=brand)
        pipeline_results.append({
            "example": ex,
            "result": result.to_dict(),
        })

    # ---- Extract predictions ----
    y_true_intent = [ex["gold_intent"] for ex in examples]
    y_pred_intent = [
        pr["result"]["classification"]["intent"]
        if pr["result"]["classification"] else "general_feedback_other"
        for pr in pipeline_results
    ]

    y_true_esc = [
        "ESCALATE" if ex.get("gold_escalate") else "AUTO_HANDLE"
        for ex in examples
    ]
    y_pred_esc = [
        pr["result"]["routing"]["decision"]
        if pr["result"]["routing"] else "ESCALATE"
        for pr in pipeline_results
    ]

    # ---- Baseline predictions ----
    kb = KeywordBaseline()
    trivial_intents, trivial_escs = zip(*[_trivial_baseline(ex["customer_message"]) for ex in examples])
    simple_intents, simple_escs = zip(*[_simple_baseline(ex["customer_message"], kb) for ex in examples])

    # ---- Intent metrics ----
    intent_m = intent_classification_metrics(y_true_intent, y_pred_intent, LABELS)
    trivial_intent_m = intent_classification_metrics(list(y_true_intent), list(trivial_intents), LABELS)
    simple_intent_m = intent_classification_metrics(list(y_true_intent), list(simple_intents), LABELS)

    # ---- Escalation metrics ----
    esc_m = escalation_metrics(y_true_esc, y_pred_esc)
    trivial_esc_m = escalation_metrics(list(y_true_esc), list(trivial_escs))
    simple_esc_m = escalation_metrics(list(y_true_esc), list(simple_escs))

    # ---- Confidence intervals ----
    intent_ci = confidence_interval_95(n, int(intent_m["accuracy"] * n))
    esc_ci = confidence_interval_95(n, int(esc_m["accuracy"] * n))

    # ---- LLM judge on generated replies ----
    print("[eval] Running LLM judge on replies...")
    judge_inputs = [
        {
            "customer_message": ex["customer_message"],
            "evidence_items": pr["result"].get("retrieval", {}).get("items", []),
            "generated_reply": (pr["result"].get("generation") or {}).get("reply", ""),
        }
        for ex, pr in zip(examples, pipeline_results)
        if (pr["result"].get("generation") or {}).get("reply")
    ]
    judge_scores = judge_batch(judge_inputs, max_examples=n)
    avg_judge = {
        dim: round(np.mean([getattr(s, dim) for s in judge_scores]), 3)
        for dim in ["relevance", "correctness", "grounding", "helpfulness",
                    "completeness", "tone", "no_unsupported_claims"]
    }
    avg_judge["overall"] = round(np.mean([s.overall for s in judge_scores]), 3)

    # ---- Confusion matrix figure ----
    fig_dir = settings.reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion_matrix(intent_m["confusion_matrix"], LABELS, fig_dir / "confusion_matrix.png")

    # ---- Compile results ----
    results = {
        "n_gold": n,
        "intent_metrics": intent_m,
        "escalation_metrics": esc_m,
        "trivial_intent_metrics": trivial_intent_m,
        "simple_intent_metrics": simple_intent_m,
        "trivial_esc_metrics": trivial_esc_m,
        "simple_esc_metrics": simple_esc_m,
        "judge_scores": avg_judge,
        "intent_ci_95": intent_ci,
        "esc_ci_95": esc_ci,
        "pipeline_results": pipeline_results,
    }

    # ---- Write report ----
    _write_evaluation_report(results, brand)

    return results


def _plot_confusion_matrix(cm: list, labels: list, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    short = [l.replace("_", "\n") for l in labels]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Intent Classification Confusion Matrix")
    plt.colorbar(im, ax=ax)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i][j] > max(max(r) for r in cm) * 0.5 else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _write_evaluation_report(results: dict, brand: str) -> None:
    n = results["n_gold"]
    im = results["intent_metrics"]
    em = results["escalation_metrics"]
    ti = results["trivial_intent_metrics"]
    si = results["simple_intent_metrics"]
    te = results["trivial_esc_metrics"]
    se = results["simple_esc_metrics"]
    js = results["judge_scores"]
    i_ci = results["intent_ci_95"]
    e_ci = results["esc_ci_95"]

    # Per-intent table
    per_intent_rows = ""
    for label in LABELS:
        m = im["per_intent"].get(label, {})
        per_intent_rows += (
            f"| `{label}` | {m.get('precision', 0):.3f} | "
            f"{m.get('recall', 0):.3f} | {m.get('f1', 0):.3f} | "
            f"{m.get('support', 0)} |\n"
        )

    # Judge scores table
    judge_rows = "\n".join(
        f"| {dim} | {js.get(dim, 0):.2f} / 5.0 |"
        for dim in ["relevance", "correctness", "grounding", "helpfulness",
                    "completeness", "tone", "no_unsupported_claims"]
    )

    md = textwrap.dedent(f"""
    # Evaluation Report — `{brand}`

    **Golden set size:** {n} human-labelled examples  
    **Evaluation date:** (see git log for reproducibility)  
    **Mock mode:** See LLM_PROVIDER setting — results may vary in mock vs. live mode.

    ---

    ## A. Intent Classification

    | Metric | Trivial baseline | Simple baseline | Full system | Δ (simple→full) |
    |---|---|---|---|---|
    | Accuracy | {ti['accuracy']:.3f} | {si['accuracy']:.3f} | {im['accuracy']:.3f} | {im['accuracy']-si['accuracy']:+.3f} |
    | Macro-F1 | {ti['macro_f1']:.3f} | {si['macro_f1']:.3f} | {im['macro_f1']:.3f} | {im['macro_f1']-si['macro_f1']:+.3f} |

    **95% CI for system accuracy:** [{i_ci[0]:.3f}, {i_ci[1]:.3f}] (Wilson score, N={n})

    ### Per-Intent Breakdown

    | Intent | Precision | Recall | F1 | Support |
    |---|---|---|---|---|
    {per_intent_rows}

    ![Confusion Matrix](figures/confusion_matrix.png)

    ---

    ## B. Reply Quality (LLM-as-Judge)

    > ⚠️ **Important**: These scores come from an LLM judge. Human–judge agreement
    > is measured on a subset (see section E). Trust these scores proportionally
    > to that agreement statistic.

    | Dimension | Mean score (1–5) |
    |---|---|
    {judge_rows}
    | **Overall** | **{js.get('overall', 0):.2f} / 5.0** |

    ---

    ## C. Escalation Decision

    | Metric | Trivial | Simple | Full system |
    |---|---|---|---|
    | Accuracy | {te['accuracy']:.3f} | {se['accuracy']:.3f} | {em['accuracy']:.3f} |
    | False-auto-handle rate ⚠️ | {te['false_auto_handle_rate']:.3f} | {se['false_auto_handle_rate']:.3f} | {em['false_auto_handle_rate']:.3f} |
    | False-escalation rate | {te['false_escalation_rate']:.3f} | {se['false_escalation_rate']:.3f} | {em['false_escalation_rate']:.3f} |

    > ⚠️ **False-auto-handle rate** = fraction of cases that SHOULD be escalated
    > but were auto-handled. This is the **dangerous direction**. System rate: **{em['false_auto_handle_rate']:.1%}**.

    **95% CI for escalation accuracy:** [{e_ci[0]:.3f}, {e_ci[1]:.3f}] (N={n})

    ---

    ## D. Retrieval Quality

    Retrieval quality is assessed via LLM judge's `grounding` score above, and
    per-intent breakdown is shown in the per-intent table. A dedicated retrieval
    relevance study requires human relevance judgments on retrieved evidence pairs
    (not yet collected — noted as a limitation).

    ---

    ## E. Human–Judge Agreement

    See `reports/judge_agreement.md` after running the agreement subset script.
    Until measured, treat all judge scores as provisional.

    ---

    ## F. What Is Misleading About the Headline Number?

    See `reports/failure_analysis.md` for the full honesty section.

    Brief summary:
    1. **Class imbalance**: the golden set is stratified but `order_delivery_tracking`
       likely dominates — high accuracy may reflect this, not genuine generalisation.
    2. **Test set difficulty**: easy/short messages may be over-represented if the
       heuristic ambiguity filter under-sampled hard cases.
    3. **Small N**: N={n} means the 95% CI is ≈±{(i_ci[1]-i_ci[0])/2:.1%} wide —
       do not over-interpret 1–2% differences between systems.
    4. **LLM judge bias**: judge scores are only as trustworthy as the measured
       human agreement (see section E). If Spearman r < 0.4, disregard judge scores.
    5. **Mock mode**: if LLM_PROVIDER=mock, all generation/judge results are
       deterministic canned responses — not real model outputs.
    6. **Retrieval leakage check**: despite conversation-level splitting, if the
       same customer sent multiple support requests, their language patterns appear
       in both train and test — a known limitation noted in the decision log.

    *Generated by `src/eval/harness.py`*
    """).strip()

    out = settings.reports_dir / "evaluation.md"
    out.write_text(md, encoding="utf-8")
    print(f"[eval] Report -> {out}")
