"""
Phase 9 — Evaluation Metrics.

All metrics are computed from real data against real gold labels.
Never fabricate numbers.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)


def intent_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict:
    """Compute accuracy, macro-F1, per-intent P/R/F1."""
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro",
                        labels=labels, zero_division=0)

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_intent": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": int(report[label]["support"]),
            }
            for label in labels if label in report
        },
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def escalation_metrics(
    y_true: list[str],  # "AUTO_HANDLE" or "ESCALATE"
    y_pred: list[str],
) -> dict:
    """
    Compute escalation decision metrics.

    false_auto_handle_rate: rate at which truly-escalation-worthy cases were
    auto-handled (the dangerous direction — flag prominently).
    """
    # Map to binary: ESCALATE=1, AUTO_HANDLE=0
    pos = "ESCALATE"
    y_true_bin = [1 if y == pos else 0 for y in y_true]
    y_pred_bin = [1 if y == pos else 0 for y in y_pred]

    accuracy = accuracy_score(y_true_bin, y_pred_bin)

    tp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 0)

    # False-auto-handle = SHOULD escalate but was auto-handled (FN for ESCALATE)
    n_true_escalate = sum(y_true_bin)
    false_auto_handle_rate = fn / max(n_true_escalate, 1)
    false_escalation_rate = fp / max(sum(1 - b for b in y_true_bin), 1)

    kappa = cohen_kappa_score(y_true_bin, y_pred_bin) if len(set(y_true_bin)) > 1 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "kappa": round(kappa, 4),
        "false_auto_handle_rate": round(false_auto_handle_rate, 4),
        "false_escalation_rate": round(false_escalation_rate, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_true_escalate": n_true_escalate,
        "n_true_auto": len(y_true) - n_true_escalate,
    }


def confidence_interval_95(n: int, k: int) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    z = 1.96
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(max(0, centre - margin), 4), round(min(1, centre + margin), 4))


def judge_agreement_stats(
    human_scores: list[float],
    llm_scores: list[float],
) -> dict:
    """
    Compute agreement between human raters and LLM judge.
    Uses Spearman correlation (appropriate for ordinal 1-5 scores).
    """
    from scipy.stats import spearmanr
    if len(human_scores) < 3:
        return {"note": "Too few samples for agreement statistics"}

    corr, pvalue = spearmanr(human_scores, llm_scores)
    # Round to discrete for kappa
    h_int = [round(s) for s in human_scores]
    l_int = [round(s) for s in llm_scores]
    try:
        kappa = cohen_kappa_score(h_int, l_int, weights="linear")
    except Exception:
        kappa = float("nan")

    return {
        "spearman_r": round(corr, 4),
        "spearman_p": round(pvalue, 4),
        "weighted_kappa": round(kappa, 4) if not math.isnan(kappa) else None,
        "n_pairs": len(human_scores),
        "mean_human": round(np.mean(human_scores), 3),
        "mean_llm": round(np.mean(llm_scores), 3),
        "interpretation": (
            "Strong agreement" if abs(corr) >= 0.7 else
            "Moderate agreement" if abs(corr) >= 0.4 else
            "Weak agreement — treat LLM judge scores with caution"
        ),
    }
