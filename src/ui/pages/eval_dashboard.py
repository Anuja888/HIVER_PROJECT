"""
Evaluation Dashboard.

Bootstrap behaviour:
  >= 10 labels  -> run inline preliminary metrics and show them with a
                    "n=X, preliminary" badge.
  >= 150 labels -> full run-eval report preferred; inline metrics still
                    shown until report exists.
  report exists -> render the full Markdown report + figures.

Never a bare st.warning + st.stop().
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config.settings import settings
from src.data.golden import load_golden_set, INTENTS
from src.ai.classifier import KeywordBaseline
from src.ui.components import (
    inject_global_css, card, empty_state, preliminary_badge,
)

inject_global_css()

st.title("Evaluation Dashboard")
st.caption("Intent classification, reply quality, and escalation metrics vs human-labelled golden set")

# ── Data availability ───────────────────────────────────────────────────────
golden       = load_golden_set()
n_gold       = len(golden)
report_path  = settings.reports_dir / "evaluation.md"
report_exists = report_path.exists()
cm_path      = settings.reports_dir / "figures" / "confusion_matrix.png"
dist_path    = settings.reports_dir / "figures" / "data_distributions.png"

# ══════════════════════════════════════════════════════════════════════════
# Case 1: No labels at all — full empty state
# ══════════════════════════════════════════════════════════════════════════
if n_gold == 0 and not report_exists:
    empty_state(
        title="Evaluation not yet available",
        description=(
            "Label at least 10 examples in Label Mode to see preliminary "
            "metrics, or 150+ for the full evaluation report."
        ),
        preview_items=[
            "Intent classification — accuracy, macro-F1, per-intent P/R/F1, confusion matrix",
            "Reply quality — LLM-as-judge scores across 7 dimensions (1–5 scale)",
            "Escalation — accuracy, false-auto-handle rate, false-escalation rate",
            "Two baselines (trivial & simple) compared on the same golden set",
            "'What is misleading about my headline number?' — honest analysis",
        ],
        cta_label="Go to Label Mode →",
        cta_page="pages/label_mode.py",
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# Case 2: Full report exists — render it
# ══════════════════════════════════════════════════════════════════════════
if report_exists:
    st.success(
        f"Full evaluation report available (generated from {n_gold} golden examples).",
        icon="✅",
    )
    with card("Full Evaluation Report"):
        st.markdown(report_path.read_text(encoding="utf-8"))

    if cm_path.exists():
        st.image(str(cm_path),
                 caption="Intent Classification Confusion Matrix",
                 use_container_width=True)
    if dist_path.exists():
        with card("Data Split Distributions"):
            st.image(str(dist_path), use_container_width=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# Case 3: ≥ 10 labels but no full report — bootstrap inline metrics
# ══════════════════════════════════════════════════════════════════════════
if n_gold >= 10:
    st.info(
        f"**Preliminary metrics** from {n_gold} labelled examples. "
        "Run `python run.py eval` after labelling 150+ examples for the full report.",
        icon="ℹ️",
    )
    preliminary_badge(n_gold)

    # ── A. Intent classification (keyword baseline as proxy) ───────────────
    with card("A · Intent Classification  (keyword baseline)"):
        preliminary_badge(n_gold)
        st.caption(
            "Using the keyword-baseline classifier as a fast proxy. "
            "Run full eval for LLM-classifier numbers."
        )
        kb = KeywordBaseline()
        y_true, y_pred = [], []
        for ex in golden:
            if ex.get("gold_intent"):
                y_true.append(ex["gold_intent"])
                y_pred.append(kb.predict(ex["customer_message"]).intent)

        if y_true:
            from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
            acc  = accuracy_score(y_true, y_pred)
            mf1  = f1_score(y_true, y_pred, average="macro",
                            labels=INTENTS, zero_division=0)

            c1, c2 = st.columns(2)
            c1.metric("Accuracy",  f"{acc:.1%}")
            c2.metric("Macro-F1",  f"{mf1:.3f}")

            # Per-intent table
            from sklearn.metrics import classification_report
            rpt = classification_report(
                y_true, y_pred, labels=INTENTS,
                output_dict=True, zero_division=0,
            )
            rows = []
            for lbl in INTENTS:
                m = rpt.get(lbl, {})
                rows.append({
                    "Intent":    lbl,
                    "Precision": f"{m.get('precision', 0):.2f}",
                    "Recall":    f"{m.get('recall',    0):.2f}",
                    "F1":        f"{m.get('f1-score',  0):.2f}",
                    "Support":   int(m.get("support",  0)),
                })
            st.dataframe(
                pd.DataFrame(rows), use_container_width=True, hide_index=True
            )

            # Confusion matrix heatmap
            cm = confusion_matrix(y_true, y_pred, labels=INTENTS)
            fig, ax = plt.subplots(figsize=(9, 7))
            im = ax.imshow(cm, cmap="Blues")
            short = [l.replace("_", "\n") for l in INTENTS]
            ax.set_xticks(range(len(INTENTS))); ax.set_yticks(range(len(INTENTS)))
            ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(short, fontsize=7)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"Confusion Matrix  (n={n_gold}, preliminary)")
            plt.colorbar(im, ax=ax)
            for i in range(len(INTENTS)):
                for j in range(len(INTENTS)):
                    mx = max(max(r) for r in cm) or 1
                    ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                            fontsize=7,
                            color="white" if cm[i][j] > mx * 0.5 else "black")
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

    # ── B. Escalation (gold labels) ────────────────────────────────────────
    with card("B · Escalation Decision  (gold labels vs keyword baseline)"):
        preliminary_badge(n_gold)
        has_esc_labels = any(ex.get("gold_escalate") is not None for ex in golden)
        if not has_esc_labels:
            st.caption("No escalation labels found in golden set yet.")
        else:
            from src.ai.classifier import ESCALATION_SENSITIVE
            y_true_e, y_pred_e = [], []
            for ex in golden:
                if ex.get("gold_escalate") is not None:
                    gold_d = "ESCALATE" if ex["gold_escalate"] else "AUTO_HANDLE"
                    pred   = kb.predict(ex["customer_message"])
                    pred_d = "ESCALATE" if pred.intent in ESCALATION_SENSITIVE else "AUTO_HANDLE"
                    y_true_e.append(gold_d)
                    y_pred_e.append(pred_d)

            from src.eval.metrics import escalation_metrics
            em = escalation_metrics(y_true_e, y_pred_e)
            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy",            f"{em['accuracy']:.1%}")
            c2.metric("False-auto-handle ⚠️", f"{em['false_auto_handle_rate']:.1%}",
                      help="Dangerous direction — should escalate but auto-handled")
            c3.metric("False-escalation",    f"{em['false_escalation_rate']:.1%}")

    # ── C. Confidence intervals ────────────────────────────────────────────
    with card("C · Confidence Intervals"):
        preliminary_badge(n_gold)
        from src.eval.metrics import confidence_interval_95
        n_correct_intent = sum(
            1 for t, p in zip(y_true, y_pred) if t == p
        ) if y_true else 0
        ci = confidence_interval_95(len(y_true), n_correct_intent) if y_true else (0, 0)
        st.markdown(
            f"**95% CI for intent accuracy** (Wilson score, n={len(y_true)}): "
            f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        )
        ci_width = (ci[1] - ci[0]) / 2
        if ci_width > 0.07:
            st.caption(
                f"CI half-width is ±{ci_width:.1%} — too wide to distinguish "
                "small differences between classifiers. Label more examples."
            )

    st.divider()
    st.caption("👆 All numbers above are preliminary. Run `python run.py eval` for full results.")

elif n_gold > 0:
    # < 10 labels — show minimal progress rather than empty state
    st.info(
        f"**{n_gold} example(s) labelled so far.** "
        "Label at least 10 to see preliminary metrics.",
        icon="ℹ️",
    )
    st.progress(n_gold / 10, text=f"{n_gold}/10 minimum for preliminary metrics")
    if st.button("Continue labelling →", type="primary"):
        st.switch_page("pages/label_mode.py")
