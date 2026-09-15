"""
Analytics & Failure Explorer.

Bootstrap behaviour: shows real numbers as soon as ≥ 10 examples are labelled,
clearly flagged as "n=X, preliminary". Does not require the full 150-250.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config.settings import settings
from src.ai.feedback import load_feedback, feedback_stats
from src.ai.classifier import KeywordBaseline, ESCALATION_SENSITIVE
from src.data.golden import load_golden_set, INTENTS
from src.ui.components import (
    inject_global_css, card, empty_state, preliminary_badge,
)

inject_global_css()

st.title("Analytics & Failure Explorer")
st.caption("Intent distribution, escalation rates, failure modes, and human override log")

# ── Load data ──────────────────────────────────────────────────────────────
golden  = load_golden_set()
n_gold  = len(golden)
PARTIAL = n_gold > 0 and n_gold < 150  # bootstrap / preliminary mode
FULL    = n_gold >= 150

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Intent Distribution",
    "⚡ Escalation Analysis",
    "❌ Failure Explorer",
    "👤 Human Override Log",
])

# ══════════════════════════════════════════════════════════════════════════
# Tab 1 — Intent Distribution
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    if n_gold == 0:
        empty_state(
            title="No labels yet",
            description=(
                "Label examples in Label Mode to see the intent distribution "
                "of your golden evaluation set."
            ),
            preview_items=[
                "Bar chart: count per intent",
                "Class-imbalance warning when one intent dominates (>30%)",
                "Table: intent, count, % of total",
            ],
            cta_label="Go to Label Mode →",
            cta_page="pages/label_mode.py",
        )
    else:
        with card("Golden Set — Intent Distribution"):
            if PARTIAL:
                preliminary_badge(n_gold)
            counts = Counter(
                ex["gold_intent"] for ex in golden if ex.get("gold_intent")
            )
            df_counts = pd.DataFrame(
                [{"Intent": k, "Count": v,
                  "Share": f"{v / n_gold * 100:.1f}%"}
                 for k, v in sorted(counts.items(), key=lambda x: -x[1])]
            )
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.barh(
                    df_counts["Intent"],
                    df_counts["Count"],
                    color="#2563EB",
                    edgecolor="none",
                )
                ax.set_xlabel("Count")
                ax.set_title(
                    f"Intent Distribution  (n={n_gold}"
                    + (" — preliminary" if PARTIAL else "") + ")"
                )
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
            with col2:
                st.dataframe(df_counts, use_container_width=True, hide_index=True)

            most_common_pct = max(counts.values()) / n_gold * 100
            if most_common_pct > 30:
                st.warning(
                    f"⚠️ Class imbalance: most common intent is "
                    f"**{most_common_pct:.1f}%** of examples. "
                    "Headline accuracy may be inflated. "
                    "See *What is misleading?* in the evaluation report.",
                    icon="⚠️",
                )

# ══════════════════════════════════════════════════════════════════════════
# Tab 2 — Escalation Analysis
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    if n_gold == 0:
        empty_state(
            title="No labels yet",
            description="Label examples to see escalation-rate analysis.",
            preview_items=[
                "Counts: total labelled / should escalate / auto-handle",
                "Per-intent escalation rate table",
                "⚠️ False-auto-handle rate highlighted prominently",
            ],
            cta_label="Go to Label Mode →",
            cta_page="pages/label_mode.py",
        )
    else:
        with card("Escalation Summary"):
            if PARTIAL:
                preliminary_badge(n_gold)
            n_esc  = sum(1 for ex in golden if ex.get("gold_escalate"))
            n_auto = n_gold - n_esc

            c1, c2, c3 = st.columns(3)
            c1.metric("Total labelled", n_gold)
            c2.metric("Should Escalate", n_esc,
                       f"{n_esc / n_gold * 100:.1f}%" if n_gold else "—")
            c3.metric("Auto-Handle", n_auto,
                       f"{n_auto / n_gold * 100:.1f}%" if n_gold else "—")

        with card("Per-intent escalation rate"):
            rows = []
            for intent in INTENTS:
                subset = [ex for ex in golden if ex.get("gold_intent") == intent]
                if subset:
                    rate = sum(1 for e in subset if e.get("gold_escalate")) / len(subset)
                    rows.append({"Intent": intent, "Count": len(subset),
                                 "Escalation rate": f"{rate:.1%}"})
            if rows:
                st.dataframe(
                    pd.DataFrame(rows).sort_values("Escalation rate", ascending=False),
                    use_container_width=True, hide_index=True,
                )

        # Run pipeline on golden subset to get system predictions
        if n_gold >= 5:
            with card("⚠️ False-Auto-Handle Rate (dangerous direction)"):
                st.caption(
                    "False-auto-handle = case labelled 'should escalate' but "
                    "the system decided AUTO_HANDLE. This is the harmful error direction."
                )
                kb = KeywordBaseline()
                false_auto = 0
                total_should_esc = 0
                for ex in golden:
                    if ex.get("gold_escalate"):
                        total_should_esc += 1
                        pred = kb.predict(ex["customer_message"])
                        if pred.intent not in ESCALATION_SENSITIVE and pred.confidence >= 0.55:
                            false_auto += 1
                rate_fa = false_auto / total_should_esc if total_should_esc else 0
                col_fa, col_info = st.columns([1, 3])
                col_fa.metric(
                    "False-auto-handle rate",
                    f"{rate_fa:.1%}",
                    delta=None,
                    help="Lower is safer",
                )
                col_info.markdown(
                    "_Computed against keyword-baseline classifier. "
                    "Run `python run.py eval` for full-system numbers._"
                )

# ══════════════════════════════════════════════════════════════════════════
# Tab 3 — Failure Explorer
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    # Always show the static failure analysis report if it exists
    fa_path = settings.reports_dir / "failure_analysis.md"
    if fa_path.exists():
        with card("Top-5 Failure Modes (from static analysis)"):
            st.markdown(fa_path.read_text(encoding="utf-8"))
        st.divider()

    if n_gold == 0:
        empty_state(
            title="No labels yet — interactive filter unavailable",
            description=(
                "Once you have labelled examples you can filter them here "
                "by intent and escalation decision."
            ),
            preview_items=[
                "Filter by intent and escalation decision",
                "Expandable example cards with message, gold label, notes",
                "Up to 20 examples shown per filter combination",
            ],
            cta_label="Go to Label Mode →",
            cta_page="pages/label_mode.py",
        )
    else:
        with card("Interactive Filter"):
            if PARTIAL:
                preliminary_badge(n_gold)
            col_fi, col_fe = st.columns(2)
            with col_fi:
                filter_intent = st.selectbox("Filter by intent", ["All"] + INTENTS)
            with col_fe:
                filter_esc = st.selectbox(
                    "Filter by escalation", ["All", "Escalate", "Auto-handle"]
                )

            filtered = golden
            if filter_intent != "All":
                filtered = [e for e in filtered if e.get("gold_intent") == filter_intent]
            if filter_esc == "Escalate":
                filtered = [e for e in filtered if e.get("gold_escalate")]
            elif filter_esc == "Auto-handle":
                filtered = [e for e in filtered if not e.get("gold_escalate")]

            st.caption(f"Showing {min(len(filtered), 20)} of {len(filtered)} matching examples")
            for ex in filtered[:20]:
                label = ex.get("gold_intent", "unlabelled")
                preview = ex["customer_message"][:80]
                with st.expander(f"[{label}]  {preview}…"):
                    st.markdown(f"**Message:** {ex['customer_message']}")
                    c_int, c_esc = st.columns(2)
                    c_int.markdown(f"**Gold intent:** `{label}`")
                    c_esc.markdown(
                        f"**Escalate:** {'Yes' if ex.get('gold_escalate') else 'No'}"
                    )
                    if ex.get("gold_notes"):
                        st.caption(f"Notes: {ex['gold_notes']}")
                    if ex.get("is_ambiguous"):
                        st.caption("⚡ Marked ambiguous by annotator")

# ══════════════════════════════════════════════════════════════════════════
# Tab 4 — Human Override Log
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    stats        = feedback_stats()
    total_fb     = stats["total"]

    with card("Feedback Statistics"):
        st.caption(
            "All feedback is **logged for future use**. "
            "The system does not currently retrain from this data."
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total",            total_fb)
        c2.metric("Accepted",         stats["accepted"])
        c3.metric("Rejected",         stats["rejected"])
        c4.metric("Decision overrides", stats["overrides"])
        c5.metric("Intent corrections", stats["corrections"])

    if total_fb == 0:
        empty_state(
            title="No feedback logged yet",
            description=(
                "Use the Inbox to triage customer messages. "
                "Each time you accept, reject, or override a reply "
                "the action is recorded here."
            ),
            preview_items=[
                "Table: timestamp, message, predicted vs corrected intent",
                "Predicted vs overridden escalation decision",
                "Reply accepted / rejected flag",
            ],
            cta_label="Go to Inbox →",
            cta_page="pages/inbox.py",
        )
    else:
        rows = load_feedback(limit=200)
        df_fb = pd.DataFrame([
            {
                "Time":               r["created_at"][:19],
                "Message":            (r["customer_message"] or "")[:55] + "…",
                "Predicted intent":   r["predicted_intent"],
                "Corrected":          r["corrected_intent"] or "—",
                "Decision":           r["predicted_decision"],
                "Overridden":         r["overridden_decision"] or "—",
                "Reply":              (
                    "✅ Accepted" if r["reply_accepted"] == 1
                    else "❌ Rejected" if r["reply_accepted"] == 0
                    else "—"
                ),
            }
            for r in rows
        ])
        with card("Override Log"):
            st.dataframe(df_fb, use_container_width=True, hide_index=True)
