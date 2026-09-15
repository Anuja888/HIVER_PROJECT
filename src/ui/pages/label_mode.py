"""
Label Mode page — human golden-set labelling tool.

Phase 4: Streamlit labelling UI.

Shows:
- Thread context + customer message
- Intent selector (with definitions)
- Ambiguity flag
- Escalate/auto-handle judgment
- Optional free-text notes
- Progress tracking

IMPORTANT: Only human-confirmed labels are written to golden_set.jsonl.
The model NEVER auto-fills labels and claims they are human-authored.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import streamlit as st

from src.config.settings import settings
from src.data.golden import (
    load_candidate_pool, load_golden_set,
    save_golden_label, sample_candidate_pool, save_candidate_pool,
    INTENTS,
)

st.title("Label Mode — Golden Set")
st.caption("Human labelling tool for the evaluation golden set (150–250 examples)")

# ---- Sidebar info ----
with st.sidebar:
    st.subheader("📋 Labelling Guide")
    st.markdown("""
**Goal:** Label 150–200 examples accurately.

**Per example:**
1. Read the customer message + thread context
2. Select the best-matching intent
3. Mark if you're unsure (ambiguous)
4. Decide: should this be escalated to a human agent?
5. Add notes if helpful
6. Click **Save Label**

**Keyboard tip:** Use Tab to move between fields quickly.

**Quality > speed.** If you're unsure, mark it ambiguous rather than guessing.
    """)
    st.markdown("---")
    st.subheader("Intent Quick Reference")
    for intent in INTENTS:
        esc = "🔴" if intent in {
            "product_return_refund", "account_access_login",
            "account_security_compromise", "prime_subscription_billing"
        } else "🟢"
        st.caption(f"{esc} `{intent.replace('_', ' ')}`")

# ---- Initialize candidate pool ----
pool_path = settings.gold_dir / "candidate_pool.jsonl"

if not pool_path.exists():
    st.info("No candidate pool found. Generating from test split...")
    brand_path = settings.reports_dir / "brand_selection" / "chosen_brand.txt"
    brand = brand_path.read_text(encoding="utf-8").strip() if brand_path.exists() else "AmazonHelp"
    try:
        with st.spinner("Sampling candidate pool..."):
            pool = sample_candidate_pool(brand)
            save_candidate_pool(pool, brand)
        st.success(f"Generated {len(pool)} candidate examples. Start labelling below!")
        st.rerun()
    except Exception as e:
        st.error(f"Error generating pool: {e}")
        st.stop()

# Load pool + existing labels
pool = load_candidate_pool()
labelled = load_golden_set()
labelled_ids = {ex["thread_id"] for ex in labelled}
remaining = [ex for ex in pool if ex["thread_id"] not in labelled_ids]

# Progress
col1, col2, col3 = st.columns(3)
col1.metric("Total pool", len(pool))
col2.metric("Labelled", len(labelled))
col3.metric("Remaining", len(remaining))

target = 200
progress_val = min(len(labelled) / target, 1.0)
st.progress(progress_val, text=f"{len(labelled)}/{target} examples labelled ({progress_val*100:.0f}%)")

if len(labelled) >= target:
    st.success(f"✅ Target of {target} labels reached! You can continue for higher coverage.")
    st.balloons()

if not remaining:
    st.info("All candidate examples have been labelled!")
    st.stop()

# ---- Navigation ----
if "label_idx" not in st.session_state:
    st.session_state["label_idx"] = 0

idx = st.session_state["label_idx"]
if idx >= len(remaining):
    st.session_state["label_idx"] = 0
    idx = 0

col_prev, col_counter, col_next = st.columns([1, 3, 1])
with col_prev:
    if st.button("← Prev", use_container_width=True) and idx > 0:
        st.session_state["label_idx"] -= 1
        st.rerun()
with col_counter:
    st.markdown(f"<p style='text-align:center'>Example {idx+1} of {len(remaining)} remaining</p>",
                unsafe_allow_html=True)
with col_next:
    if st.button("Next →", use_container_width=True) and idx < len(remaining) - 1:
        st.session_state["label_idx"] += 1
        st.rerun()

# ---- Current example ----
ex = remaining[idx]
st.markdown("---")

# Thread context
if ex.get("thread_context") and len(ex["thread_context"]) > 1:
    with st.expander("📜 Thread Context (click to expand)", expanded=False):
        for msg in ex["thread_context"][:-1]:  # exclude last (= current customer message)
            role_icon = "👤" if msg["role"] == "customer" else "🏢"
            st.markdown(f"**{role_icon} {msg['role'].title()}:** {msg.get('text', '')[:300]}")

# Customer message (main)
with st.container(border=True):
    st.subheader("Customer Message")
    st.markdown(f"**{ex['customer_message']}**")
    hint = ex.get("guessed_intent_heuristic")
    if hint:
        st.caption(f"💡 Heuristic hint (for reference only, not a label): `{hint}`")

# ---- Labelling form ----
with st.form(f"label_form_{idx}"):
    col_intent, col_flags = st.columns([2, 1])

    with col_intent:
        # Pre-select heuristic hint as default
        hint = ex.get("guessed_intent_heuristic", "general_feedback_other")
        default_idx = INTENTS.index(hint) if hint in INTENTS else 0
        gold_intent = st.selectbox(
            "Intent *",
            INTENTS,
            index=default_idx,
            help="Select the best-matching intent for this customer message",
        )

    with col_flags:
        is_ambiguous = st.checkbox(
            "Mark as ambiguous",
            value=ex.get("is_ambiguous_heuristic", False),
            help="Check if you're unsure or if multiple intents apply equally",
        )
        gold_escalate = st.checkbox(
            "Should escalate to human agent",
            value=gold_intent in {
                "product_return_refund", "account_access_login",
                "account_security_compromise", "prime_subscription_billing",
            },
            help="Check if this case should be handled by a human, not auto-resolved",
        )

    gold_notes = st.text_input(
        "Notes (optional)",
        placeholder="e.g. 'borderline between tracking and return', 'non-English words mixed in'",
    )

    submitted = st.form_submit_button("💾 Save Label & Next", type="primary", use_container_width=True)

if submitted:
    record = dict(ex)
    record["gold_intent"] = gold_intent
    record["gold_escalate"] = gold_escalate
    record["gold_notes"] = gold_notes or None
    record["is_ambiguous"] = is_ambiguous
    record["labelled_by"] = "human"

    save_golden_label(record)
    st.success(f"✓ Saved: `{gold_intent}` | Escalate: {'Yes' if gold_escalate else 'No'}")

    # Advance to next
    if idx < len(remaining) - 1:
        st.session_state["label_idx"] = idx + 1
    st.rerun()

# ---- Export labelled set info ----
st.markdown("---")
st.caption(
    f"Labels stored in golden set  |  "
    f"each record marked with `labelled_by: 'human'`"
)
