"""
Inbox — main support-agent interface.

Click path (warm, after first load):
  button click → run_pipeline() → display results
  All expensive resources (embedding model, index) are held in
  st.cache_resource via src/ui/cache.py — see Phase 1 fix.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import time
import streamlit as st

from src.ai.pipeline import run_pipeline
from src.ai.feedback import save_feedback
from src.config.settings import settings
from src.ui.components import (
    inject_global_css, status_badge, card,
    confidence_bar, risk_chips, sim_colour_class,
)

inject_global_css()

# ── Page header ────────────────────────────────────────────────────────────
st.title("Support Inbox")
st.caption("AI-powered triage and reply generation · AmazonHelp customer support")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Run settings")
    classifier_type = st.selectbox(
        "Classifier",
        ["auto", "llm_fewshot", "keyword_baseline"],
        help="'auto' reads type from config.yaml",
    )
    brand = st.text_input("Brand", value="AmazonHelp")
    use_llm_grounding = st.checkbox("LLM grounding check", value=True)
    st.divider()
    provider = settings.llm_provider
    if provider == "mock":
        st.info("🔧 Mock mode — no API cost.\nAll LLM calls return deterministic canned responses.")
    else:
        st.success(f"🔗 Live mode — provider: `{provider}`")

# ── Input area ─────────────────────────────────────────────────────────────
with card("Customer message"):
    col_msg, col_ctx = st.columns([3, 1])
    with col_msg:
        message = st.text_area(
            "message",
            placeholder=(
                "e.g. My package was supposed to arrive 3 days ago "
                "and I still haven't received it."
            ),
            height=110,
            label_visibility="collapsed",
        )
    with col_ctx:
        context_text = st.text_area(
            "Thread context (optional, one turn per line)",
            placeholder="Earlier messages…",
            height=110,
        )

run_btn = st.button(
    "🚀 Analyze & Generate Reply",
    type="primary",
    use_container_width=True,
    disabled=not message.strip(),
)

if not message.strip():
    st.caption("Enter a customer message above to get started.")

# ── Pipeline execution ──────────────────────────────────────────────────────
if run_btn and message.strip():
    ctx = []
    if context_text.strip():
        for line in context_text.strip().splitlines():
            if line.strip():
                ctx.append({"role": "customer", "text": line.strip()})

    t_start = time.perf_counter()
    with st.spinner("Analyzing…"):
        try:
            result = run_pipeline(
                message,
                thread_context=ctx or None,
                brand=brand,
                classifier_type=classifier_type,
                use_llm_grounding=use_llm_grounding,
            )
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()
    elapsed = time.perf_counter() - t_start

    if result.error:
        st.error(f"Pipeline error: {result.error}")
        st.stop()

    st.session_state["last_result"] = result
    st.session_state["last_message"] = message
    st.caption(f"⏱ Completed in {elapsed:.2f}s")

    st.divider()

    # ── Row 1: Intent  |  Escalation decision ──────────────────────────────
    col_intent, col_decision = st.columns(2, gap="medium")

    clf = result.classification
    rt  = result.routing

    with col_intent:
        with card("🎯 Intent Classification"):
            st.markdown(f"**`{clf.intent}`**")
            confidence_bar(clf.confidence)
            st.caption(f"Reasoning: {clf.reason}")
            if clf.is_escalation_sensitive:
                st.warning("⚠️ Escalation-sensitive intent.", icon="⚠️")
            if clf.needs_context:
                st.info("Classifier flagged this message as ambiguous.", icon="ℹ️")

    with col_decision:
        with card("🚦 Escalation Decision"):
            if rt:
                status_badge(rt.decision)
                st.markdown(f"**Reason:** {rt.reason}")
                if rt.risk_flags:
                    st.markdown("**Risk flags:**")
                    risk_chips(rt.risk_flags)
                st.markdown("")
                confidence_bar(rt.confidence, label="System confidence")
            else:
                st.warning("No routing result.")

    # ── Row 2: Evidence trace ───────────────────────────────────────────────
    st.divider()
    with card("🔍 Evidence Trace — Grounding"):
        ret = result.retrieval
        if ret and ret.items:
            if ret.has_sufficient_evidence:
                st.success(
                    f"Sufficient evidence (top similarity: **{ret.top_similarity:.2f}**)",
                    icon="✅",
                )
            else:
                st.warning(
                    f"Weak evidence — top similarity {ret.top_similarity:.2f} "
                    f"< threshold {settings.retrieval_similarity_threshold}. "
                    "Reply confidence is reduced.",
                    icon="⚠️",
                )

            for i, ev in enumerate(ret.items[:3], 1):
                css_cls = sim_colour_class(ev.similarity)
                sim_label = (
                    f'<span class="{css_cls}">{ev.similarity:.2f}</span>'
                )
                with st.expander(
                    f"Evidence [{i}]  ·  similarity {ev.similarity:.2f}  ·  {ev.relevance_reason}",
                    expanded=(i == 1),
                ):
                    st.markdown(
                        f"**Historical customer message:**  \n_{ev.customer_message[:250]}_"
                    )
                    st.markdown(
                        f"**Amazon replied:**  \n_{ev.brand_reply[:350]}_"
                    )
                    st.caption(f"Thread ID: {ev.thread_id}")
        else:
            st.warning("No evidence retrieved for this message.")

    # ── Row 3: Generated reply ──────────────────────────────────────────────
    st.divider()
    with card("💬 Generated Reply"):
        gen = result.generation
        val = result.validation

        if gen:
            # Grounding status banner
            if val:
                if val.passed:
                    st.success(
                        f"Grounding check passed (score: {val.score:.2f})",
                        icon="✅",
                    )
                else:
                    st.error(
                        f"Grounding issues detected (score: {val.score:.2f})",
                        icon="✗",
                    )
                    for flag in (val.rule_flags + val.llm_flags):
                        st.markdown(f"  ⚠️ {flag}")

            if gen.resolution_pattern:
                st.caption(f"Historical pattern: {gen.resolution_pattern}")

            reply_area = st.text_area(
                "Reply — edit before sending",
                value=gen.reply,
                height=150,
                key="reply_edit",
            )

            # ── Feedback controls ───────────────────────────────────────────
            st.divider()
            with card("👤 Human Review"):
                st.caption(
                    "Feedback is **logged for future use**. "
                    "The system does not currently retrain from this data."
                )
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    accept = st.button("✅ Accept", use_container_width=True)
                with col_b:
                    reject = st.button("❌ Reject", use_container_width=True)
                with col_c:
                    override_intent = st.selectbox(
                        "Correct intent (if wrong)",
                        ["(keep predicted)",
                         "order_delivery_tracking", "product_return_refund",
                         "account_access_login", "account_security_compromise",
                         "product_question_usage", "technical_bug_malfunction",
                         "prime_subscription_billing", "seller_third_party_issue",
                         "general_feedback_other"],
                        key="override_intent",
                    )
                override_esc = st.selectbox(
                    "Override escalation",
                    ["(keep system decision)", "AUTO_HANDLE", "ESCALATE"],
                    key="override_esc",
                )
                notes = st.text_input("Notes (optional)", key="feedback_notes")

                if accept or reject:
                    save_feedback(
                        customer_message=message,
                        predicted_intent=clf.intent,
                        corrected_intent=(
                            None if override_intent == "(keep predicted)"
                            else override_intent
                        ),
                        predicted_decision=rt.decision if rt else "?",
                        overridden_decision=(
                            None if override_esc == "(keep system decision)"
                            else override_esc
                        ),
                        reply_accepted=accept,
                        edited_reply=(
                            reply_area if reply_area != gen.reply else None
                        ),
                        notes=notes or None,
                        pipeline_result=result.to_dict(),
                    )
                    st.success("Feedback logged. Thank you!")
        else:
            st.warning("No reply generated.")
