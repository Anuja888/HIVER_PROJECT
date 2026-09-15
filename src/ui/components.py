"""
Shared UI components for the Hiver support-agent app.

Three reusable building blocks used consistently across all pages:
  - status_badge()    — AUTO_HANDLE / ESCALATE pill (color + icon + text)
  - card()            — context manager wrapping st.container(border=True)
  - confidence_bar()  — progress bar that always renders the uncalibrated disclaimer

All badge colours use inline CSS so they are independent of the Streamlit theme
(important for the ESCALATE red which must remain red regardless of primaryColor).
"""
from __future__ import annotations

import contextlib
from typing import Iterator

import streamlit as st

# ---------------------------------------------------------------------------
# Global CSS injected once per page load.
# Call inject_global_css() at the top of every page.
# ---------------------------------------------------------------------------

_CSS = """
<style>
/* ── Status badges ─────────────────────────────────────────────────── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.82rem;
    letter-spacing: 0.03em;
    line-height: 1;
}
.badge-auto {
    background: #D1FAE5;   /* green-100 */
    color: #065F46;         /* green-800 */
    border: 1.5px solid #6EE7B7;
}
.badge-escalate {
    background: #FEE2E2;   /* red-100 */
    color: #991B1B;         /* red-800 */
    border: 1.5px solid #FCA5A5;
}
.badge-preliminary {
    background: #FEF3C7;   /* amber-100 */
    color: #92400E;
    border: 1.5px solid #FCD34D;
    font-size: 0.72rem;
    padding: 3px 9px;
}

/* ── Card ──────────────────────────────────────────────────────────── */
/* Streamlit's border=True container already draws a border; we add    */
/* consistent internal padding via a wrapper class.                    */
.card-heading {
    font-size: 0.92rem;
    font-weight: 700;
    color: #374151;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}

/* ── Confidence bar label ───────────────────────────────────────────── */
.uncalibrated-note {
    font-size: 0.73rem;
    color: #6B7280;
    margin-top: 2px;
    font-style: italic;
}

/* ── Empty-state preview box ────────────────────────────────────────── */
.empty-preview {
    border: 1.5px dashed #CBD5E1;
    border-radius: 8px;
    padding: 18px 22px;
    background: #F1F5F9;
    margin: 8px 0;
}
.empty-preview p {
    margin: 4px 0;
    color: #64748B;
    font-size: 0.85rem;
}
.empty-preview .ep-title {
    font-weight: 700;
    font-size: 0.9rem;
    color: #1E3A5F;
    margin-bottom: 8px;
}

/* ── Risk flag chips ────────────────────────────────────────────────── */
.risk-chip {
    display: inline-block;
    background: #FEF9C3;
    color: #713F12;
    border: 1px solid #FDE68A;
    border-radius: 12px;
    padding: 2px 9px;
    font-size: 0.72rem;
    margin: 2px 2px 2px 0;
    font-weight: 600;
}

/* ── Evidence card ──────────────────────────────────────────────────── */
.ev-sim-high  { color: #065F46; font-weight: 700; }
.ev-sim-med   { color: #92400E; font-weight: 700; }
.ev-sim-low   { color: #991B1B; font-weight: 700; }
</style>
"""


def inject_global_css() -> None:
    """Call once at the top of each page module."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# status_badge
# ---------------------------------------------------------------------------

def status_badge(decision: str) -> None:
    """
    Render an AUTO_HANDLE or ESCALATE pill.

    Uses inline HTML/CSS so the colour is guaranteed regardless of theme.
    Accessibility: colour + icon + text (not colour alone).

    Args:
        decision: "AUTO_HANDLE" or "ESCALATE" (any other value → grey pill)
    """
    if decision == "AUTO_HANDLE":
        html = '<span class="badge badge-auto">✅ AUTO-HANDLE</span>'
    elif decision == "ESCALATE":
        html = '<span class="badge badge-escalate">🚨 ESCALATE</span>'
    else:
        html = f'<span class="badge" style="background:#E5E7EB;color:#374151;border:1.5px solid #9CA3AF">{decision}</span>'
    st.markdown(html, unsafe_allow_html=True)


def preliminary_badge(n: int) -> None:
    """Small 'n=X, preliminary' label shown when golden set is partial."""
    st.markdown(
        f'<span class="badge badge-preliminary">⚠ n={n} — preliminary</span>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# card
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def card(heading: str | None = None) -> Iterator[None]:
    """
    Context manager: renders content inside a styled bordered container.

    Usage::

        with card("Intent Classification"):
            st.write("...")

    The heading is rendered in small-caps above the content using the
    .card-heading CSS class (not st.subheader, which is too large).
    """
    with st.container(border=True):
        if heading:
            st.markdown(f'<p class="card-heading">{heading}</p>',
                        unsafe_allow_html=True)
        yield


# ---------------------------------------------------------------------------
# confidence_bar
# ---------------------------------------------------------------------------

def confidence_bar(value: float, label: str = "Confidence") -> None:
    """
    Render a labelled progress bar that always includes the uncalibrated disclaimer.

    Args:
        value: Float 0.0–1.0
        label: Display label (default "Confidence")
    """
    # Clamp to valid range defensively
    value = max(0.0, min(1.0, float(value)))
    st.progress(value, text=f"{label}: {value:.0%}")
    st.markdown(
        '<p class="uncalibrated-note">'
        "⚠️ Uncalibrated model signal — not a calibrated probability."
        "</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# risk_chips
# ---------------------------------------------------------------------------

def risk_chips(flags: list[str]) -> None:
    """Render a row of yellow risk-flag chips."""
    if not flags:
        return
    chips = "".join(
        f'<span class="risk-chip">{f}</span>'
        for f in flags
    )
    st.markdown(chips, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# empty_state
# ---------------------------------------------------------------------------

def empty_state(
    title: str,
    description: str,
    preview_items: list[str] | None = None,
    cta_label: str | None = None,
    cta_page: str | None = None,
) -> bool:
    """
    Render a designed empty-state panel instead of a bare st.warning/st.info.

    Returns True if the CTA button was clicked (caller may use st.switch_page).

    Args:
        title:          Heading line, e.g. "No labels yet"
        description:    One sentence explaining why the page is empty
        preview_items:  Bullet list of what will appear when data exists
        cta_label:      Button label, e.g. "Go to Label Mode"
        cta_page:       Streamlit page path for st.switch_page (optional)
    """
    inject_global_css()
    preview_html = ""
    if preview_items:
        bullets = "".join(f"<p>• {item}</p>" for item in preview_items)
        preview_html = f"""
        <div class="empty-preview">
          <p class="ep-title">Once data is available this page will show:</p>
          {bullets}
        </div>"""

    st.markdown(
        f"""
        <div class="empty-preview">
          <p class="ep-title">{title}</p>
          <p>{description}</p>
        </div>
        {preview_html}
        """,
        unsafe_allow_html=True,
    )

    if cta_label and cta_page:
        if st.button(cta_label, type="primary", use_container_width=False):
            st.switch_page(cta_page)
            return True
    return False


# ---------------------------------------------------------------------------
# sim_colour_class
# ---------------------------------------------------------------------------

def sim_colour_class(sim: float) -> str:
    """CSS class name for a similarity score."""
    if sim >= 0.60:
        return "ev-sim-high"
    if sim >= 0.40:
        return "ev-sim-med"
    return "ev-sim-low"
