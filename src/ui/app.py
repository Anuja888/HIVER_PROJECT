"""
Hiver SDE Intern Take-Home — Main Streamlit Application.

Multi-page app with:
  - Inbox (message classification + reply generation)
  - Evaluation Dashboard
  - Analytics / Failure Explorer
  - Label Mode (golden set labelling)

Run: streamlit run src/ui/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Hiver AI Support Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Shared CSS ----
st.markdown("""
<style>
.badge-auto {
    background-color: #1a7f4b;
    color: white;
    padding: 4px 12px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 0.85em;
}
.badge-escalate {
    background-color: #c0392b;
    color: white;
    padding: 4px 12px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 0.85em;
}
.evidence-card {
    border: 1px solid #2d3748;
    border-radius: 6px;
    padding: 10px;
    margin: 6px 0;
    background: #1A1D27;
}
.metric-card {
    background: #1A1D27;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.warning-box {
    background: #4a3500;
    border: 1px solid #f5a623;
    border-radius: 6px;
    padding: 10px;
    margin: 6px 0;
}
.uncalibrated-note {
    color: #aaa;
    font-size: 0.78em;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)

inbox = st.Page("pages/inbox.py", title="📥 Inbox", icon="📥")
eval_page = st.Page("pages/eval_dashboard.py", title="📊 Evaluation", icon="📊")
analytics = st.Page("pages/analytics.py", title="🔍 Analytics", icon="🔍")
label = st.Page("pages/label_mode.py", title="🏷️ Label Mode", icon="🏷️")

pg = st.navigation([inbox, eval_page, analytics, label])
pg.run()
