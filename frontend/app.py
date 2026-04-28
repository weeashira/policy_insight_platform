import streamlit as st
from dashboard import render_dashboard
from methodology import render_methodology

st.set_page_config(page_title="Policy Insight Platform", layout="wide")

dashboard_tab, methodology_tab = st.tabs([
    "📊 Dashboard",
    "📘 Methodology & Disclaimers"
])

with dashboard_tab:
    render_dashboard()

with methodology_tab:
    render_methodology()