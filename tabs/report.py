"""Tab 8 - Executive summary and project coverage."""
import pandas as pd
import streamlit as st

from src import report, ui

COVERAGE = [
    ("Handle time ordering", "Clock rebuilt in strict minute order; duplicated day repaired", "Model Lab > Data quality"),
    ("Encode and scale", "One-hot operation mode, standard-scaled metrics", "src/modeling.py"),
    ("Class imbalance", "Class-weighted models, macro F1 reported", "Model Lab"),
    ("Feature engineering", "Sensor stability, energy ratio, error-to-output ratio, network reliability", "Model Lab > Engineered features"),
    ("Baseline and advanced models", "Logistic Regression, Random Forest, Gradient Boosting (XGBoost if installed)", "Model Lab > Comparison"),
    ("Feature importance and explainability", "Permutation importance, per-reading reasons, change attribution", "Explain tab"),
    ("Real-time label with confidence", "Every live reading carries a class and a confidence", "Live Monitor, Predict"),
    ("Efficiency prediction dashboard", "Live replay, prediction, confidence gauge", "Live Monitor, Predict"),
    ("Machine-level insights", "Per-machine trends, history, profiles", "Machines tab"),
    ("Operational monitoring view", "Mode and network versus sensor comparison", "Operations tab"),
    ("User controls", "Machine, mode, network quality, time window, sensitivity sliders", "Sidebar, Predict"),
    ("Executive summary", "Generated from live data, downloadable", "This tab"),
]


@st.fragment
def render(ctx):
    md, page = report.build(ctx)
    with st.container(border=True):
        ui.section("Executive summary for government stakeholders", "Written from the data currently selected in the sidebar")
        st.markdown(md.split("\n", 2)[2])
        c = st.columns([1, 1, 4])
        c[0].download_button("Download HTML", page, "executive_summary.html", "text/html")
        c[1].download_button("Download Markdown", md, "executive_summary.md", "text/markdown")
    with st.container(border=True):
        ui.section("Project requirements and where they live")
        ui.table(pd.DataFrame(COVERAGE, columns=["Requirement", "How it is met", "Where"]))
