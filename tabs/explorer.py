"""Tab 7 - Data explorer: browse readings, model lab and executive summary in one place."""
import streamlit as st

from src import ui
from src.config import RAW_NUMERIC
from tabs import lab, report

SHOW = ["Datetime", "Machine_ID", "Operation_Mode", "Error_Rate_%", "Production_Speed_units_per_hr", "Temperature_C", "Vibration_Hz",
        "Network_Latency_ms", "Packet_Loss_%", "Pred", "Confidence", "Efficiency_Status", "Source"]


@st.fragment
def _browse(ctx):
    df = ctx.fdf
    with st.container(border=True):
        ui.section("Browse readings", f"{len(df):,} readings match the filters", tip="Every row is one machine reading with the AI's verdict next to the recorded label.")
        c = st.columns([1, 1, 1, 3])
        n = c[0].select_slider("Rows to show", options=[50, 200, 1000, 5000], value=200, key="ex_rows")
        cls = c[1].selectbox("Class", ["All", "Low", "Medium", "High"], key="ex_cls")
        newest = c[2].selectbox("Order", ["Newest first", "Oldest first"], key="ex_ord")
        d = df if cls == "All" else df[df["Efficiency_Status"] == cls]
        d = d.sort_values("Datetime", ascending=newest == "Oldest first").head(n)
        ui.table(d[SHOW].round(3))
        if st.checkbox("Prepare a CSV download of this view", key="ex_prep"):
            st.download_button("Download CSV", d[SHOW].to_csv(index=False).encode(), "factorypulse_readings.csv", "text/csv", key="ex_dl")
    with st.container(border=True):
        ui.section("Summary statistics", "Average, spread and range of each metric")
        ui.table(df[RAW_NUMERIC].describe().T.round(2).reset_index().rename(columns={"index": "Metric"}))


def render(ctx):
    t1, t2, t3 = st.tabs(["Browse data", "Model lab", "Executive summary"])
    with t1:
        _browse(ctx)
    with t2:
        lab.render(ctx)
    with t3:
        report.render(ctx)
