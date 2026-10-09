"""Tab 5 - Operational monitoring: efficiency by operation mode, network vs sensor impact."""
import pandas as pd
import streamlit as st

from src import charts, ui
from src.config import C, GROUP, LABEL, MODES, NETWORK_BANDS, RAW_NUMERIC


@st.fragment
def render(ctx):
    df = ctx.fdf
    if df.empty:
        ui.note("No readings match the sidebar filters.", "warn")
        return
    a, b = st.columns(2)
    with a.container(border=True):
        ui.section("Efficiency by operation mode", "Class share of readings in each mode")
        ui.plot(charts.share_stack(df, "Operation_Mode", order=MODES))
        g = df.groupby("Operation_Mode").agg(Readings=("Machine_ID", "size"), Index=("Efficiency_Index", "mean"), Error=("Error_Rate_%", "mean"),
                                             Speed=("Production_Speed_units_per_hr", "mean")).round(2).reset_index()
        ui.table(g.rename(columns={"Operation_Mode": "Mode", "Index": "Efficiency index", "Error": "Avg error %", "Speed": "Avg units/hr"}))
    with b.container(border=True):
        ui.section("Efficiency by 6G network quality", "Poor, Fair and Good bands of the network reliability score")
        ui.plot(charts.share_stack(df, "Network_Band", order=NETWORK_BANDS))
        g = df.groupby("Network_Band").agg(Readings=("Machine_ID", "size"), Index=("Efficiency_Index", "mean"), Latency=("Network_Latency_ms", "mean"),
                                           Loss=("Packet_Loss_%", "mean")).round(2).reset_index()
        ui.table(g.rename(columns={"Network_Band": "Network", "Index": "Efficiency index", "Latency": "Avg latency ms", "Loss": "Avg loss %"}))

    with st.container(border=True):
        ui.section("Network versus sensor impact", "Does the network or the machine itself explain efficiency? Flat lines mean no influence.")
        c1, c2 = st.columns(2)
        with c1:
            ui.plot(charts.quartile_lines(df, {"Network_Reliability": C["sky"], "Sensor_Stability": C["teal"],
                                               "Thermal_Vibration_Stress": C["violet"], "Error_Rate_%": C["Low"]}))
        with c2:
            corr = df[RAW_NUMERIC].corrwith(df["Efficiency_Index"]).fillna(0)
            ui.plot(charts.corr_bar(corr, GROUP))
        imp = pd.DataFrame(ctx.metrics["importance"])
        gs = imp.groupby("group")["share"].sum().sort_values(ascending=False).to_dict()
        sensor, net = gs.get("Sensor", 0), gs.get("Network", 0)
        e = ctx.metrics["early"]
        ui.note(f"In the trained model, <b>sensor metrics</b> account for {sensor:.1%} of importance and <b>network metrics</b> for {net:.1%}; "
                f"quality and production metrics account for {gs.get('Quality', 0) + gs.get('Production', 0):.1%}. "
                f"A model using only sensor, network and maintenance data reaches ROC-AUC {e['roc_auc']:.2f} (0.50 = coin flip).", "warn")
        c3, c4 = st.columns([1, 1.4])
        with c3:
            ui.plot(charts.group_share(gs))
        with c4:
            st.markdown("**How to read this.** In this dataset, efficiency is set by output quality (error rate) and throughput (units per hour). "
                        "Network and sensor deviations do not move the label. Use them as health indicators and alerts, "
                        "and use error rate and speed to judge efficiency.")
