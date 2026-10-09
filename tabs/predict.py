"""Tab 2 - Efficiency prediction dashboard with metric sensitivity sliders."""
import numpy as np
import pandas as pd
import streamlit as st

from src import charts, data as D, narrative, ui
from src.config import C, CLASS_COLORS, CLASSES, LABEL, LIMITS, METRICS, MODES, RAW_NUMERIC, UNIT
from src.modeling import efficiency_index

STEP = {"Temperature_C": 0.1, "Vibration_Hz": 0.01, "Power_Consumption_kW": 0.05, "Network_Latency_ms": 0.1, "Packet_Loss_%": 0.01,
        "Quality_Control_Defect_Rate_%": 0.05, "Production_Speed_units_per_hr": 1.0, "Predictive_Maintenance_Score": 0.01, "Error_Rate_%": 0.05}


def _load(values: dict):
    for c in RAW_NUMERIC:
        lo, hi = LIMITS[c]
        st.session_state[f"p_{c}"] = float(min(max(values[c], lo), hi))
    st.session_state["p_mode"] = values.get("Operation_Mode", "Active")


def _row() -> pd.DataFrame:
    ss = st.session_state
    return pd.DataFrame([{**{c: float(ss[f"p_{c}"]) for c in RAW_NUMERIC}, "Operation_Mode": ss["p_mode"]}])


@st.fragment
def render(ctx):
    ss, base, model = st.session_state, ctx.base, ctx.primary
    med = base.groupby("Efficiency_Status")[RAW_NUMERIC].median()
    for c in RAW_NUMERIC:
        ss.setdefault(f"p_{c}", float(base[c].median()))
    ss.setdefault("p_mode", "Active")

    ui.page_title("Efficiency Dashboard", "Describe a machine reading and see the efficiency class, confidence and reasons instantly")
    left, right = st.columns([1.05, 1])
    with left.container(border=True):
        ui.section("Describe a machine reading", "Move any slider: the prediction and sensitivity charts update instantly")
        b = st.columns(5)
        b[0].button("Typical Low", on_click=_load, args=({**med.loc["Low"].to_dict()},), key="pr_low")
        b[1].button("Typical Medium", on_click=_load, args=({**med.loc["Medium"].to_dict()},), key="pr_med")
        b[2].button("Typical High", on_click=_load, args=({**med.loc["High"].to_dict()},), key="pr_high")
        last = base.iloc[min(ss.cursor, len(base)) - 1]
        b[3].button("Live reading", on_click=_load, args=({c: last[c] for c in RAW_NUMERIC} | {"Operation_Mode": last["Operation_Mode"]},), key="pr_live")
        rnd = base.sample(1, random_state=int(ss.get("pr_seed", 0))).iloc[0]
        b[4].button("Random", on_click=lambda r=rnd: (_load({c: r[c] for c in RAW_NUMERIC} | {"Operation_Mode": r["Operation_Mode"]}),
                                                      ss.__setitem__("pr_seed", int(ss.get("pr_seed", 0)) + 1)), key="pr_rnd")
        ss["p_mode"] = ss["p_mode"] if ss["p_mode"] in MODES else "Active"
        st.selectbox("Operation mode", MODES, key="p_mode")
        c1, c2 = st.columns(2)
        for i, c in enumerate(RAW_NUMERIC):
            lo, hi = LIMITS[c]
            (c1 if i % 2 == 0 else c2).slider(f"{LABEL[c]} {('(' + UNIT[c] + ')') if UNIT[c] else ''}", lo, hi, step=STEP[c], key=f"p_{c}")

    row = _row()
    p = model.proba(row)[0]
    pred = CLASSES[int(p.argmax())]
    d = row.iloc[0].to_dict()
    with right.container(border=True):
        ui.section("Predicted efficiency")
        k = st.columns([1, 1])
        k[0].markdown(f"<div class='bigstate' style='color:{CLASS_COLORS[pred]}'>{pred}</div><div class='s' style='color:#8B91A1'>"
                      f"Efficiency index {efficiency_index(p):.0f}/100</div>", unsafe_allow_html=True)
        k[1].markdown(ui.kpi("Confidence", f"{p.max():.0%}", "model certainty", CLASS_COLORS[pred]), unsafe_allow_html=True)
        ui.plot(charts.proba_bars(p))
        st.markdown("**Why**")
        for line in narrative.reasons(d, pred):
            st.markdown(f"- {line}")
        ui.note("💡 " + narrative.next_step(d, pred))
        ss.setdefault("pr_machine", 1)
        st.number_input("Machine ID for this reading", 1, 9999, step=1, key="pr_machine")
        if st.button("Save this reading to the dataset and live feed", key="pr_save"):
            clock = ss.get("clock", base["Datetime"].iloc[0])
            e = row.assign(Datetime=clock, Machine_ID=int(ss.get("pr_machine", 1)), Efficiency_Status=np.nan, Note="saved from Predict tab")
            clean, probs = D.validate_entries(e)
            D.save_entries(clean)
            ss.flash = f"Saved a {pred} reading. It now appears in the Live Monitor and every analysis tab."
            st.rerun(scope="app")

    with st.container(border=True):
        ui.section("Metric sensitivity", "How much does each metric move the outcome while everything else stays as set above?")
        a, b2 = st.columns([1, 1])
        grids = {}
        for c in RAW_NUMERIC:
            lo, hi = LIMITS[c]
            x = np.linspace(lo, hi, 41)
            g = pd.concat([row] * len(x), ignore_index=True)
            g[c] = x
            grids[c] = (x, model.proba(g))
        swing = pd.DataFrame([(LABEL[c], float(P[:, CLASSES.index(pred)].max() - P[:, CLASSES.index(pred)].min())) for c, (x, P) in grids.items()],
                             columns=["label", "swing"])
        with a:
            ui.plot(charts.tornado(swing, f"Swing in P({pred}) across each metric's full range"))
        with b2:
            pick = st.selectbox("Sweep one metric", RAW_NUMERIC, format_func=lambda c: LABEL[c],
                                index=RAW_NUMERIC.index("Error_Rate_%"), key="pr_sweep")
            x, P = grids[pick]
            ui.plot(charts.sens_curves(x, P, float(row[pick].iloc[0]), f"{LABEL[pick]} {UNIT[pick]}"))
        top = swing.sort_values("swing", ascending=False)
        ui.note(f"Only <b>{top.iloc[0]['label']}</b> and <b>{top.iloc[1]['label']}</b> move the prediction in a meaningful way. "
                "Sensor, 6G network and maintenance sliders barely change the class, which matches the Model Lab findings.")
