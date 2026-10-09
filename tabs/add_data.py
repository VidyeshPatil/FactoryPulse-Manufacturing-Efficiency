"""Tab 6 - Add new readings (single entry, CSV upload) and manage saved entries."""
import pandas as pd
import streamlit as st

from src import data as D, ui
from src.config import ASSETS, CLASSES, LABEL, LIMITS, MODES, RAW_NUMERIC, UNIT

DEFAULTS = {"Temperature_C": 60.0, "Vibration_Hz": 2.5, "Power_Consumption_kW": 5.7, "Network_Latency_ms": 25.0, "Packet_Loss_%": 2.5,
            "Quality_Control_Defect_Rate_%": 5.0, "Production_Speed_units_per_hr": 300.0, "Predictive_Maintenance_Score": 0.5, "Error_Rate_%": 3.0}


def _flash(msg: str):
    st.session_state.flash = msg
    st.rerun(scope="app")


def _inject(kind: str, clock):
    r = dict(DEFAULTS)
    if kind == "fault":
        r.update({"Error_Rate_%": 9.5, "Production_Speed_units_per_hr": 140.0, "Packet_Loss_%": 4.6, "Vibration_Hz": 4.8, "Temperature_C": 88.0})
    else:
        r.update({"Error_Rate_%": 0.8, "Production_Speed_units_per_hr": 470.0, "Packet_Loss_%": 0.4, "Network_Latency_ms": 4.0})
    row = pd.DataFrame([{**r, "Datetime": clock, "Machine_ID": 1, "Operation_Mode": "Active", "Efficiency_Status": None,
                         "Note": f"injected {kind} demo"}])
    clean, _ = D.validate_entries(row)
    D.save_entries(clean)
    st.session_state.flash = f"Injected a demo {kind} reading. Open the Live Monitor to see it arrive."


@st.fragment
def render(ctx):
    ss = st.session_state
    clock = ss.get("clock", ctx.base["Datetime"].iloc[0])
    ui.page_title("Live Simulator", "Add machine readings by hand or by file and watch the monitor react")
    with st.container(border=True):
        ui.image(ASSETS / "pipeline.png")
        st.caption("New readings flow through the same path as the factory data: validated, classified by the AI model, saved, and shown live.")

    t1, t2, t3 = st.tabs(["Add one reading", "Upload a CSV", "Saved entries"])
    with t1:
        with st.container(border=True):
            ui.section("New machine reading", f"Time defaults to the live replay clock ({clock:%d %b %Y %H:%M}), so the reading shows up in the Live Monitor right away")
            with st.form("entry_form"):
                a = st.columns(4)
                d = a[0].date_input("Date", value=clock.date())
                t = a[1].time_input("Time", value=clock.time())
                mid = a[2].number_input("Machine ID (new IDs add a machine)", 1, 9999, 1, 1)
                mode = a[3].selectbox("Operation mode", MODES)
                vals = {}
                cols = st.columns(3)
                for i, c in enumerate(RAW_NUMERIC):
                    lo, hi = LIMITS[c]
                    vals[c] = cols[i % 3].number_input(f"{LABEL[c]} {('(' + UNIT[c] + ')') if UNIT[c] else ''}", float(lo), float(hi), DEFAULTS[c],
                                                       step=(hi - lo) / 200, format="%.3f")
                b = st.columns([1, 1, 2])
                label = b[0].selectbox("Efficiency label", ["Let the AI decide"] + CLASSES)
                note = b[1].text_input("Note (optional)")
                submitted = st.form_submit_button("Save reading", type="primary")
            if submitted:
                row = pd.DataFrame([{**vals, "Datetime": pd.Timestamp.combine(d, t), "Machine_ID": mid, "Operation_Mode": mode,
                                     "Efficiency_Status": None if label.startswith("Let") else label, "Note": note}])
                clean, probs = D.validate_entries(row)
                if len(clean):
                    D.save_entries(clean)
                    pred = ctx.primary.predict(clean)[0]
                    _flash(f"Saved. The AI classifies this reading as {pred}."
                           + ("" if pd.Timestamp.combine(d, t) <= clock else " It will appear in the live feed when the replay clock reaches its time."))
                else:
                    st.error(" ".join(probs) or "Reading could not be saved.")
        with st.container(border=True):
            ui.section("Quick demo actions", "Create a test reading and watch the Live Monitor react")
            q = st.columns(4)
            if q[0].button("🚨 Inject a fault", key="inj_f"):
                _inject("fault", clock); st.rerun(scope="app")
            if q[1].button("🌟 Inject an excellent run", key="inj_g"):
                _inject("excellent", clock); st.rerun(scope="app")

    with t2:
        with st.container(border=True):
            ui.section("Upload many readings", "Columns: Datetime (or Date and Timestamp), Machine_ID, Operation_Mode, the nine metrics, and optional Efficiency_Status")
            st.download_button("Download CSV template", D.template_csv(), "new_readings_template.csv", "text/csv")
            up = st.file_uploader("CSV file", type="csv", key="up_csv")
            if up is not None:
                try:
                    raw = pd.read_csv(up)
                    clean, probs = D.validate_entries(raw)
                    for p in probs:
                        st.warning(p)
                    if len(clean):
                        st.success(f"{len(clean)} valid row(s) ready.")
                        ui.table(clean.head(10))
                        if st.button(f"Add {len(clean)} reading(s)", type="primary", key="up_go"):
                            D.save_entries(clean)
                            _flash(f"Added {len(clean)} reading(s) from {up.name}.")
                except Exception as exc:
                    st.error(f"Could not read that file: {exc}")

    with t3:
        with st.container(border=True):
            e = ctx.entries
            ui.section("Entries added in this app", f"{len(e)} saved · stored in data/new_entries.csv")
            if e.empty:
                st.info("Nothing added yet. Use the first tab to add a reading.")
            else:
                show = e[["Datetime", "Machine_ID", "Operation_Mode", "Error_Rate_%", "Production_Speed_units_per_hr", "Pred", "Confidence", "Efficiency_Status", "Label_Source", "Note"]]
                ui.table(show.sort_values("Datetime", ascending=False))
                c = st.columns([1, 1, 3])
                c[0].download_button("Download entries", e[D.ENTRY_COLS].to_csv(index=False).encode(), "new_entries.csv", "text/csv")
                ok = c[1].checkbox("I want to delete all entries", key="del_ok")
                if c[1].button("Delete all", disabled=not ok, key="del_go"):
                    D.delete_entries()
                    _flash("All new entries were deleted.")
