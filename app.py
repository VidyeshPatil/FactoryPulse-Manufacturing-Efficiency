"""Smart Factory Efficiency Intelligence - Streamlit entry point.   Run:  streamlit run app.py"""
from types import SimpleNamespace

import pandas as pd
import streamlit as st
from PIL import Image

from src.config import APP_NAME, ASSETS, MODES, NETWORK_BANDS

st.set_page_config(page_title=APP_NAME, page_icon=Image.open(ASSETS / "logo.png"), layout="wide", initial_sidebar_state="collapsed")

from src import data as D, filters, ui                      # noqa: E402  (after set_page_config)
from src.modeling import load_artifacts, train_all          # noqa: E402
from tabs import add_data, explain, explorer, live, machines, ops, predict   # noqa: E402

ui.inject_css()


# ------------------------------------------------------------------ cached ---
@st.cache_resource(show_spinner="Preparing the AI models (first run trains them, about a minute)...")
def get_models():
    return load_artifacts() or train_all(D.training_frame(D.load_base()))


@st.cache_resource(show_spinner="Loading and scoring 100,000 readings...")
def get_base(stamp: str):
    return D.score(D.load_base(), get_models()["primary"])


@st.cache_resource(show_spinner=False, max_entries=2)
def get_all(stamp: str, sig: tuple):
    es = D.score(D.load_entries(), get_models()["primary"])
    return D.combine(get_base(stamp), es), es


# ------------------------------------------------------------------- state ---
ss = st.session_state
for k, v in dict(cursor=600, playing=True, rows_per_tick=5, window=240, flash=None).items():
    ss.setdefault(k, v)

art = get_models()
stamp = art["metrics"]["trained_at"]
base = get_base(stamp)
all_df, entries = get_all(stamp, D.entries_signature())
machine_ids = sorted(all_df["Machine_ID"].unique().tolist())
dmin, dmax = all_df["Datetime"].min().date(), all_df["Datetime"].max().date()


# ----------------------------------------------------------------- sidebar ---
def _reset_filters():
    for k in ("f_machines", "f_modes", "f_nets", "f_dates", "f_hours"):
        ss.pop(k, None)


with st.sidebar:
    st.markdown(f"<div class='side-brand'><img src='data:image/png;base64,{ui._b64(str(ASSETS / 'logo.png'))}'/>"
                f"<div><b>{APP_NAME}</b><span>FILTERS</span></div></div>", unsafe_allow_html=True)
    # keep stored selections valid when entries are added or deleted (only touch state when something is invalid)
    if "f_machines" in ss and any(m not in machine_ids for m in ss["f_machines"]):
        ss["f_machines"] = [m for m in ss["f_machines"] if m in machine_ids]
    if "f_dates" in ss and isinstance(ss["f_dates"], (tuple, list)) and any(not (dmin <= d <= dmax) for d in ss["f_dates"]):
        ss["f_dates"] = tuple(min(max(d, dmin), dmax) for d in ss["f_dates"])
    machines_sel = st.multiselect("Machines (empty = all)", machine_ids, key="f_machines")
    modes_sel = st.multiselect("Operation mode", MODES, default=MODES, key="f_modes")
    nets_sel = st.multiselect("Network quality", NETWORK_BANDS, default=NETWORK_BANDS, key="f_nets")
    dates = st.date_input("Time window (analysis tabs)", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="f_dates")
    hours = st.slider("Hours of the day", 0, 24, (0, 24), key="f_hours")
    st.button("Reset filters", on_click=_reset_filters)
    d0, d1 = (dates[0], dates[1]) if isinstance(dates, (tuple, list)) and len(dates) == 2 else (dmin, dmax)
    F = dict(machines=machines_sel, modes=modes_sel, nets=nets_sel, d0=d0, d1=d1, h0=hours[0], h1=hours[1])
    fdf = filters.apply(all_df, F)
    st.markdown("---")
    st.caption(f"**{len(fdf):,}** of {len(all_df):,} readings selected · {len(entries)} added by you")
    st.caption("The Live Monitor follows the replay clock and uses the machine, mode and network filters.")

ctx = SimpleNamespace(primary=art["primary"], early=art["early"], metrics=art["metrics"], base=base, all=all_df, entries=entries, fdf=fdf, F=F,
                      app_name=APP_NAME, n_machines=base["Machine_ID"].nunique())

# -------------------------------------------------------------------- page ---
pm = next(r for r in art["metrics"]["models"] if r["model"] == art["metrics"]["primary"])
ui.topbar(APP_NAME, "AI efficiency classification · sensor + 6G network intelligence",
          f"Live model · {art['metrics']['primary']} · {pm['accuracy']:.2%} test accuracy")

if ss.flash:
    st.success(ss.flash)
    ss.flash = None

tabs = st.tabs(["Live Monitor", "Efficiency Dashboard", "Machine Insights", "Explainability", "Operational Monitoring", "Live Simulator", "Data Explorer"])
for tab, mod in zip(tabs, (live, predict, machines, explain, ops, add_data, explorer)):
    with tab:
        mod.render(ctx)
