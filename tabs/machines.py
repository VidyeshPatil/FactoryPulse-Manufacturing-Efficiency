"""Tab 3 - Machine-level insights: trends, historical patterns and efficiency profiles."""
import pandas as pd
import streamlit as st

from src import charts, ui
from src.config import C, CLASS_COLORS, CLASSES


def _profile(df: pd.DataFrame) -> pd.DataFrame:
    sh = pd.crosstab(df["Machine_ID"], df["Efficiency_Status"], normalize="index").reindex(columns=CLASSES, fill_value=0)
    g = df.groupby("Machine_ID")
    out = pd.DataFrame({"index": g["Efficiency_Index"].mean(), "readings": g.size(), "avg_error": g["Error_Rate_%"].mean(),
                        "avg_speed": g["Production_Speed_units_per_hr"].mean(), "latest": g["Pred"].last()})
    return out.join(sh).reset_index()


@st.fragment
def render(ctx):
    df = ctx.fdf
    if df.empty:
        ui.note("No readings match the sidebar filters.", "warn")
        return
    prof = _profile(df)
    ids = sorted(prof["Machine_ID"].tolist())
    ss = st.session_state
    if ss.get("m_sel") not in ids:
        ss["m_sel"] = ids[0]

    with st.container(border=True):
        ui.section("Fleet ranking", "Average efficiency index per machine (0 = always Low, 100 = always High) in the filtered period")
        sel = st.selectbox("Inspect machine", ids, key="m_sel")
        ui.plot(charts.leaderboard(prof, highlight=sel))

    m = df[df["Machine_ID"] == sel]
    pm = prof[prof["Machine_ID"] == sel].iloc[0]
    k = st.columns(5)
    k[0].markdown(ui.kpi(f"Machine {sel} index", f"{pm['index']:.1f}", f"fleet average {prof['index'].mean():.1f}", C["teal"]), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("High share", f"{pm['High']:.1%}", "of readings", C["High"]), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("Medium share", f"{pm['Medium']:.1%}", "of readings", C["Medium"]), unsafe_allow_html=True)
    k[3].markdown(ui.kpi("Low share", f"{pm['Low']:.1%}", "of readings", C["Low"]), unsafe_allow_html=True)
    k[4].markdown(ui.kpi("Readings", f"{int(pm['readings']):,}", f"latest class: {pm['latest']}", C["sky"]), unsafe_allow_html=True)

    a, b = st.columns(2)
    with a.container(border=True):
        ui.section("Efficiency trend")
        daily = m.groupby("Date_Str", as_index=False)["Efficiency_Index"].mean()
        ui.plot(charts.machine_trend(daily, f"Machine {sel}, daily average"))
    with b.container(border=True):
        ui.section("Historical classification pattern", "Share of each class by week")
        wk = m.assign(Week=m["Datetime"].dt.strftime("%G-W%V"))
        ui.plot(charts.share_stack(wk, "Week"))

    c, d = st.columns(2)
    with c.container(border=True):
        ui.section("When does this machine run best?", "Average index by weekday and hour")
        ui.plot(charts.hour_heat(m))
    with d.container(border=True):
        ui.section("Movers over the last 7 days", "Fleet machines whose index changed most versus the 7 days before")
        last = df["Datetime"].max()
        rec = df[df["Datetime"] > last - pd.Timedelta(days=7)].groupby("Machine_ID")["Efficiency_Index"].mean()
        pri = df[(df["Datetime"] <= last - pd.Timedelta(days=7)) & (df["Datetime"] > last - pd.Timedelta(days=14))].groupby("Machine_ID")["Efficiency_Index"].mean()
        mv = (rec - pri).dropna().sort_values()
        if mv.empty:
            st.info("Widen the time window to compare two consecutive weeks.")
        else:
            show = pd.concat([mv.head(5), mv.tail(5)]).reset_index()
            show.columns = ["Machine", "Change in index"]
            show["Direction"] = ["Declining" if v < 0 else "Improving" for v in show["Change in index"]]
            show["Change in index"] = show["Change in index"].round(1)
            ui.table(show)

    with st.container(border=True):
        ui.section("Machine efficiency profiles", "One row per machine; sort any column")
        t = prof.rename(columns={"index": "Efficiency index", "readings": "Readings", "avg_error": "Avg error %", "avg_speed": "Avg units/hr",
                                 "latest": "Latest class", "Machine_ID": "Machine"})
        for c_ in ("Low", "Medium", "High"):
            t[f"{c_} %"] = (t.pop(c_) * 100).round(1)
        t[["Efficiency index", "Avg error %", "Avg units/hr"]] = t[["Efficiency index", "Avg error %", "Avg units/hr"]].round(1)
        ui.table(t.sort_values("Efficiency index", ascending=False))
