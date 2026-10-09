"""Tab 1 - Live efficiency monitor: traffic-light status, incoming readings, rolling mix, then a deep dive."""
import pandas as pd
import streamlit as st

from src import charts, filters, ui
from src.config import ALERTS, C, CLASS_COLORS, CLASSES

RECENT = 60                                   # readings behind the fleet-level numbers in the deep dive
SPEEDS = {"Slow": 1, "Normal": 5, "Fast": 30}
STATE_TEXT = {"Low": "needs attention", "Medium": "keep an eye on it", "High": "running well"}


def _toggle():
    st.session_state.playing = not st.session_state.playing


def _speed(n: int):
    st.session_state.rows_per_tick = n
    st.session_state.playing = True


def _restart():
    st.session_state.cursor = 600


def _latest():
    st.session_state.cursor = st.session_state.get("n_rows", 100_000) - 1


def _hide_intro():
    st.session_state.show_intro = False


def render(ctx):
    ss = st.session_state
    ss.setdefault("show_intro", True)
    ss.setdefault("session_readings", 0)
    ui.page_title("Live Efficiency Monitor", "Real-time simulated feed of incoming machine readings, classified the instant they arrive")
    if ss.show_intro:
        a, b = st.columns([0.965, 0.035])
        a.markdown(
            "<div class='intro'><div class='ic'>ⓘ</div><div><b class='h'>New here? Here's the 10-second version</b>"
            f"<p>{ctx.app_name} watches <b>{ctx.n_machines} factory machines</b> and tells you, reading by reading, whether each one is running "
            "<span class='c-high'>High</span>, <span class='c-med'>Medium</span>, or <span class='c-low'>Low</span> efficiency, using an AI model trained on real "
            "sensor and network data. Green / yellow / red = good / watch it / needs attention, just like a traffic light. "
            "Hover over the small <i class='info'>i</i> icons next to technical terms for a plain-English explanation.</p></div></div>", unsafe_allow_html=True)
        b.button("✕", on_click=_hide_intro, key="intro_x")
    _body(ctx)


@st.fragment(run_every=2)
def _body(ctx):
    ss = st.session_state
    base, n = ctx.base, len(ctx.base)
    ss.n_rows = n
    if ss.playing:
        ss.cursor += ss.rows_per_tick
        ss.session_readings += ss.rows_per_tick
    if ss.cursor >= n:
        ss.cursor = ss.window if ss.playing else n
    cur = int(min(max(ss.cursor, 50), n))
    ss.cursor = cur

    w0 = max(0, cur - ss.window)
    wb = base.iloc[w0:cur]
    t0, t1 = wb["Datetime"].iloc[0], wb["Datetime"].iloc[-1]
    ss.clock = t1
    w = wb
    if len(ctx.entries):
        we = ctx.entries[(ctx.entries["Datetime"] >= t0) & (ctx.entries["Datetime"] <= t1)]
        if len(we):
            w = pd.concat([wb, we]).sort_values("Datetime", kind="stable")
    w = filters.apply(w, ctx.F, time=False)
    if w.empty:
        ui.note("No readings match the current filters in this window. Open the filter panel (arrow at the top left) and widen them.", "warn")
        return

    last = w.iloc[-1]
    cls = last["Pred"]
    last20 = w.tail(20)
    pct_low = (last20["Pred"] == "Low").mean() * 100
    alert_color = C["Low"] if pct_low >= 60 else C["Medium"] if pct_low >= 30 else C["High"]

    # ------------------------------------------------------------ top row
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        lights = "".join(f"<i class='{k}{' on' if cls == v else ''}'></i>" for k, v in (("r", "Low"), ("y", "Medium"), ("g", "High")))
        live = "<span class='live-dot'></span>LIVE FEED" if ss.playing else "<span class='paused-dot'></span>PAUSED"
        st.markdown(
            f"<div style='display:flex;gap:18px;align-items:center'><div style='flex:0 0 140px'><div class='tl'>{lights}</div></div>"
            f"<div style='flex:1'><div class='live-tag'>{live}</div>"
            f"<div class='tlstate'><span class='w' style='color:{CLASS_COLORS[cls]}'>{cls}</span>"
            f"<span class='tag' style='color:{CLASS_COLORS[cls]}'>· {STATE_TEXT[cls]}</span></div>"
            f"<div class='tlmeta'>Machine M-{int(last['Machine_ID'])} · {last['Confidence']:.1%} confidence · recorded {last['Datetime']:%d %b %H:%M}</div>"
            "</div></div>", unsafe_allow_html=True)
        b = st.columns([1.15, 1.3, 0.9, 1.1, 0.9])
        b[1].button("⏸ Pause" if ss.playing else "▶ Play", on_click=_toggle, type="primary" if ss.playing else "secondary", key="btn_play")
        for col, (name, rows) in zip(b[2:], SPEEDS.items()):
            col.button(name, on_click=_speed, args=(rows,), type="primary" if ss.rows_per_tick == rows and ss.playing else "secondary", key=f"spd_{name}")
    c2.markdown(ui.kpi("Live session readings", f"{ss.session_readings:,}", "Since this page was opened",
                       tip="How many machine readings have been classified since you opened this page."), unsafe_allow_html=True)
    c3.markdown(ui.kpi("Current alert level", f"{pct_low:.0f}%", "% Low efficiency, last 20 readings", vcolor=alert_color,
                       tip="The share of the latest 20 readings the AI rated Low. Under 30% is healthy, 30-60% needs watching, above 60% needs attention."),
                unsafe_allow_html=True)

    # --------------------------------------------------- readings + class mix
    r1, r2 = st.columns([2, 1])
    with r1.container(border=True):
        ui.section("Incoming readings", "Newest first · streamed from the factory dataset and any readings you add",
                   tip="Each row is one machine reading. The colour is the AI's verdict and the percentage is how sure it is.")
        rows = "".join(
            "<div class='rrow'><span class='d' style='background:%s'></span><span class='m'>M-%02d</span>%s<span class='t'>%s</span><span class='c'>%.1f%%</span></div>" % (
                CLASS_COLORS[r.Pred], r.Machine_ID, ui.pill(r.Pred, "<span class='pill New'>NEW</span>" if r.Source == "Manual" else ""),
                r.Datetime.strftime("%d %b %H:%M"), r.Confidence * 100)
            for r in w.tail(30).iloc[::-1].itertuples())
        st.markdown(f"<div class='rlist'>{rows}</div>", unsafe_allow_html=True)
    with r2.container(border=True):
        ui.section("Rolling class mix", "Last 30 live readings", tip="How the most recent 30 readings split between High, Medium and Low.")
        mix = w.tail(30)["Pred"].value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict()
        ui.plot(charts.donut(mix, pct=False))

    # ----------------------------------------------------------- deep dive
    st.markdown("<div class='ptitle' style='font-size:1.45rem;margin-top:1.4rem'>Deep dive</div>", unsafe_allow_html=True)
    cc = st.columns([1, 1, 2, 4])
    cc[0].button("⏮ Restart", on_click=_restart, key="btn_restart")
    cc[1].button("⏭ Latest", on_click=_latest, key="btn_latest")
    cc[2].select_slider("Window (readings)", options=[120, 240, 480, 960], key="window")
    cc[3].markdown(f"<div class='tlmeta' style='text-align:left;padding-top:26px'>Replay clock {t1:%d %b %Y, %H:%M} · reading {cur:,} of {n:,}</div>",
                   unsafe_allow_html=True)
    st.progress(cur / n)

    rec, prev = w.tail(RECENT), w.iloc[-2 * RECENT:-RECENT]
    idx = rec["Efficiency_Index"].mean()
    d_idx = idx - prev["Efficiency_Index"].mean() if len(prev) >= 10 else 0.0
    k = st.columns(4)
    k[0].markdown(ui.kpi("Efficiency index", f"{idx:.1f}", ui.delta_html(d_idx, " pts") + " vs previous 60", C["teal"],
                         tip="A 0-100 score: 0 means every reading is Low, 100 means every reading is High."), unsafe_allow_html=True)
    sp = rec["Production_Speed_units_per_hr"].mean()
    sp0 = prev["Production_Speed_units_per_hr"].mean() if len(prev) else sp
    k[1].markdown(ui.kpi("Output speed", f"{sp:.0f}", ui.delta_html(sp - sp0, " u/hr", True, 0) + " units per hour", C["Medium"],
                         tip="Average units produced per hour. Above 400 is needed for High efficiency."), unsafe_allow_html=True)
    er = rec["Error_Rate_%"].mean()
    er0 = prev["Error_Rate_%"].mean() if len(prev) else er
    k[2].markdown(ui.kpi("Error rate", f"{er:.1f}%", ui.delta_html(er - er0, " pp", False) + " vs previous 60", C["Low"],
                         tip="Share of operations that went wrong. Above 5% always means Low efficiency."), unsafe_allow_html=True)
    k[3].markdown(ui.kpi("6G reliability", f"{rec['Network_Reliability'].mean():.0f}/100",
                         f"latency {rec['Network_Latency_ms'].mean():.0f} ms · loss {rec['Packet_Loss_%'].mean():.1f}%", C["sky"],
                         tip="Combines network delay and lost data packets into one score. Higher is better."), unsafe_allow_html=True)

    a, b2 = st.columns([1, 2.1])
    with a.container(border=True):
        ui.section("Fleet efficiency", "Index of the last 60 readings", tip="The needle shows the average efficiency index; the small arrow compares it to the 60 readings before.")
        ui.plot(charts.gauge(idx, "Efficiency index", ref=prev["Efficiency_Index"].mean() if len(prev) >= 10 else None, color=C["teal"]))
    with b2.container(border=True):
        ui.section("Error rate and output speed", "Smoothed over 8 readings · dotted line = 5% error limit")
        ui.plot(charts.live_trend(w))
        ui.section("Class of every reading", "One bar per minute, coloured by the AI's verdict")
        ui.plot(charts.class_strip(w))

    g, f = st.columns([1, 1.4])
    with g.container(border=True):
        ui.section("Machine status board", "Latest class per machine in this window", tip="Each tile is a machine. Grey tiles have not reported in this window yet.")
        ids = sorted(ctx.all["Machine_ID"].unique().tolist())
        ui.plot(charts.machine_grid(w.groupby("Machine_ID").tail(1), ids))
        st.markdown("<span class='pill High'>High</span> <span class='pill Medium'>Medium</span> <span class='pill Low'>Low</span>", unsafe_allow_html=True)
    with f.container(border=True):
        ui.section("Alerts", "Threshold breaches in the last 120 readings", tip="Early-warning checks on error rate, packet loss, latency, vibration, temperature and maintenance score.")
        ev = []
        tail = w.tail(120)
        for colname, (op, lim, msg) in ALERTS.items():
            hit = tail[tail[colname] > lim] if op == ">" else tail[tail[colname] < lim]
            ev.extend((t, int(m), msg) for t, m in zip(hit["Datetime"], hit["Machine_ID"]))
        ev.sort(key=lambda x: x[0], reverse=True)
        if not ev:
            st.markdown("<div class='alert-row ok'>✅ No threshold breaches in the recent window.</div>", unsafe_allow_html=True)
        else:
            st.caption(f"{len(ev)} breach(es) · showing the newest 6")
            for t, m, msg in ev[:6]:
                st.markdown(f"<div class='alert-row'><b>{t:%H:%M}</b><span>Machine M-{m}: {msg}</span></div>", unsafe_allow_html=True)
