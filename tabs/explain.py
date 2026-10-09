"""Tab 4 - Explainability panel: global drivers, single-reading reasons, and why efficiency changed."""
import numpy as np
import pandas as pd
import streamlit as st

from src import charts, narrative, ui
from src.config import C, CLASS_COLORS, CLASSES, GROUP, LABEL, RAW_NUMERIC, UNIT
from src.modeling import attribute_shift, efficiency_index, explain_local


def _bump():
    st.session_state.ex_seed = st.session_state.get("ex_seed", 0) + 1


@st.fragment
def render(ctx):
    t1, t2, t3 = st.tabs(["Global drivers", "Why this reading?", "Why did efficiency change?"])
    model, ss = ctx.primary, st.session_state

    with t1:
        imp = pd.DataFrame(ctx.metrics["importance"])
        a, b = st.columns([1.2, 1])
        with a.container(border=True):
            ui.section("What drives the efficiency class", "Share of model accuracy lost when a metric is shuffled (permutation importance, hold-out data)")
            ui.plot(charts.importance_bar(imp))
        with b.container(border=True):
            ui.section("Rules the model learned", "A small decision tree built on the top two drivers")
            r = ctx.metrics["rule_agreement"]
            st.markdown(
                f"- **Low**: error rate above 5 %, or speed at or below 200 units/hr\n"
                f"- **High**: speed above 400 units/hr and error rate at or below 2 %\n"
                f"- **Medium**: everything in between\n\n"
                f"These three lines reproduce the labels on **{r:.3%}** of all rows.")
            ui.note("Sensor, 6G network and maintenance metrics carry almost no weight, so the label follows output quality, not machine condition.", "warn")
        with st.container(border=True):
            ui.section("Efficiency zone map", "Observed average efficiency for every error-rate × speed cell; the three zones are visible by eye")
            ui.plot(charts.zone_map(ctx.all))

    with t2:
        with st.container(border=True):
            ui.section("Explain one reading", "How much each metric pushed the prediction toward the chosen class")
            c = st.columns([2, 1])
            src = c[0].selectbox("Reading to explain", ["Latest live reading", "A Low reading", "A Medium reading", "A High reading"], key="ex_src")
            c[1].write("")
            c[1].button("Pick another", on_click=_bump, key="ex_bump")
            pool = ctx.fdf if len(ctx.fdf) else ctx.all
            if src == "Latest live reading":
                row = ctx.base.iloc[[min(ss.cursor, len(ctx.base)) - 1]]
            else:
                cls = src.split()[1]
                sub = pool[pool["Efficiency_Status"] == cls]
                row = sub.sample(1, random_state=ss.get("ex_seed", 0)) if len(sub) else None
            if row is None:
                st.info("No reading of that class in the filtered data.")
            else:
                raw = row[RAW_NUMERIC + ["Operation_Mode"]].reset_index(drop=True)
                tgt, contrib = explain_local(model, raw, ctx.base.sample(400, random_state=1))
                d = raw.iloc[0].to_dict()
                x, y = st.columns([1, 1])
                with x:
                    st.markdown(f"<div class='bigstate' style='color:{CLASS_COLORS[tgt]}'>{tgt}</div><div class='s'>Machine {int(row['Machine_ID'].iloc[0])} · "
                                f"{row['Datetime'].iloc[0]:%d %b %H:%M} · {model.proba(raw)[0].max():.0%} sure</div>", unsafe_allow_html=True)
                    st.write("")
                    for line in narrative.reasons(d, tgt):
                        st.markdown(f"- {line}")
                    ui.note("💡 " + narrative.next_step(d, tgt))
                    vals = pd.DataFrame({"Metric": [LABEL.get(k, "Operation mode") for k in RAW_NUMERIC] + ["Operation mode"],
                                         "Value": [f"{d[k]:.2f} {UNIT.get(k, '')}".strip() for k in RAW_NUMERIC] + [d["Operation_Mode"]]})
                    ui.table(vals)
                with y:
                    ui.plot(charts.contribution_bar(contrib, f"Push toward {tgt} (probability points)", "Change in P(" + tgt + ")"))

    with t3:
        with st.container(border=True):
            ui.section("Compare two consecutive windows", "The newest N readings against the N before them")
            pool = ctx.fdf
            c = st.columns(3)
            ids = ["All machines"] + sorted(pool["Machine_ID"].unique().tolist())
            who = c[0].selectbox("Scope", ids, key="ch_who")
            nwin = c[1].slider("Readings per window", 50, 1000, 300, 50, key="ch_n")
            asof = c[2].radio("As of", ["Live clock", "End of data"], horizontal=True, key="ch_asof")
            if who != "All machines":
                pool = pool[pool["Machine_ID"] == who]
            if asof == "Live clock":
                pool = pool[pool["Datetime"] <= ss.get("clock", pool["Datetime"].max())]
            pool = pool.sort_values("Datetime", kind="stable")
            if len(pool) < 2 * nwin:
                st.info(f"Need at least {2 * nwin} readings; only {len(pool)} match. Lower the window size or relax the filters.")
            else:
                before, after = pool.iloc[-2 * nwin:-nwin], pool.iloc[-nwin:]
                i0, i1 = before["Efficiency_Index"].mean(), after["Efficiency_Index"].mean()
                verb = "improved" if i1 > i0 else "dropped"
                att = attribute_shift(model, before[RAW_NUMERIC + ["Operation_Mode"]].reset_index(drop=True),
                                      after[RAW_NUMERIC + ["Operation_Mode"]].reset_index(drop=True))
                top = att.iloc[0]
                big = abs(i1 - i0) >= 1
                ui.note(f"Efficiency {verb} from <b>{i0:.1f}</b> to <b>{i1:.1f}</b> ({i1 - i0:+.1f} points)."
                        + (f" The biggest contributor is <b>{top['label']}</b> ({top['impact']:+.2f} points)." if big else " The change is within normal noise."),
                        "" if i1 >= i0 else "bad")
                x, y = st.columns(2)
                with x:
                    ui.plot(charts.contribution_bar(att, "Contribution to the change (index points)", "Index points"))
                with y:
                    cmp = pd.DataFrame({"Metric": [LABEL[k] for k in RAW_NUMERIC], "Before": [before[k].mean() for k in RAW_NUMERIC],
                                        "After": [after[k].mean() for k in RAW_NUMERIC]})
                    cmp["Change"] = cmp["After"] - cmp["Before"]
                    ui.table(cmp.round(2))
                    mix = pd.DataFrame({"Class": CLASSES, "Before %": [(before["Efficiency_Status"] == k).mean() * 100 for k in CLASSES],
                                        "After %": [(after["Efficiency_Status"] == k).mean() * 100 for k in CLASSES]}).round(1)
                    ui.table(mix)
