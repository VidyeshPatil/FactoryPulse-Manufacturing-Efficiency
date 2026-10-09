"""Tab 7 - Model lab: comparison, evaluation, reality check, EDA and data quality. Retraining lives here too."""
import pandas as pd
import streamlit as st

from src import charts, data as D, ui
from src.config import C, CLASSES, ENGINEERED, LABEL, RAW_NUMERIC
from src.modeling import train_all


def _retrain(ctx):
    bar = st.progress(0.0, text="Starting")
    train_all(D.training_frame(D.load_base()), progress=lambda msg, f: bar.progress(min(f, 1.0), text=msg))
    st.cache_resource.clear()
    st.session_state.flash = "Models retrained with the latest labelled data."
    st.rerun(scope="app")


@st.fragment
def render(ctx):
    m = ctx.metrics
    models = m["models"]
    prim = next(r for r in models if r["model"] == m["primary"])
    e = m["early"]

    with st.container(border=True):
        ui.section("Reality check: what the data can and cannot predict")
        c = st.columns(3)
        c[0].markdown(ui.kpi("Full model (all inputs)", f"{prim['accuracy']:.2%}", f"macro F1 {prim['macro_f1']:.3f}", C["High"]), unsafe_allow_html=True)
        c[1].markdown(ui.kpi("Sensors + network only", f"AUC {e['roc_auc']:.2f}", "0.50 is a coin flip", C["Low"]), unsafe_allow_html=True)
        c[2].markdown(ui.kpi("Simple 3-line rule", f"{m['rule_agreement']:.3%}", "agreement with the labels", C["sky"]), unsafe_allow_html=True)
        ui.note("The efficiency label is defined by <b>error rate</b> and <b>production speed</b>. The models reach near-perfect scores because they recover that definition. "
                "Temperature, vibration, power, 6G latency and packet loss do not predict the label in this dataset. "
                "Treat the classifier as a real-time labeller for operators, and treat sensor and network data as separate health signals.", "warn")

    a, b = st.columns([1.2, 1])
    with a.container(border=True):
        ui.section("Model comparison", f"Chronological split: first {m['train_rows']:,} readings train, last {m['test_rows']:,} test. Selected: {m['primary']}")
        ui.plot(charts.model_compare(models))
        t = pd.DataFrame([{"Model": r["model"], "Accuracy": r["accuracy"], "Macro F1": r["macro_f1"], "ROC-AUC": r["roc_auc"],
                           "CV F1 (mean)": r["cv_mean"], "CV F1 (std)": r["cv_std"], "Train s": r["train_seconds"],
                           "Interpretability": r["interpretability"], "Selected": "✔" if r["model"] == m["primary"] else ""} for r in models]).round(4)
        ui.table(t)
    with b.container(border=True):
        ui.section("Confusion matrix", f"{m['primary']} on the hold-out period")
        ui.plot(charts.confusion(prim["confusion"]))
        pc = pd.DataFrame(prim["per_class"]).T.reset_index().rename(columns={"index": "Class"})
        ui.table(pc.round(4))

    with st.container(border=True):
        ui.section("Exploratory data analysis", "Filtered by the sidebar")
        df = ctx.fdf
        c1, c2, c3 = st.columns(3)
        with c1:
            ui.plot(charts.donut(df["Efficiency_Status"].value_counts().to_dict(), "Classes"))
            st.caption("Heavy class imbalance: handled with class weights.")
        with c2:
            ui.plot(charts.box_by_class(df, "Error_Rate_%"))
        with c3:
            ui.plot(charts.box_by_class(df, "Production_Speed_units_per_hr"))
        ui.plot(charts.corr_heat(df, RAW_NUMERIC + ["Efficiency_Index"]))

    x, y = st.columns(2)
    with x.container(border=True):
        ui.section("Data quality")
        base = ctx.base
        q = pd.DataFrame({"Check": ["Rows", "Machines", "Missing values", "Exact duplicate rows", "Rows with a repaired clock", "Entries added in app"],
                          "Result": [f"{len(base):,}", base["Machine_ID"].nunique(), int(base[RAW_NUMERIC].isna().sum().sum()),
                                     int(base.duplicated(subset=["Machine_ID", "Datetime"] + RAW_NUMERIC).sum()),
                                     f"{int(base['Clock_Repaired'].sum()):,} (date 01-03-2025 appears twice in the file)", len(ctx.entries)]})
        ui.table(q)
    with y.container(border=True):
        ui.section("Engineered features")
        ui.table(pd.DataFrame({"Feature": list(ENGINEERED), "Meaning": list(ENGINEERED.values())}))

    with st.container(border=True):
        ui.section("Retrain", f"Last trained {m['trained_at']} on {m['trained_rows']:,} rows. New entries join training only when a person supplied the label.")
        if st.button("Retrain all models", key="retrain"):
            _retrain(ctx)
