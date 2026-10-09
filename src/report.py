"""Executive-summary generator (Markdown + printable HTML) built from live data and model metrics."""
from __future__ import annotations

import html

import pandas as pd

from .config import APP_NAME, CLASSES, LABEL


def build(ctx) -> tuple[str, str]:
    df, m = ctx.fdf, ctx.metrics
    if df.empty:
        return "No data in the current filter.", "<p>No data in the current filter.</p>"
    n = len(df)
    share = df["Efficiency_Status"].value_counts(normalize=True).reindex(CLASSES).fillna(0)
    idx = df["Efficiency_Index"].mean()
    prof = df.groupby("Machine_ID")["Efficiency_Index"].mean().sort_values()
    modes = df.groupby("Operation_Mode")["Efficiency_Index"].mean()
    imp = pd.DataFrame(m["importance"])
    top = imp.head(2)
    pm = next(r for r in m["models"] if r["model"] == m["primary"])
    e = m["early"]
    lowest = ", ".join(f"{i} ({v:.1f})" for i, v in prof.head(3).items())
    highest = ", ".join(f"{i} ({v:.1f})" for i, v in prof.tail(3).iloc[::-1].items())
    lines = [
        ("Situation", f"{n:,} readings from {df['Machine_ID'].nunique()} machines were reviewed. {share['Low']:.1%} were classified Low, "
                      f"{share['Medium']:.1%} Medium and {share['High']:.1%} High. The average efficiency index is {idx:.1f} out of 100."),
        ("What drives efficiency", f"{top.iloc[0]['label']} and {top.iloc[1]['label']} together explain {top['share'].sum():.0%} of the model's decisions. "
                                   f"A Low status means errors above 5 % or speed at or below 200 units/hr; High needs speed above 400 units/hr with errors at or below 2 %."),
        ("Sensors and 6G network", f"Sensor and network metrics have almost no influence on the label here. A model that uses only those inputs scores ROC-AUC {e['roc_auc']:.2f}, "
                                   f"which is chance level. They remain useful as health alerts, not as efficiency predictors."),
        ("Where to act", f"Lowest-ranked machines: {lowest}. Best performers: {highest}. "
                         f"Operation mode makes little difference (range {modes.min():.1f} to {modes.max():.1f})."),
        ("AI model", f"{m['primary']} classifies a held-out, later period with {pm['accuracy']:.2%} accuracy and macro F1 {pm['macro_f1']:.3f}. "
                     f"Each prediction carries a confidence score and a plain-language reason."),
    ]
    recs = [
        "Target the error rate first: keeping it under 5 % is the single biggest lever for leaving the Low class.",
        "Raise throughput on machines stuck between 200 and 400 units/hr; High needs both speed above 400 and errors at or below 2 %.",
        "Use the live monitor's alerts for packet loss, latency, vibration, temperature and maintenance score as machine-health signals.",
        "Collect labelled readings from real operations so the model can be retrained and its confidence checked against ground truth.",
    ]
    risks = [
        "Labels in the source data follow a rule based on error rate and speed, so high accuracy shows the rule is learnable, not that sensors predict efficiency.",
        "Data is a one-minute replay of 50 machines; one calendar day (01-03-2025) appears twice in the export and the clock was rebuilt in order.",
        "Efficiency classes are imbalanced (about 78 % Low); class weighting was used and macro F1 is reported.",
    ]
    md = f"# {APP_NAME}\n## Executive summary\n\n" + "\n\n".join(f"**{h}.** {t}" for h, t in lines)
    md += "\n\n### Recommendations\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(recs, 1))
    md += "\n\n### Limits to keep in mind\n" + "\n".join(f"- {r}" for r in risks)
    body = "".join(f"<p><b>{html.escape(h)}.</b> {html.escape(t)}</p>" for h, t in lines)
    body += "<h3>Recommendations</h3><ol>" + "".join(f"<li>{html.escape(r)}</li>" for r in recs) + "</ol>"
    body += "<h3>Limits to keep in mind</h3><ul>" + "".join(f"<li>{html.escape(r)}</li>" for r in risks) + "</ul>"
    page = ("<!doctype html><meta charset='utf-8'><title>Executive summary</title><style>body{font:16px/1.6 Segoe UI,Arial,sans-serif;max-width:780px;"
            "margin:40px auto;color:#0B2233;padding:0 20px}h1{color:#0E3B4A}h3{color:#0FB5A6}</style>"
            f"<h1>{html.escape(APP_NAME)}</h1><h2>Executive summary</h2>{body}")
    return md, page
