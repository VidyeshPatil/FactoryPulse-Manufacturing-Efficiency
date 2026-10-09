"""Plotly figure builders. Every function returns a ready-to-render go.Figure."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import C, CLASSES, CLASS_COLORS, GROUP_COLORS, LABEL

FONT = dict(family="Manrope, system-ui, sans-serif", color=C["ink"], size=12)
SCALE = [[0, C["Low"]], [0.5, C["Medium"]], [1, C["High"]]]


def _base(fig: go.Figure, h: int = 320, legend: bool = True, margin=(10, 10, 34, 10)) -> go.Figure:
    l, r, t, b = margin
    fig.update_layout(height=h, margin=dict(l=l, r=r, t=t, b=b), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=FONT, showlegend=legend,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#222633", zeroline=False, linecolor="#2A2F3B")
    fig.update_yaxes(gridcolor="#222633", zeroline=False, linecolor="#2A2F3B")
    return fig


def _title(fig: go.Figure, text: str) -> go.Figure:
    fig.update_layout(title=dict(text=text, x=0, font=dict(size=14, family="Sora, Manrope, sans-serif")))
    return fig


# ------------------------------------------------------------- live monitor ---
def gauge(value: float, title: str, ref: float | None = None, color: str = C["teal"], suffix: str = "", rng=(0, 100)) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if ref is not None else ""), value=value,
        delta={"reference": ref, "increasing": {"color": C["High"]}, "decreasing": {"color": C["Low"]}} if ref is not None else None,
        number={"suffix": suffix, "font": {"size": 38, "family": "Sora, sans-serif"}},
        title={"text": title, "font": {"size": 14}},
        gauge={"axis": {"range": list(rng), "tickwidth": 0, "tickfont": {"size": 10}}, "bar": {"color": color, "thickness": 0.3},
               "bgcolor": "#232733", "borderwidth": 0}))
    return _base(fig, 215, legend=False, margin=(18, 18, 30, 4))


def live_trend(w: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = w["Datetime"]
    fig.add_trace(go.Scatter(x=x, y=w["Error_Rate_%"].rolling(8, min_periods=1).mean(), name="Error rate % (8-reading avg)",
                             line=dict(color=C["Low"], width=2.5), fill="tozeroy", fillcolor="rgba(255,92,108,.12)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=w["Production_Speed_units_per_hr"].rolling(8, min_periods=1).mean(), name="Speed units/hr (8-reading avg)",
                             line=dict(color=C["teal"], width=2.5)), secondary_y=True)
    if len(x):
        fig.add_trace(go.Scatter(x=[x.iloc[0], x.iloc[-1]], y=[5, 5], name="Error limit 5 %", mode="lines",
                                 line=dict(color=C["Low"], width=1.2, dash="dot")), secondary_y=False)
    fig.update_yaxes(title_text="Error rate %", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="Units / hr", secondary_y=True, showgrid=False)
    return _base(fig, 300)


def class_strip(w: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=w["Datetime"], y=np.ones(len(w)), marker_color=[CLASS_COLORS[c] for c in w["Pred"]],
                           customdata=np.c_[w["Machine_ID"], w["Pred"], (w["Confidence"] * 100).round(0)],
                           hovertemplate="%{x|%H:%M}  machine %{customdata[0]}<br>%{customdata[1]} (%{customdata[2]}% sure)<extra></extra>",
                           marker_line_width=0))
    fig.update_layout(bargap=0)
    fig.update_yaxes(visible=False)
    return _base(fig, 92, legend=False, margin=(10, 10, 4, 10))


def machine_grid(latest: pd.DataFrame, machine_ids: list[int], cols: int = 10) -> go.Figure:
    rows = math.ceil(len(machine_ids) / cols)
    z = np.full((rows, cols), np.nan)
    txt = [[""] * cols for _ in range(rows)]
    hov = [[""] * cols for _ in range(rows)]
    cmap = {r.Machine_ID: r for r in latest.itertuples()}
    for i, mid in enumerate(machine_ids):
        r, c = divmod(i, cols)
        txt[r][c] = str(mid)
        rec = cmap.get(mid)
        if rec is not None:
            z[r][c] = {"Low": 0, "Medium": 1, "High": 2}[rec.Pred]
            hov[r][c] = f"Machine {mid}: {rec.Pred} ({rec.Confidence:.0%} sure)"
        else:
            hov[r][c] = f"Machine {mid}: no reading in this window"
    scale = [[0, C["Low"]], [0.333, C["Low"]], [0.333, C["Medium"]], [0.667, C["Medium"]], [0.667, C["High"]], [1, C["High"]]]
    fig = go.Figure(go.Heatmap(z=z, text=txt, texttemplate="%{text}", textfont=dict(color="white", size=11), colorscale=scale, zmin=0, zmax=2,
                               xgap=3, ygap=3, showscale=False, hoverongaps=False, hovertext=hov, hoverinfo="text"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, autorange="reversed", scaleanchor="x")
    return _base(fig, 240, legend=False, margin=(4, 4, 4, 4))


def donut(counts: dict, title: str = "", pct: bool = True) -> go.Figure:
    labels = [c for c in CLASSES if c in counts]
    fig = go.Figure(go.Pie(labels=labels, values=[counts[c] for c in labels], hole=0.62, sort=False,
                           marker=dict(colors=[CLASS_COLORS[c] for c in labels], line=dict(color=C["card"], width=2)),
                           textinfo="percent" if pct else "none", textfont=dict(color="white", size=12)))
    _base(fig, 235, margin=(4, 4, 4, 4))
    fig.update_layout(legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center", yanchor="top"))
    if title:
        fig.add_annotation(text=title, x=0.5, y=0.5, showarrow=False, font=dict(size=13, family="Sora, sans-serif"))
    return fig


# ---------------------------------------------------------------- predict ----
def proba_bars(p) -> go.Figure:
    fig = go.Figure(go.Bar(y=CLASSES, x=[float(v) for v in p], orientation="h", marker_color=[CLASS_COLORS[c] for c in CLASSES],
                           text=[f"{v:.1%}" for v in p], textposition="outside", cliponaxis=False))
    fig.update_xaxes(range=[0, 1.18], tickformat=".0%")
    return _base(fig, 170, legend=False, margin=(10, 40, 4, 10))


def sens_curves(x, P, current: float, label: str) -> go.Figure:
    fig = go.Figure()
    for k, c in enumerate(CLASSES):
        fig.add_trace(go.Scatter(x=x, y=P[:, k], name=c, line=dict(color=CLASS_COLORS[c], width=3), mode="lines"))
    fig.add_vline(x=current, line_dash="dot", line_color=C["ink"], annotation_text="now", annotation_position="top")
    fig.update_yaxes(tickformat=".0%", range=[-0.02, 1.02], title_text="Probability")
    fig.update_xaxes(title_text=label)
    return _base(fig, 280)


def tornado(df: pd.DataFrame, title: str) -> go.Figure:
    d = df.sort_values("swing")
    fig = go.Figure(go.Bar(y=d["label"], x=d["swing"], orientation="h", marker_color=C["teal"],
                           text=[f"{v:.0%}" for v in d["swing"]], textposition="outside", cliponaxis=False))
    fig.update_xaxes(tickformat=".0%", range=[0, max(1.0, float(d["swing"].max()) * 1.15)])
    _title(fig, title)
    return _base(fig, 330, legend=False, margin=(10, 40, 36, 10))


# ------------------------------------------------------------ explainability ---
def importance_bar(imp: pd.DataFrame, title: str = "") -> go.Figure:
    d = imp.sort_values("share")
    fig = go.Figure(go.Bar(y=d["label"], x=d["share"], orientation="h", marker_color=[GROUP_COLORS.get(g, C["muted"]) for g in d["group"]],
                           text=[f"{v:.1%}" for v in d["share"]], textposition="outside", cliponaxis=False,
                           customdata=d["group"], hovertemplate="%{y} (%{customdata}): %{x:.1%}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", range=[0, max(0.1, float(d["share"].max()) * 1.18)])
    if title:
        _title(fig, title)
    return _base(fig, 340, legend=False, margin=(10, 50, 36 if title else 6, 10))


def contribution_bar(df: pd.DataFrame, title: str, x_title: str = "Effect") -> go.Figure:
    d = df.iloc[::-1] if "contribution" in df else df
    col = "contribution" if "contribution" in d else "impact"
    d = d.sort_values(col, key=abs)
    fig = go.Figure(go.Bar(y=d["label"], x=d[col], orientation="h", marker_color=[C["teal"] if v >= 0 else C["Low"] for v in d[col]],
                           text=[f"{v:+.2f}" if col == "impact" else f"{v:+.0%}" for v in d[col]], textposition="outside", cliponaxis=False))
    fig.update_xaxes(title_text=x_title, zeroline=True, zerolinecolor=C["muted"])
    _title(fig, title)
    return _base(fig, 340, legend=False, margin=(10, 50, 36, 10))


def zone_map(df: pd.DataFrame) -> go.Figure:
    """Observed efficiency (0-100) for every error-rate × production-speed cell: shows the decision zones."""
    sp = pd.cut(df["Production_Speed_units_per_hr"], np.arange(50, 525, 25), right=False)
    er = pd.cut(df["Error_Rate_%"], np.arange(0, 16, 1), right=False)
    g = df.assign(idx=df["Efficiency_Status"].map({"Low": 0, "Medium": 50, "High": 100})).groupby([er, sp], observed=False)["idx"].mean().unstack()
    fig = go.Figure(go.Heatmap(z=g.values, x=[f"{int(i.left)}" for i in g.columns], y=[f"{i.left:.0f}" for i in g.index], colorscale=SCALE,
                               zmin=0, zmax=100, colorbar=dict(title="Efficiency", thickness=12), xgap=1, ygap=1,
                               hovertemplate="speed ≥ %{x}, error ≥ %{y} %<br>observed efficiency %{z:.0f}<extra></extra>"))
    fig.update_xaxes(title_text="Production speed (units/hr)")
    fig.update_yaxes(title_text="Error rate (%)", autorange="reversed")
    return _base(fig, 330, legend=False)


# ------------------------------------------------------------- machine view ---
def machine_trend(daily: pd.DataFrame, name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=daily["Date_Str"], y=daily["Efficiency_Index"], name="Daily index", marker_color="rgba(255,122,61,.35)"))
    fig.add_trace(go.Scatter(x=daily["Date_Str"], y=daily["Efficiency_Index"].rolling(7, min_periods=1).mean(), name="7-day average",
                             line=dict(color=C["deep"], width=3)))
    fig.update_yaxes(title_text="Efficiency index (0-100)", rangemode="tozero")
    _title(fig, name)
    return _base(fig, 300, margin=(10, 10, 44, 10))


def share_stack(df: pd.DataFrame, by: str, title: str = "", order: list | None = None) -> go.Figure:
    t = pd.crosstab(df[by], df["Efficiency_Status"], normalize="index").reindex(columns=CLASSES, fill_value=0)
    if order:
        t = t.reindex([o for o in order if o in t.index])
    fig = go.Figure()
    for c in CLASSES:
        fig.add_trace(go.Bar(x=t.index.astype(str), y=t[c], name=c, marker_color=CLASS_COLORS[c], text=[f"{v:.0%}" for v in t[c]],
                             textposition="inside", textfont=dict(color="white")))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(tickformat=".0%")
    if title:
        _title(fig, title)
    return _base(fig, 310, margin=(10, 10, 44 if title else 34, 10))


def leaderboard(m: pd.DataFrame, highlight: int | None = None) -> go.Figure:
    d = m.sort_values("index", ascending=False)
    colors = [C["deep"] if (highlight is not None and i == highlight) else C["teal"] for i in d["Machine_ID"]]
    fig = go.Figure(go.Bar(x=d["Machine_ID"].astype(str), y=d["index"], marker_color=colors,
                           hovertemplate="Machine %{x}: index %{y:.1f}<extra></extra>"))
    fig.update_xaxes(type="category", title_text="Machine (ranked)")
    fig.update_yaxes(title_text="Avg efficiency index")
    return _base(fig, 280, legend=False)


def hour_heat(df: pd.DataFrame) -> go.Figure:
    d = df.assign(dow=df["Datetime"].dt.dayofweek).pivot_table(index="Hour", columns="dow", values="Efficiency_Index", aggfunc="mean")
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    d = d.reindex(columns=range(7))
    fig = go.Figure(go.Heatmap(z=d.values, x=days, y=d.index, colorscale=SCALE, colorbar=dict(title="Index", thickness=12), xgap=2, ygap=2,
                               hovertemplate="%{x} %{y}:00 → %{z:.1f}<extra></extra>"))
    fig.update_yaxes(title_text="Hour of day", autorange="reversed")
    return _base(fig, 330, legend=False)


# ---------------------------------------------------------------- operations ---
def quartile_lines(df: pd.DataFrame, cols: dict[str, str]) -> go.Figure:
    """Average efficiency index across quartiles of each driver (flat lines = no influence)."""
    fig = go.Figure()
    for col, color in cols.items():
        q = pd.qcut(df[col].rank(method="first"), 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
        m = df.groupby(q, observed=True)["Efficiency_Index"].mean()
        fig.add_trace(go.Scatter(x=m.index.astype(str), y=m.values, name=LABEL.get(col, col.replace("_", " ")), mode="lines+markers",
                                 line=dict(color=color, width=3), marker=dict(size=8)))
    fig.update_yaxes(title_text="Avg efficiency index", rangemode="tozero")
    return _base(fig, 320)


def corr_bar(corr: pd.Series, group_of: dict) -> go.Figure:
    d = corr.sort_values(key=abs)
    fig = go.Figure(go.Bar(y=[LABEL.get(i, i.replace("_", " ")) for i in d.index], x=d.values, orientation="h",
                           marker_color=[GROUP_COLORS.get(group_of.get(i, "Mode"), C["muted"]) for i in d.index],
                           text=[f"{v:+.2f}" for v in d.values], textposition="outside", cliponaxis=False))
    fig.update_xaxes(range=[-1, 1], title_text="Correlation with efficiency index", zeroline=True, zerolinecolor=C["muted"])
    return _base(fig, 330, legend=False, margin=(10, 40, 6, 10))


def group_share(shares: dict) -> go.Figure:
    labels = list(shares)
    fig = go.Figure(go.Bar(x=labels, y=[shares[k] for k in labels], marker_color=[GROUP_COLORS.get(k, C["muted"]) for k in labels],
                           text=[f"{shares[k]:.1%}" for k in labels], textposition="outside", cliponaxis=False))
    fig.update_yaxes(tickformat=".0%", range=[0, 1.12])
    return _base(fig, 300, legend=False)


# ------------------------------------------------------------------ model lab ---
def confusion(cm) -> go.Figure:
    cm = np.array(cm)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig = go.Figure(go.Heatmap(z=norm, x=CLASSES, y=CLASSES, colorscale=[[0, "#1B1E27"], [1, C["teal"]]], showscale=False, zmin=0, zmax=1,
                               text=[[f"{cm[i][j]:,}<br>{norm[i][j]:.1%}" for j in range(3)] for i in range(3)], texttemplate="%{text}",
                               xgap=3, ygap=3))
    fig.update_xaxes(title_text="Predicted", side="bottom")
    fig.update_yaxes(title_text="Actual", autorange="reversed")
    return _base(fig, 320, legend=False)


def model_compare(models: list[dict]) -> go.Figure:
    names = [m["model"] for m in models]
    fig = go.Figure()
    for key, label, color in [("accuracy", "Accuracy", C["teal"]), ("macro_f1", "Macro F1", C["sky"]), ("cv_mean", "CV macro F1", C["violet"])]:
        fig.add_trace(go.Bar(x=names, y=[m[key] for m in models], name=label, marker_color=color,
                             text=[f"{m[key]:.3f}" for m in models], textposition="outside", cliponaxis=False))
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0.75, 1.03])
    return _base(fig, 320, margin=(10, 10, 40, 10))


def box_by_class(df: pd.DataFrame, col: str) -> go.Figure:
    fig = go.Figure()
    for c in CLASSES:
        fig.add_trace(go.Box(y=df.loc[df["Efficiency_Status"] == c, col], name=c, marker_color=CLASS_COLORS[c], boxpoints=False))
    fig.update_yaxes(title_text=LABEL.get(col, col))
    return _base(fig, 300, legend=False)


def corr_heat(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    cm = df[cols].corr()
    fig = go.Figure(go.Heatmap(z=cm.values, x=[LABEL.get(c, c) for c in cols], y=[LABEL.get(c, c) for c in cols], zmin=-1, zmax=1,
                               colorscale=[[0, C["Low"]], [0.5, "#1B1E27"], [1, C["teal"]]], colorbar=dict(thickness=12),
                               text=np.round(cm.values, 2), texttemplate="%{text}", textfont=dict(size=10)))
    fig.update_yaxes(autorange="reversed")
    return _base(fig, 420, legend=False)
