"""Streamlit styling and small UI helpers (kept version-tolerant on purpose)."""
from __future__ import annotations

import base64
import html
import inspect

import streamlit as st

from .config import ASSETS, C, CLASS_COLORS

FONTS = ("@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700"
         "&family=JetBrains+Mono:wght@400;500;600&display=swap');")

CSS = FONTS + """
html, body, [class*="css"], .stMarkdown, button, input, textarea, label { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: radial-gradient(1100px 460px at 8% -8%, rgba(255,122,61,.09), transparent 60%), %mist%; color: %ink%; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4, .grot { font-family: 'Space Grotesk', 'Inter', sans-serif !important; letter-spacing: -0.015em; color: %ink%; }
.mono, .kpi .l, .kpi .s, .chip, .topbar .sub, .badge, .rrow .t, .rrow .m, .rrow .c, .tlmeta { font-family: 'JetBrains Mono', ui-monospace, monospace; }
.block-container { padding-top: .6rem; max-width: 1480px; }
footer { visibility: hidden; }
section[data-testid="stSidebar"] { background: #0F1117; border-right: 1px solid %line%; }
.side-brand { display:flex; gap:12px; align-items:center; padding:4px 0 14px; }
.side-brand img { width:42px; height:42px; border-radius:11px; }
.side-brand b { font-family:'Space Grotesk'; font-size:1.05rem; color:%ink%; display:block; line-height:1.15; }
.side-brand span { font-size:.72rem; color:%muted%; font-family:'JetBrains Mono'; }

/* top bar */
.topbar { display:flex; justify-content:space-between; align-items:center; gap:16px; padding:14px 2px 18px; flex-wrap:wrap; }
.topbar .brand { display:flex; align-items:center; gap:16px; }
.topbar .logo { width:56px; height:56px; border-radius:15px; box-shadow:0 8px 28px rgba(255,122,61,.35); }
.topbar .name { font-family:'Space Grotesk'; font-weight:700; font-size:1.9rem; line-height:1.1; color:%ink%; }
.topbar .sub { font-size:.74rem; letter-spacing:.09em; text-transform:uppercase; color:%muted%; margin-top:3px; }
.badge { display:inline-flex; align-items:center; gap:9px; border:1px solid %line%; background:%card%; border-radius:999px; padding:9px 18px;
    font-size:.74rem; letter-spacing:.08em; text-transform:uppercase; color:#C9CEDA; }
.dot { width:8px; height:8px; border-radius:50%; background:%High%; box-shadow:0 0 10px %High%; display:inline-block; }

/* tabs: underline navigation */
.stTabs [data-baseweb="tab-list"] { gap:6px; background:transparent; padding:0; border-bottom:1px solid %line%; overflow-x:auto; }
.stTabs [data-baseweb="tab-border"] { background:transparent; }
.stTabs [data-baseweb="tab-highlight"] { background:%teal%; height:2px; }
.stTabs button[data-baseweb="tab"] { height:54px; padding:0 22px; font-weight:600; color:#DDE0E8; white-space:nowrap; background:transparent; border-radius:0; }
.stTabs button[data-baseweb="tab"] p { font-size:1rem; font-weight:600; }
.stTabs button[data-baseweb="tab"]:hover { color:#fff; background:rgba(255,255,255,.03); }
.stTabs button[data-baseweb="tab"][aria-selected="true"], .stTabs button[data-baseweb="tab"][aria-selected="true"] p { color:%teal%; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.3rem; }
.stTabs .stTabs [data-baseweb="tab-list"] { border-bottom:1px solid %line%; }
.stTabs .stTabs button[data-baseweb="tab"] { height:44px; padding:0 16px; }

/* cards */
div[data-testid="stVerticalBlockBorderWrapper"] { background:%card%; border:1px solid %line%; border-radius:18px; box-shadow:none; }

.ptitle { font-family:'Space Grotesk'; font-weight:700; font-size:2.05rem; margin:.4rem 0 .15rem; color:%ink%; }
.psub { color:#B6BBC8; font-size:1.02rem; margin-bottom:1.1rem; }
.sec-title { font-family:'Inter'; font-weight:700; font-size:.98rem; letter-spacing:.045em; text-transform:uppercase; color:#C9CEDA; margin:2px 0 2px; }
.sec-sub { color:%muted%; font-size:.88rem; margin-bottom:10px; }

/* intro banner */
.intro { display:flex; gap:16px; padding:18px 22px; border-radius:16px; border:1px solid rgba(255,122,61,.25);
    background:linear-gradient(90deg, rgba(255,122,61,.10), rgba(255,122,61,.025)); }
.intro .ic { flex:0 0 44px; height:44px; border-radius:12px; background:rgba(255,122,61,.16); color:%teal%; display:flex; align-items:center; justify-content:center; font-size:1.25rem; }
.intro b.h { font-size:1.08rem; display:block; margin-bottom:4px; color:%ink%; }
.intro p { margin:0; color:#B6BBC8; line-height:1.55; font-size:.96rem; }
.c-high { color:%High%; font-weight:700; } .c-med { color:%Medium%; font-weight:700; } .c-low { color:%Low%; font-weight:700; }

/* KPI tiles */
.kpi { background:%card%; border:1px solid %line%; border-radius:16px; padding:18px 22px 16px; height:100%; }
.kpi .bar { width:26px; height:3px; border-radius:3px; margin-bottom:10px; }
.kpi .l { color:#A3A9B8; font-size:.76rem; letter-spacing:.07em; text-transform:uppercase; display:flex; justify-content:space-between; align-items:center; }
.kpi .v { font-family:'Space Grotesk'; font-size:2.35rem; font-weight:700; color:%ink%; line-height:1.2; margin:6px 0 4px; }
.kpi .s { font-size:.78rem; color:%muted%; letter-spacing:.02em; }
.up { color:%High%; font-weight:700; } .down { color:%Low%; font-weight:700; }

/* info tooltips */
.info { display:inline-flex; width:17px; height:17px; border-radius:50%; background:#232733; color:#9AA0B0; font:600 10px 'JetBrains Mono';
    align-items:center; justify-content:center; margin-left:7px; position:relative; cursor:help; font-style:normal; text-transform:none; letter-spacing:0; vertical-align:middle; }
.info:hover::after { content:attr(data-tip); position:absolute; bottom:135%; left:50%; transform:translateX(-50%); width:250px; z-index:999; background:#07080B;
    border:1px solid #2F3544; padding:9px 11px; border-radius:9px; color:%ink%; font:400 .78rem/1.45 'Inter', sans-serif; text-align:left; white-space:normal; }

/* pills */
.pill { display:inline-block; padding:3px 13px; border-radius:999px; font-weight:700; font-size:.8rem; }
.pill.High { background:rgba(45,212,160,.15); color:%High%; } .pill.Medium { background:rgba(245,200,76,.15); color:%Medium%; }
.pill.Low { background:rgba(255,92,108,.16); color:%Low%; }
.pill.New { background:rgba(165,139,255,.18); color:%violet%; margin-left:6px; font-size:.68rem; padding:2px 9px; }
.bigstate { font-family:'Space Grotesk'; font-size:2.6rem; font-weight:700; line-height:1.1; }
.note { background:%card2%; border:1px solid %line%; border-left:4px solid %teal%; border-radius:12px; padding:12px 16px; color:#D5D8E0; font-size:.92rem; }
.note.warn { border-left-color:%Medium%; background:rgba(245,200,76,.06); } .note.bad { border-left-color:%Low%; background:rgba(255,92,108,.07); }
.note b { color:%ink%; }

/* traffic light */
.tl { width:128px; height:300px; margin:6px auto; border-radius:34px; background:#0A0B0F; border:1px solid #2A2F3B; display:flex; flex-direction:column;
    justify-content:space-evenly; align-items:center; box-shadow:inset 0 0 24px rgba(0,0,0,.7); }
.tl i { width:66px; height:66px; border-radius:50%; background:#1E212A; display:block; }
.tl i.r.on { background:radial-gradient(circle at 35% 30%, #FF9AA4, #FF3D55 70%); box-shadow:0 0 46px rgba(255,61,85,.75); }
.tl i.y.on { background:radial-gradient(circle at 35% 30%, #FFE79A, #F5B82E 70%); box-shadow:0 0 46px rgba(245,184,46,.7); }
.tl i.g.on { background:radial-gradient(circle at 35% 30%, #9CF5D6, #17C38B 70%); box-shadow:0 0 46px rgba(23,195,139,.7); }
.live-tag { text-align:center; font:500 .78rem 'JetBrains Mono'; letter-spacing:.14em; color:#A3A9B8; margin-top:6px; }
.live-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:%Low%; margin-right:8px; animation:pulse 1.7s infinite; box-shadow:0 0 0 0 rgba(255,92,108,.7); }
.paused-dot { display:inline-block; width:8px; height:8px; border-radius:50%; background:%Medium%; margin-right:8px; }
@keyframes pulse { 70% { box-shadow:0 0 0 9px rgba(255,92,108,0); } 100% { box-shadow:0 0 0 0 rgba(255,92,108,0); } }
@media (prefers-reduced-motion: reduce) { .live-dot { animation:none; } }
.tlstate { text-align:center; margin:10px 0 6px; } .tlstate .w { font-family:'Space Grotesk'; font-size:2.7rem; font-weight:700; }
.tlstate .tag { font-size:.82rem; margin-left:10px; }
.tlmeta { text-align:center; color:%muted%; font-size:.78rem; line-height:1.6; margin-bottom:12px; }

/* incoming readings */
.rlist { max-height:430px; overflow-y:auto; padding-right:6px; }
.rrow { display:flex; align-items:center; gap:14px; padding:13px 16px; background:%card2%; border:1px solid #232733; border-radius:12px; margin-bottom:9px; }
.rrow .d { width:10px; height:10px; border-radius:50%; flex:0 0 10px; }
.rrow .m { width:62px; color:#E6E8EE; font-size:.9rem; } .rrow .t { color:%muted%; font-size:.82rem; } .rrow .c { margin-left:auto; color:%muted%; font-size:.82rem; }
table.feed { width:100%; border-collapse:collapse; font-size:.82rem; }
table.feed th { text-align:left; color:%muted%; font-weight:600; padding:6px 8px; border-bottom:1px solid %line%; }
table.feed td { padding:6px 8px; border-bottom:1px solid #1D2029; }
.alert-row { display:flex; gap:10px; padding:8px 12px; border-radius:10px; margin-bottom:6px; background:rgba(255,92,108,.08); border:1px solid rgba(255,92,108,.22); font-size:.84rem; color:#E9D9DB; }
.alert-row.ok { background:rgba(45,212,160,.08); border-color:rgba(45,212,160,.25); color:#D5EFE7; }

/* buttons */
.stButton > button, .stDownloadButton > button { border-radius:11px; font-weight:600; background:%card2%; border:1px solid #2A2F3B; color:#C9CEDA; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color:%teal%; color:%teal%; }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] { background:rgba(255,122,61,.10); border:1px solid %teal%; color:%teal%; }
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { background:rgba(255,122,61,.2); }
"""

for _k, _v in {**C}.items():
    CSS = CSS.replace(f"%{_k}%", _v)


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _b64(path: str) -> str:
    return base64.b64encode(open(path, "rb").read()).decode()


def info(text: str) -> str:
    return f"<i class='info' data-tip=\"{html.escape(text, quote=True)}\">i</i>"


def topbar(title: str, subtitle: str, badge: str) -> None:
    logo = _b64(str(ASSETS / "logo.png"))
    st.markdown(f"<div class='topbar'><div class='brand'><img class='logo' src='data:image/png;base64,{logo}'/>"
                f"<div><div class='name'>{title}</div><div class='sub'>{subtitle}</div></div></div>"
                f"<div class='badge'><span class='dot'></span>{badge}</div></div>", unsafe_allow_html=True)


def page_title(title: str, sub: str = "") -> None:
    st.markdown(f"<div class='ptitle'>{title}</div><div class='psub'>{sub}</div>", unsafe_allow_html=True)


def section(title: str, sub: str = "", tip: str = "") -> None:
    st.markdown(f"<div class='sec-title'>{title}{info(tip) if tip else ''}</div>" + (f"<div class='sec-sub'>{sub}</div>" if sub else ""),
                unsafe_allow_html=True)


def note(text: str, tone: str = "") -> None:
    st.markdown(f"<div class='note {tone}'>{text}</div>", unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", color: str | None = None, tip: str = "", vcolor: str | None = None) -> str:
    bar = f"<div class='bar' style='background:{color}'></div>" if color else ""
    vstyle = f" style='color:{vcolor}'" if vcolor else ""
    return (f"<div class='kpi'>{bar}<div class='l'><span>{label}</span>{info(tip) if tip else ''}</div>"
            f"<div class='v'{vstyle}>{value}</div><div class='s'>{sub}</div></div>")


def pill(cls: str, extra: str = "") -> str:
    return f"<span class='pill {cls}'>{cls}</span>{extra}"


def delta_html(d: float, unit: str = "", good_when_up: bool = True, digits: int = 1) -> str:
    if abs(d) < 10 ** -digits / 2:
        return "<span>no change</span>"
    up = d > 0
    arrow = "▲" if up else "▼"
    return f"<span class='{'up' if up == good_when_up else 'down'}'>{arrow} {abs(d):.{digits}f}{unit}</span>"


# ------------------------------------------------ version-tolerant wrappers ----
def plot(fig, key: str | None = None) -> None:
    """st.plotly_chart across Streamlit versions (use_container_width -> width='stretch')."""
    kw = {"key": key} if key else {}
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, **kw)
    else:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, **kw)


def table(df, **kw) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def image(path, caption: str | None = None) -> None:
    try:
        st.image(str(path), caption=caption, width="stretch")
    except Exception:
        st.image(str(path), caption=caption, use_container_width=True)
