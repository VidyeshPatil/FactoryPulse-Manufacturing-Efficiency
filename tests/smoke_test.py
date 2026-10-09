"""Offline smoke test: runs app.py and every tab against stand-in streamlit/plotly modules.
It exercises all pandas/model logic (not browser rendering).  Usage: python tests/smoke_test.py"""
import runpy
import sys
import types
from datetime import date, time
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class SS(dict):
    __getattr__ = lambda s, k: s.get(k)
    def __setattr__(s, k, v): s[k] = v


STACK = []


class Box:
    """Stand-in for a Streamlit container; tracks column nesting (Streamlit allows columns inside columns only once)."""
    def __init__(s, depth=0): s.depth = depth
    def __enter__(s): STACK.append(s.depth); return s
    def __exit__(s, *a): STACK.pop(); return False
    def __getattr__(s, name): return lambda *a, **k: call(name, *a, _d=s.depth, **k)


SESSION = SS()
MEMO = {}


class Cache:
    def __call__(s, *a, **k):
        def deco(fn):
            def w(*x):
                key = (fn.__name__, x)
                if key not in MEMO: MEMO[key] = fn(*x)
                return MEMO[key]
            return w
        return deco(a[0]) if a and callable(a[0]) else deco
    def clear(s): MEMO.clear()


def call(name, *a, _d=None, **k):
    key = k.get("key")
    d = _d if _d is not None else (STACK[-1] if STACK else 0)
    if name == "columns":
        if d + 1 > 2: raise RuntimeError("Columns nested more than one level deep: Streamlit raises an exception here")
        return [Box(d + 1) for _ in range(a[0] if isinstance(a[0], int) else len(a[0]))]
    if name == "tabs": return [Box(d) for _ in a[0]]
    if name in ("container", "form", "sidebar", "expander", "spinner", "status"): return Box(d)
    if name == "fragment":
        if a and callable(a[0]): return a[0]
        return lambda fn: fn
    if name == "slider": return SESSION[key] if key in SESSION else (a[3] if len(a) > 3 else a[1])
    if name == "select_slider": return SESSION[key] if key in SESSION else a[1][0] if len(a) > 1 else k["options"][0]
    if name == "selectbox":
        opts = list(a[1]) if len(a) > 1 else list(k["options"])
        return SESSION[key] if key in SESSION and SESSION[key] in opts else opts[k.get("index", 0)]
    if name == "multiselect": return SESSION[key] if key in SESSION else k.get("default", [])
    if name == "number_input": return SESSION[key] if key in SESSION else a[3]
    if name == "date_input": return k["value"]
    if name == "time_input": return k["value"]
    if name == "radio": return a[1][0]
    if name in ("button", "form_submit_button", "checkbox", "download_button"): return False
    if name in ("file_uploader",): return None
    if name == "text_input": return ""
    return MagicMock()


st = types.ModuleType("streamlit")
st.__getattr__ = lambda n: Cache() if n in ("cache_resource", "cache_data") else (lambda *a, **k: call(n, *a, **k))
st.cache_resource, st.cache_data = Cache(), Cache()
st.session_state, st.sidebar, st.__version__ = SESSION, Box(), "1.50.0"
sys.modules["streamlit"] = st
for n in ("plotly", "plotly.graph_objects", "plotly.subplots"):
    sys.modules[n] = MagicMock()
for n in ("columns", "tabs", "fragment", "set_page_config", "markdown", "container", "button", "slider", "select_slider", "selectbox",
          "multiselect", "date_input", "time_input", "number_input", "radio", "checkbox", "download_button", "form", "form_submit_button",
          "text_input", "file_uploader", "progress", "success", "info", "warning", "error", "caption", "write", "dataframe", "image",
          "plotly_chart", "rerun", "spinner", "stop"):
    setattr(st, n, (lambda nn: (lambda *a, **k: call(nn, *a, **k)))(n))
st.plotly_chart = lambda *a, **k: None


def run(label):
    runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    print("ok:", label)


if __name__ == "__main__":
    from src import data as D
    D.delete_entries()
    SESSION.update(dict(cursor=600, playing=True, rows_per_tick=5, window=240))
    run("first pass")
    for i in range(3):
        run(f"live tick {i}")
    # add entries (faulty, excellent, unlabelled) and a brand-new machine, then render again
    import pandas as pd
    clock = SESSION["clock"]
    rows = pd.DataFrame([dict(Datetime=clock, Machine_ID=m, Operation_Mode="Active", Temperature_C=70, Vibration_Hz=4.8, Power_Consumption_kW=6,
                              Network_Latency_ms=45, **{"Packet_Loss_%": 4.5, "Quality_Control_Defect_Rate_%": 5, "Error_Rate_%": e}, Production_Speed_units_per_hr=s,
                              Predictive_Maintenance_Score=0.05, Efficiency_Status=lab, Note="t")
                         for m, e, s, lab in [(1, 9, 140, None), (777, 0.8, 470, "High"), (3, 3, 300, None)]])
    clean, probs = D.validate_entries(rows)
    assert len(clean) == 3 and not probs, probs
    D.save_entries(clean)
    run("with entries")
    SESSION["m_sel"] = 777
    SESSION["cursor"] = 99_990
    run("end of stream + new machine")
    D.delete_entries()
    print("ALL SMOKE TESTS PASSED")
