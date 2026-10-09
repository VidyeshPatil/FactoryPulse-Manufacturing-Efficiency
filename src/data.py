"""Data loading, cleaning, scoring and persistence of new entries (no Streamlit imports)."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

from .config import (CLASSES, DATA_PATH, ENTRIES_PATH, LIMITS, MODES, RAW_NUMERIC, TARGET)
from .features import engineer, network_band
from .modeling import Predictor, efficiency_index

ENTRY_COLS = ["Datetime", "Machine_ID", "Operation_Mode", *RAW_NUMERIC, TARGET, "Label_Source", "Note"]


# ---------------------------------------------------------------- loading ----
def load_base() -> pd.DataFrame:
    """Read the factory CSV, repair the clock and return a chronologically ordered frame."""
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Time": "Timestamp"})
    parsed = pd.to_datetime(df["Date"] + " " + df["Timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    # The export contains one calendar day twice (01-03-2025). Readings are exactly one minute apart,
    # so rebuild a strictly increasing clock from the first timestamp.
    start = parsed.dropna().min()
    df["Datetime"] = start + pd.to_timedelta(np.arange(len(df)), unit="min")
    df["Clock_Repaired"] = parsed.ne(df["Datetime"])
    df["Label_Source"] = "Dataset"
    df["Note"] = ""
    return df.sort_values("Datetime", kind="stable").reset_index(drop=True)


def _read_entries() -> pd.DataFrame:
    if not ENTRIES_PATH.exists() or ENTRIES_PATH.stat().st_size == 0:
        return pd.DataFrame(columns=ENTRY_COLS)
    e = pd.read_csv(ENTRIES_PATH, parse_dates=["Datetime"])
    for c in ENTRY_COLS:
        if c not in e:
            e[c] = "" if c in ("Note", "Label_Source") else np.nan
    return e[ENTRY_COLS]


def load_entries() -> pd.DataFrame:
    e = _read_entries()
    if len(e):
        e["Date"] = e["Datetime"].dt.strftime("%d-%m-%Y")
        e["Timestamp"] = e["Datetime"].dt.strftime("%H:%M:%S")
        e["Clock_Repaired"] = False
    return e


def entries_signature() -> tuple:
    return (ENTRIES_PATH.stat().st_mtime_ns, ENTRIES_PATH.stat().st_size) if ENTRIES_PATH.exists() else (0, 0)


# ---------------------------------------------------------------- scoring ----
def score(df: pd.DataFrame, model: Predictor) -> pd.DataFrame:
    """Add probabilities, prediction, confidence, efficiency index and engineered features."""
    if df.empty:
        return df.assign(P_Low=[], P_Medium=[], P_High=[], Pred=[], Confidence=[], Efficiency_Index=[],
                         Sensor_Stability=[], Network_Reliability=[], Network_Band=[])
    P = model.proba(df)
    out = df.copy()
    out["P_Low"], out["P_Medium"], out["P_High"] = P[:, 0], P[:, 1], P[:, 2]
    out["Pred"] = np.array(CLASSES)[P.argmax(1)]
    out["Confidence"] = P.max(1)
    out["Efficiency_Index"] = efficiency_index(P)
    eng = engineer(df)
    for c in ("Sensor_Stability", "Thermal_Vibration_Stress", "Network_Reliability", "Energy_Wh_per_Unit",
              "Error_Output_Ratio", "Effective_Throughput"):
        out[c] = eng[c].values
    out["Network_Band"] = network_band(out["Network_Reliability"])
    out["Source"] = np.where(out["Label_Source"].eq("Dataset"), "Dataset", "Manual")
    # rows without a label (new entries classified by the AI) take the prediction as their label
    out[TARGET] = out[TARGET].where(out[TARGET].isin(CLASSES), out["Pred"])
    out["Date_Str"] = out["Datetime"].dt.strftime("%Y-%m-%d")
    out["Hour"] = out["Datetime"].dt.hour
    return out


def combine(base_scored: pd.DataFrame, entries_scored: pd.DataFrame) -> pd.DataFrame:
    if entries_scored.empty:
        return base_scored
    return pd.concat([base_scored, entries_scored], ignore_index=True).sort_values("Datetime", kind="stable").reset_index(drop=True)


# ------------------------------------------------------------- validation ----
def validate_entries(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Coerce, range-check and normalise a dataframe of new readings. Returns (clean_rows, problems)."""
    problems: list[str] = []
    d = df.copy()
    d = d.rename(columns={"Time": "Timestamp"})
    if "Datetime" not in d:
        if {"Date", "Timestamp"} <= set(d.columns):
            d["Datetime"] = pd.to_datetime(d["Date"].astype(str) + " " + d["Timestamp"].astype(str), dayfirst=True, errors="coerce")
        else:
            problems.append("Provide either a `Datetime` column or both `Date` and `Timestamp` columns.")
    missing = [c for c in ["Machine_ID", "Operation_Mode", *RAW_NUMERIC] if c not in d]
    if missing:
        problems.append("Missing columns: " + ", ".join(missing))
        return d.iloc[0:0], problems
    d["Datetime"] = pd.to_datetime(d["Datetime"], errors="coerce")
    for c in ["Machine_ID", *RAW_NUMERIC]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    bad = d[["Datetime", "Machine_ID", *RAW_NUMERIC]].isna().any(axis=1) | ~d["Operation_Mode"].isin(MODES)
    if bad.any():
        problems.append(f"{int(bad.sum())} row(s) have blanks, unreadable numbers or an Operation_Mode outside {MODES}; they were skipped.")
    d = d[~bad].copy()
    out_of_range = pd.Series(False, index=d.index)
    for c, (lo, hi) in LIMITS.items():
        out_of_range |= (d[c] < lo) | (d[c] > hi)
    if out_of_range.any():
        problems.append(f"{int(out_of_range.sum())} row(s) fall outside the physical ranges seen in training; they were skipped.")
        d = d[~out_of_range]
    d["Machine_ID"] = d["Machine_ID"].astype(int)
    if TARGET not in d:
        d[TARGET] = np.nan
    d["Label_Source"] = np.where(d[TARGET].isin(CLASSES), "User", "Model")
    if "Note" not in d:
        d["Note"] = ""
    return d[ENTRY_COLS].reset_index(drop=True), problems


def save_entries(new_rows: pd.DataFrame) -> int:
    if new_rows.empty:
        return 0
    ENTRIES_PATH.parent.mkdir(exist_ok=True)
    exists = ENTRIES_PATH.exists() and ENTRIES_PATH.stat().st_size > 0
    new_rows[ENTRY_COLS].to_csv(ENTRIES_PATH, mode="a", header=not exists, index=False)
    return len(new_rows)


def delete_entries() -> None:
    if ENTRIES_PATH.exists():
        ENTRIES_PATH.unlink()


def training_frame(base: pd.DataFrame) -> pd.DataFrame:
    """Dataset rows + entries whose label was supplied by a person (never self-labelled rows)."""
    e = _read_entries()
    e = e[e["Label_Source"].eq("User")]
    return pd.concat([base, e], ignore_index=True) if len(e) else base


def template_csv() -> bytes:
    ex = pd.DataFrame([{"Datetime": "2025-03-11 08:00:00", "Machine_ID": 7, "Operation_Mode": "Active",
                        "Temperature_C": 61.2, "Vibration_Hz": 2.4, "Power_Consumption_kW": 5.6,
                        "Network_Latency_ms": 22.5, "Packet_Loss_%": 1.8, "Quality_Control_Defect_Rate_%": 4.1,
                        "Production_Speed_units_per_hr": 410.0, "Predictive_Maintenance_Score": 0.62,
                        "Error_Rate_%": 1.4, "Efficiency_Status": "", "Note": "example row"}])
    buf = io.StringIO()
    ex.to_csv(buf, index=False)
    return buf.getvalue().encode()
