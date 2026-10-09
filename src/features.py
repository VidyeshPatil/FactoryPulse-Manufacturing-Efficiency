"""Feature engineering used identically in training, batch scoring and live what-if."""
import numpy as np
import pandas as pd

from .config import ENGINEERED, LIMITS, RAW_NUMERIC, MODES


def _norm(s: pd.Series, col: str) -> pd.Series:
    lo, hi = LIMITS[col]
    return ((s - lo) / (hi - lo)).clip(0, 1)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Return raw model inputs plus engineered columns. Never mutates *df*."""
    x = df[RAW_NUMERIC].astype(float).copy()
    x["Operation_Mode"] = pd.Categorical(df["Operation_Mode"], categories=MODES).astype(object)

    # sensor stability: 100 = every sensor at its nominal centre, 0 = every sensor at an extreme
    dev = sum((_norm(x[c], c) - 0.5).abs() * 2 for c in ("Temperature_C", "Vibration_Hz", "Power_Consumption_kW")) / 3
    x["Sensor_Stability"] = 100 * (1 - dev)
    x["Thermal_Vibration_Stress"] = 100 * (_norm(x["Temperature_C"], "Temperature_C") + _norm(x["Vibration_Hz"], "Vibration_Hz")) / 2

    # 6G network reliability: packet loss weighted a little higher than latency
    x["Network_Reliability"] = 100 * (1 - (0.4 * _norm(x["Network_Latency_ms"], "Network_Latency_ms")
                                           + 0.6 * _norm(x["Packet_Loss_%"], "Packet_Loss_%")))

    speed = x["Production_Speed_units_per_hr"].clip(lower=1.0)
    err, dfc = x["Error_Rate_%"], x["Quality_Control_Defect_Rate_%"]
    x["Energy_Wh_per_Unit"] = 1000 * x["Power_Consumption_kW"] / speed
    x["Error_Output_Ratio"] = 100 * err / speed
    x["Defect_Output_Ratio"] = 100 * dfc / speed
    x["Effective_Throughput"] = speed * (1 - err / 100) * (1 - dfc / 100)
    return x


def network_band(reliability: pd.Series) -> pd.Series:
    return pd.cut(reliability, [-np.inf, 40, 60, np.inf], labels=["Poor", "Fair", "Good"]).astype(str)


assert set(ENGINEERED) <= set(engineer(pd.DataFrame({**{c: [LIMITS[c][0]] for c in RAW_NUMERIC}, "Operation_Mode": ["Active"]})).columns)
