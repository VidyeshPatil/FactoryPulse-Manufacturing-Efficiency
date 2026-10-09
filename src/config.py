"""Central configuration: paths, palette, column metadata and business thresholds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Thales_Group_Manufacturing.csv"
ENTRIES_PATH = ROOT / "data" / "new_entries.csv"
MODEL_PATH = ROOT / "models" / "efficiency_models.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
ASSETS = ROOT / "assets"

APP_NAME = "FactoryPulse"

# ---------------------------------------------------------------- target ----
CLASSES = ["Low", "Medium", "High"]               # order = probability columns
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}
MODES = ["Active", "Idle", "Maintenance"]
TARGET = "Efficiency_Status"

# --------------------------------------------------------------- palette ----
C = dict(                                           # dark theme, orange accent
    ink="#ECEEF3", deep="#FFD9C2", teal="#FF7A3D", aqua="#FFB38A", sky="#5BA8FF",
    mist="#0B0C10", card="#14161D", card2="#1B1E27", line="#262A35", muted="#8B91A1",
    High="#2DD4A0", Medium="#F5C84C", Low="#FF5C6C", violet="#A58BFF",
)
CLASS_COLORS = {k: C[k] for k in CLASSES}

# ------------------------------------------------------- raw metric meta ----
# name: (label, unit, group, min, max); min/max are the physical ranges in the dataset
METRICS = {
    "Temperature_C":                 ("Temperature", "°C", "Sensor", 30.0, 90.0),
    "Vibration_Hz":                  ("Vibration", "Hz", "Sensor", 0.1, 5.0),
    "Power_Consumption_kW":          ("Power draw", "kW", "Sensor", 1.5, 10.0),
    "Network_Latency_ms":            ("Network latency", "ms", "Network", 1.0, 50.0),
    "Packet_Loss_%":                 ("Packet loss", "%", "Network", 0.0, 5.0),
    "Quality_Control_Defect_Rate_%": ("Defect rate", "%", "Quality", 0.0, 10.0),
    "Production_Speed_units_per_hr": ("Production speed", "units/hr", "Production", 50.0, 500.0),
    "Predictive_Maintenance_Score":  ("Maintenance score", "", "Maintenance", 0.0, 1.0),
    "Error_Rate_%":                  ("Error rate", "%", "Quality", 0.0, 15.0),
}
RAW_NUMERIC = list(METRICS)
LABEL = {k: v[0] for k, v in METRICS.items()}
UNIT = {k: v[1] for k, v in METRICS.items()}
GROUP = {k: v[2] for k, v in METRICS.items()}
LIMITS = {k: (v[3], v[4]) for k, v in METRICS.items()}
GROUP_COLORS = {"Sensor": C["teal"], "Network": C["sky"], "Production": C["Medium"],
                "Quality": C["Low"], "Maintenance": C["violet"], "Mode": C["muted"]}

# ---------------------------------------------------- engineered features ----
ENGINEERED = {
    "Sensor_Stability":         "Sensor stability (0-100): how close temperature, vibration and power sit to their nominal centre",
    "Thermal_Vibration_Stress": "Thermal-vibration stress (0-100): combined heat and shaking load",
    "Network_Reliability":      "Network reliability (0-100): blends latency and packet loss, packet loss weighted higher",
    "Energy_Wh_per_Unit":       "Energy per unit (Wh): power draw divided by production speed",
    "Error_Output_Ratio":       "Error-to-output ratio: error rate per unit of hourly output",
    "Defect_Output_Ratio":      "Defect-to-output ratio: defect rate per unit of hourly output",
    "Effective_Throughput":     "Effective throughput (units/hr): speed after removing errors and defects",
}
LEAKY_ENGINEERED = ["Energy_Wh_per_Unit", "Error_Output_Ratio", "Defect_Output_Ratio", "Effective_Throughput"]
EARLY_EXCLUDE_RAW = ["Error_Rate_%", "Production_Speed_units_per_hr"]

# ------------------------------------------------ discovered business rule ----
# Found by a depth-4 surrogate tree during training; agrees with the label on 99.998 % of rows.
RULE = dict(error_low=5.0, speed_low=200.0, speed_high=400.0, error_high=2.0)

# --------------------------------------------------------------- alerting ----
ALERTS = {
    "Error_Rate_%":                 (">", 5.0,  "Error rate above 5 % (Low-efficiency zone)"),
    "Packet_Loss_%":                (">", 4.0,  "Packet loss above 4 % on the 6G link"),
    "Network_Latency_ms":           (">", 40.0, "6G latency above 40 ms"),
    "Vibration_Hz":                 (">", 4.5,  "Vibration above 4.5 Hz"),
    "Temperature_C":                (">", 85.0, "Temperature above 85 °C"),
    "Predictive_Maintenance_Score": ("<", 0.10, "Maintenance score below 0.10: service due"),
}
NETWORK_BANDS = ["Poor", "Fair", "Good"]
