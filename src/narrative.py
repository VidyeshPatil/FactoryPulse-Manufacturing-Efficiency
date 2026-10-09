"""Plain-language explanations that engineers and non-technical stakeholders can read."""
from .config import RULE

E, S = "Error_Rate_%", "Production_Speed_units_per_hr"


def reasons(r: dict, pred: str) -> list[str]:
    e, s = float(r[E]), float(r[S])
    out = []
    if e > RULE["error_low"]:
        out.append(f"Error rate is **{e:.1f} %**, above the {RULE['error_low']:.0f} % ceiling: efficiency cannot rise above Low.")
    if s <= RULE["speed_low"]:
        out.append(f"Production speed is **{s:.0f} units/hr**, at or below the {RULE['speed_low']:.0f} units/hr floor for Low.")
    if pred == "High":
        out.append(f"Speed **{s:.0f}** is above {RULE['speed_high']:.0f} units/hr and errors **{e:.1f} %** are within {RULE['error_high']:.0f} %: the High zone.")
    elif pred == "Medium":
        out.append(f"Speed **{s:.0f}** and errors **{e:.1f} %** sit between the Low floor and the High zone.")
    elif not out:
        out.append("Speed and error rate sit close to the Low boundary; the model leans Low on the combined pattern.")
    return out


def next_step(r: dict, pred: str) -> str:
    e, s = float(r[E]), float(r[S])
    if pred == "Low":
        tips = []
        if e > RULE["error_low"]:
            tips.append(f"cut the error rate by {e - RULE['error_low']:.1f} points to {RULE['error_low']:.0f} % or less")
        if s <= RULE["speed_low"]:
            tips.append(f"raise output by {RULE['speed_low'] - s + 1:.0f} units/hr to clear {RULE['speed_low']:.0f}")
        return "To reach Medium: " + " and ".join(tips or ["improve error rate and speed together"]) + "."
    if pred == "Medium":
        tips = []
        if s <= RULE["speed_high"]:
            tips.append(f"lift speed by {RULE['speed_high'] - s + 1:.0f} units/hr to pass {RULE['speed_high']:.0f}")
        if e > RULE["error_high"]:
            tips.append(f"reduce errors by {e - RULE['error_high']:.1f} points to {RULE['error_high']:.0f} % or less")
        return "To reach High: " + " and ".join(tips) + "."
    return (f"Hold the line: speed has {s - RULE['speed_high']:.0f} units/hr of headroom and errors have "
            f"{RULE['error_high'] - e:.1f} points before the status drops to Medium.")
