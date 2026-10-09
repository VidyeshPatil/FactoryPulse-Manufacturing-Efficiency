"""Sidebar filter application (pure pandas, no Streamlit)."""
import pandas as pd


def apply(df: pd.DataFrame, F: dict, time: bool = True) -> pd.DataFrame:
    """Empty selections mean 'all'. Set time=False for the live view, which follows the replay clock instead."""
    m = pd.Series(True, index=df.index)
    if F.get("machines"):
        m &= df["Machine_ID"].isin(F["machines"])
    if F.get("modes"):
        m &= df["Operation_Mode"].isin(F["modes"])
    if F.get("nets"):
        m &= df["Network_Band"].isin(F["nets"])
    if time:
        m &= df["Datetime"].between(pd.Timestamp(F["d0"]), pd.Timestamp(F["d1"]) + pd.Timedelta(hours=23, minutes=59, seconds=59))
        m &= df["Hour"].between(F["h0"], max(F["h0"], F["h1"] - 1))
    return df[m]
