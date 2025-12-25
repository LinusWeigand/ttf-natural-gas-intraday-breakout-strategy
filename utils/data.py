from __future__ import annotations
import math
from typing import Sequence
import pandas as pd
from constants import CSV_PATH, FREQ, SESSION_END, SESSION_START, WEEKDAYS_ONLY

def load_filtered_df(min_perc_first_hour: float, min_perc_total: float) -> pd.DataFrame:
    resolution_minutes = 1

    df = pd.read_csv(CSV_PATH, sep=";", decimal=",", parse_dates=["Time"])
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.dropna(subset=["Time"]).copy()
    df["Time"] = df["Time"].dt.tz_localize(None)

    full_index = pd.date_range(df["Time"].min(), df["Time"].max(), freq=FREQ)
    master_df = pd.DataFrame(index=full_index)
    master_df.index.name = "Time"

    audit_df = master_df.join(df.set_index("Time"), how="left")

    audit_df = audit_df.between_time(SESSION_START, SESSION_END, inclusive="both").copy()
    if WEEKDAYS_ONLY:
        audit_df = audit_df[audit_df.index.weekday < 5].copy()

    expected_first_hour = math.ceil(60 / resolution_minutes)

    template_day = pd.Timestamp("2000-01-01")
    expected_total = len(
        pd.date_range(
            template_day.strftime("%Y-%m-%d") + f" {SESSION_START}",
            template_day.strftime("%Y-%m-%d") + f" {SESSION_END}",
            freq=FREQ,
            )
    )

    audit_df["Date_Only"] = audit_df.index.date
    is_first_hour = audit_df.index.time < pd.to_datetime("08:00").time()

    per_day_first = (
        audit_df[is_first_hour]
        .groupby("Date_Only")["close"]
        .apply(lambda s: int(s.notna().sum()))
    )
    per_day_total = (
        audit_df
        .groupby("Date_Only")["close"]
        .apply(lambda s: int(s.notna().sum()))
    )

    cov = pd.DataFrame({"first_hour_valid": per_day_first, "total_valid": per_day_total}).fillna(0)

    cov["first_hour_ok"] = cov["first_hour_valid"] >= expected_first_hour * float(min_perc_first_hour)
    cov["total_ok"] = cov["total_valid"] >= expected_total * float(min_perc_total)

    keep_dates = cov.index[cov["first_hour_ok"] & cov["total_ok"]]

    out = audit_df[audit_df["Date_Only"].isin(keep_dates)].drop(columns=["Date_Only"])
    return out


def load_session_df(
        *,
        min_perc_first_hour: float,
        min_perc_total: float,
        require_cols: Sequence[str] | None = None,
        dropna_cols: Sequence[str] | None = None,
        allow_negative_volume: bool = False,
) -> pd.DataFrame:
    df = load_filtered_df(
        min_perc_first_hour=min_perc_first_hour,
        min_perc_total=min_perc_total,
    ).copy()

    if require_cols:
        missing = [c for c in require_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    if dropna_cols:
        df = df.dropna(subset=list(dropna_cols)).copy()

    if ("volume" in df.columns) and (not allow_negative_volume):
        df = df[(df["volume"].isna()) | (df["volume"] >= 0)].copy()

    df = df.sort_index()
    return df


def bucket_time_to_resolution(
        df: pd.DataFrame,
        time_col: str,
        resolution_minutes: int,
) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).copy()
    out[time_col] = out[time_col].dt.tz_localize(None)
    if int(resolution_minutes) != 1:
        out[time_col] = out[time_col].dt.floor(f"{int(resolution_minutes)}min")
    return out
