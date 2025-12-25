from __future__ import annotations
from datetime import time
from typing import Tuple
import numpy as np
import pandas as pd
from constants import DEFAULT_MAX_GAP_MINUTES

def first_hour_mask(
        index: pd.DatetimeIndex,
        *,
        start: time = time(7, 0),
        end: time = time(8, 0),
) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    return (idx.time >= start) & (idx.time < end)


def post_first_hour_mask(
        index: pd.DatetimeIndex,
        *,
        start: time = time(8, 0),
) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    return idx.time >= start


def gap_aware_sigma_log_per_minute(
        day: pd.DataFrame,
        *,
        close_col: str = "close",
        fh_start: time = time(7, 0),
        fh_end: time = time(8, 0),
        max_gap_minutes: int | None = DEFAULT_MAX_GAP_MINUTES,
) -> Tuple[float, float]:
    if close_col not in day.columns:
        return np.nan, 0.0

    fh = day.loc[first_hour_mask(day.index, start=fh_start, end=fh_end)].copy()
    fh = fh.dropna(subset=[close_col]).sort_index()
    if len(fh) < 3:
        return np.nan, 0.0

    close = fh[close_col].astype(float)
    close = close[close > 0]
    if len(close) < 3:
        return np.nan, 0.0

    logp = np.log(close.to_numpy())
    r = np.diff(logp)

    dt = fh.index.to_series().diff().dt.total_seconds().to_numpy()[1:] / 60.0  # minutes
    ok = (dt > 0) & np.isfinite(dt) & np.isfinite(r)

    if max_gap_minutes is not None:
        ok &= (dt <= float(max_gap_minutes))

    if int(ok.sum()) < 2:
        return np.nan, 0.0

    total_min = float(dt[ok].sum())
    if not (np.isfinite(total_min) and total_min > 0):
        return np.nan, 0.0

    rv = float(np.sum(r[ok] ** 2))
    sigma_per_min = float(np.sqrt(rv / total_min))
    return sigma_per_min, total_min


def sigma_px_over_horizon(
        ref_price: float,
        sigma_per_min: float,
        total_min: float,
) -> float:
    if not (np.isfinite(ref_price) and ref_price > 0 and np.isfinite(sigma_per_min) and sigma_per_min > 0 and np.isfinite(total_min) and total_min > 0):
        return np.nan
    sigma_h = float(sigma_per_min * np.sqrt(float(total_min)))
    return float(ref_price * (np.exp(sigma_h) - 1.0))


def choose_sigma_ref_price(
        day: pd.DataFrame,
        *,
        prefer_vwap: bool = True,
        fh_start: time = time(7, 0),
        fh_end: time = time(8, 0),
        close_col: str = "close",
        volume_col: str = "volume",
) -> float:
    if close_col not in day.columns:
        return np.nan

    fh = day.loc[first_hour_mask(day.index, start=fh_start, end=fh_end)].copy()

    if prefer_vwap and volume_col in fh.columns:
        tmp = fh.dropna(subset=[close_col, volume_col]).copy()
        tmp = tmp[(tmp[close_col] > 0) & (tmp[volume_col] > 0)]
        if not tmp.empty:
            vwap = float((tmp[close_col].astype(float) * tmp[volume_col].astype(float)).sum() / tmp[volume_col].astype(float).sum())
            if np.isfinite(vwap) and vwap > 0:
                return vwap

    s = pd.to_numeric(day[close_col], errors="coerce").dropna()
    s = s[s > 0]
    if s.empty:
        return np.nan
    return float(s.iloc[0])