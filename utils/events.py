from __future__ import annotations
from dataclasses import dataclass
from datetime import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from volatility import post_first_hour_mask

def first_true_cross_time(
        day: pd.DataFrame,
        level: float,
        direction: str,
        *,
        after_time: time = time(8, 0),
        close_col: str = "close",
) -> pd.Timestamp | None:
    if close_col not in day.columns or not np.isfinite(level):
        return None

    post = day.loc[post_first_hour_mask(day.index, start=after_time)].copy()
    post = post.dropna(subset=[close_col]).sort_index()
    if len(post) < 2:
        return None

    prev_close = post[close_col].shift(1)

    if direction == "up":
        hit = (prev_close < level) & (post[close_col] > level)
    else:
        hit = (prev_close > level) & (post[close_col] < level)

    idx = post.index[hit.fillna(False)]
    return None if len(idx) == 0 else pd.Timestamp(idx[0])

def first_confluence_cross_time(
        day: pd.DataFrame,
        levels: Dict[str, float],
        legs: List[Tuple[str, str]],
        *,
        after_time: time = time(8, 0),
        close_col: str = "close",
) -> pd.Timestamp | None:
    times: List[pd.Timestamp] = []
    for level_name, direction in legs:
        lvl = levels.get(level_name, np.nan)
        if lvl is None or not np.isfinite(lvl):
            return None
        t = first_true_cross_time(day, float(lvl), direction, after_time=after_time, close_col=close_col)
        if t is None:
            return None
        times.append(t)
    return max(times) if times else None

@dataclass(frozen=True)
class ExcursionResult:
    mfe: float
    mae: float
    t_mfe_min: float
    t_mae_min: float
    fail_within_window: float

def compute_excursions(
        day: pd.DataFrame,
        level: float,
        t0: pd.Timestamp,
        direction: str,
        epsilon: float,
        *,
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        fail_window_minutes: int = 30,
) -> ExcursionResult:
    needed = [c for c in [high_col, low_col, close_col] if c not in day.columns]
    if needed:
        return ExcursionResult(np.nan, np.nan, np.nan, np.nan, np.nan)
    if not (np.isfinite(level) and pd.notna(t0)):
        return ExcursionResult(np.nan, np.nan, np.nan, np.nan, np.nan)

    fwd = day.loc[t0:].dropna(subset=[high_col, low_col, close_col]).sort_index()
    if fwd.empty:
        return ExcursionResult(np.nan, np.nan, np.nan, np.nan, np.nan)

    if direction == "up":
        fav = fwd[high_col].astype(float) - float(level)
        adv = float(level) - fwd[low_col].astype(float)
    else:
        fav = float(level) - fwd[low_col].astype(float)
        adv = fwd[high_col].astype(float) - float(level)

    mfe = float(fav.max()) if len(fav) else np.nan
    mae = float(adv.max()) if len(adv) else np.nan

    try:
        t_mfe = float((pd.Timestamp(fav.idxmax()) - pd.Timestamp(t0)).total_seconds() / 60.0) if np.isfinite(mfe) else np.nan
    except Exception:
        t_mfe = np.nan

    try:
        t_mae = float((pd.Timestamp(adv.idxmax()) - pd.Timestamp(t0)).total_seconds() / 60.0) if np.isfinite(mae) else np.nan
    except Exception:
        t_mae = np.nan

    w_end = pd.Timestamp(t0) + pd.Timedelta(minutes=int(fail_window_minutes))
    w = fwd.loc[t0:w_end].dropna(subset=[close_col])
    if w.empty:
        fail = np.nan
    else:
        cl = w[close_col].astype(float)
        if direction == "up":
            fail = float((cl <= (float(level) - float(epsilon))).any())
        else:
            fail = float((cl >= (float(level) + float(epsilon))).any())

    return ExcursionResult(mfe=mfe, mae=mae, t_mfe_min=t_mfe, t_mae_min=t_mae, fail_within_window=fail)

def combo_cross_time(
        day: pd.DataFrame,
        primary_level: float,
        confirm_level: float,
        direction: str,
        *,
        after_time: time = time(8, 0),
        close_col: str = "close",
) -> pd.Timestamp | None:
    t1 = first_true_cross_time(day, primary_level, direction, after_time=after_time, close_col=close_col)
    if t1 is None:
        return None
    t2 = first_true_cross_time(day, confirm_level, direction, after_time=after_time, close_col=close_col)
    if t2 is None:
        return None
    return max(t1, t2)