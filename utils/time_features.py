from __future__ import annotations
from datetime import time
from typing import Union
import numpy as np
import pandas as pd

TimeLike = Union[str, time]

def ensure_time(t: TimeLike) -> time:
    if isinstance(t, time):
        return t
    if isinstance(t, str):
        ts = pd.to_datetime(t).time()
        return ts
    raise TypeError(f"Unsupported time type: {type(t)}")


def minutes_since_midnight(t: time) -> int:
    return int(t.hour * 60 + t.minute)


def minute_of_day(index: pd.DatetimeIndex) -> pd.Series:
    idx = pd.DatetimeIndex(index)
    return pd.Series(idx.hour * 60 + idx.minute, index=idx)


def minute_of_session(index: pd.DatetimeIndex, session_start: TimeLike) -> pd.Series:
    ss = ensure_time(session_start)
    ss_min = minutes_since_midnight(ss)

    idx = pd.DatetimeIndex(index)
    mod = idx.hour * 60 + idx.minute
    return pd.Series(mod - ss_min, index=idx)


def between_session(
        df: pd.DataFrame,
        session_start: TimeLike,
        session_end: TimeLike,
        inclusive: str = "both",
) -> pd.DataFrame:
    ss = ensure_time(session_start)
    se = ensure_time(session_end)
    # between_time accepts time objects
    return df.between_time(ss, se, inclusive=inclusive)


def expected_bars_in_session(
        session_start: TimeLike,
        session_end: TimeLike,
        bar_minutes: int = 1,
        inclusive: bool = True,
) -> int:
    bar_minutes = int(bar_minutes)

    ss = ensure_time(session_start)
    se = ensure_time(session_end)

    ss_min = minutes_since_midnight(ss)
    se_min = minutes_since_midnight(se)

    span = se_min - ss_min
    if inclusive:
        span += 1

    return int(np.floor((span - 1) / bar_minutes) + 1) if span > 0 else 0


def add_date_and_weekday_cols(
        df: pd.DataFrame,
        date_col: str = "date",
        weekday_col: str = "weekday",
        weekday_name_col: str = "weekday_name",
        weekday_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    out[date_col] = pd.to_datetime(idx.date)
    out[weekday_col] = idx.weekday
    if weekday_names is not None:
        out[weekday_name_col] = out[weekday_col].map(weekday_names)
    return out