from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Union
import numpy as np
import pandas as pd
from constants import CRISIS_END, CRISIS_START, REGIME_ORDER_THREE_STATE

DateLike = Union[pd.Timestamp, str]

@dataclass(frozen=True)
class CrisisWindow:
    start: pd.Timestamp = CRISIS_START
    end: pd.Timestamp = CRISIS_END

def _to_ts(x: DateLike) -> pd.Timestamp:
    return pd.Timestamp(x).normalize()

def _normalize_any(ts: Union[pd.DatetimeIndex, pd.Series, Iterable]) -> tuple[Union[pd.DatetimeIndex, pd.Series], Union[pd.Index, None]]:
    if isinstance(ts, pd.DatetimeIndex):
        idx = ts
        tnorm = pd.DatetimeIndex(pd.to_datetime(idx)).normalize()
        return tnorm, idx
    if isinstance(ts, pd.Series):
        s = pd.to_datetime(ts, errors="coerce")
        tnorm = s.dt.normalize()
        return tnorm, ts.index
    arr = pd.to_datetime(list(ts), errors="coerce")
    tnorm = pd.DatetimeIndex(arr).normalize()
    return tnorm, None

def label_crisis_regime(
    dates: Union[pd.DatetimeIndex, pd.Series, Iterable],
    *,
    crisis_start: DateLike = CRISIS_START,
    crisis_end: DateLike = CRISIS_END,
    mode: str = "binary",
    binary_labels: Tuple[str, str] = ("crisis", "non_crisis"),
    three_labels: Tuple[str, str, str] = ("pre-crisis", "crisis", "post-crisis"),
    ordered: bool = True,
) -> pd.Series:
    cs = _to_ts(crisis_start)
    ce = _to_ts(crisis_end)

    t0, index = _normalize_any(dates)

    if mode == "binary":
        crisis_label, non_label = binary_labels
        lab = np.where((t0 >= cs) & (t0 <= ce), crisis_label, non_label)
        out = pd.Series(lab, index=index)
        if ordered:
            out = pd.Categorical(out, categories=[crisis_label, non_label], ordered=False)
            out = pd.Series(out, index=index)
        return out

    if mode == "three":
        pre, crisis, post = three_labels
        lab = np.where(t0 < cs, pre, np.where(t0 <= ce, crisis, post))
        out = pd.Series(lab, index=index)
        if ordered:
            cats = list(three_labels)
            out = pd.Series(pd.Categorical(out, categories=cats, ordered=True), index=index)
        return out

    raise ValueError("binary or three")

def regime_for_date(
    d: DateLike,
    regimes: Dict[str, Tuple[DateLike, DateLike]],
    *,
    inclusive: bool = True,
) -> str | None:
    dd = _to_ts(d)
    for name, (start, end) in regimes.items():
        s = _to_ts(start)
        e = _to_ts(end)
        if inclusive:
            if s <= dd <= e:
                return name
        else:
            if s <= dd < e:
                return name
    return None