from __future__ import annotations
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

def summarize_expansion(values: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) == 0:
        return {"N": 0, "Mean": np.nan, "Median": np.nan, "P25": np.nan, "P75": np.nan}
    return {
        "N": int(len(v)),
        "Mean": float(v.mean()),
        "Median": float(v.median()),
        "P25": float(v.quantile(0.25)),
        "P75": float(v.quantile(0.75)),
    }

def grouped_quantile_summary(
        df: pd.DataFrame,
        group_cols: List[str],
        value_col: str,
        *,
        quantiles: Sequence[float] = (0.10, 0.50, 0.90),
        extra_aggs: Optional[Dict[str, Tuple[str, Union[str, Callable]]]] = None,
        count_name: str = "n",
        dropna_value: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    d = df.copy()
    if dropna_value:
        d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
        d = d.dropna(subset=[value_col])

    if d.empty:
        return pd.DataFrame()

    def _q(p: float) -> Callable[[pd.Series], float]:
        return lambda s: float(pd.to_numeric(s, errors="coerce").dropna().quantile(p)) if s.notna().any() else np.nan

    agg: Dict[str, Tuple[str, Union[str, Callable]]] = {
        count_name: (value_col, "size"),
        "mean": (value_col, "mean"),
    }

    for p in quantiles:
        key = f"q{int(round(p * 100)):02d}"
        agg[key] = (value_col, _q(float(p)))

    if extra_aggs:
        agg.update(extra_aggs)

    out = (
        d.groupby(group_cols, dropna=False)
        .agg(**agg)
        .reset_index()
    )

    for c in ["mean", *[f"q{int(round(p * 100)):02d}" for p in quantiles]]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if count_name in out.columns:
        out[count_name] = pd.to_numeric(out[count_name], errors="coerce").astype("Int64")

    return out

def breakout_event_summary(
        events: pd.DataFrame,
        *,
        group_cols: List[str],
        mfe_col: str = "mfe_sig",
        mae_col: str = "mae_sig",
        asym_col: str = "asym",
        fail_col: str = "fail_30m",
        t_mfe_col: str = "t_mfe",
        quantiles: Tuple[float, float] = (0.10, 0.90),
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    q10, q90 = quantiles

    def q(p: float) -> Callable[[pd.Series], float]:
        return lambda s: float(pd.to_numeric(s, errors="coerce").dropna().quantile(p)) if s.notna().any() else np.nan

    d = events.copy()

    for col in [mfe_col, mae_col, asym_col, t_mfe_col, fail_col]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    agg = {"n": (asym_col if asym_col in d.columns else mfe_col, "size")}

    def add_block(prefix: str, col: str):
        if col not in d.columns:
            agg[f"mean_{prefix}"] = (col, lambda _: np.nan)
            agg[f"q10_{prefix}"] = (col, lambda _: np.nan)
            agg[f"med_{prefix}"] = (col, lambda _: np.nan)
            agg[f"q90_{prefix}"] = (col, lambda _: np.nan)
            return
        agg[f"mean_{prefix}"] = (col, "mean")
        agg[f"q10_{prefix}"] = (col, q(q10))
        agg[f"med_{prefix}"] = (col, "median")
        agg[f"q90_{prefix}"] = (col, q(q90))

    add_block("mfe_sig", mfe_col)
    add_block("mae_sig", mae_col)
    add_block("asym", asym_col)

    # fail rate
    if fail_col in d.columns:
        agg["fail_rate"] = (fail_col, "mean")
    else:
        agg["fail_rate"] = (fail_col, lambda _: np.nan)

    add_block("t_mfe", t_mfe_col)

    out = (
        d.groupby(group_cols, dropna=False)
        .agg(**agg)
        .reset_index()
    )

    out["n"] = pd.to_numeric(out["n"], errors="coerce").astype("Int64")
    return out


def sign_stability_compare(
        summary_a: pd.DataFrame,
        summary_b: pd.DataFrame,
        *,
        keys: List[str],
        n_col: str = "n",
        median_col: str = "med_asym",
        suffixes: Tuple[str, str] = ("_a", "_b"),
) -> pd.DataFrame:
    a = summary_a[keys + [n_col, median_col]].rename(columns={n_col: f"{n_col}{suffixes[0]}", median_col: f"{median_col}{suffixes[0]}"})
    b = summary_b[keys + [n_col, median_col]].rename(columns={n_col: f"{n_col}{suffixes[1]}", median_col: f"{median_col}{suffixes[1]}"})

    m = a.merge(b, on=keys, how="outer")

    m[f"{n_col}_total"] = m[f"{n_col}{suffixes[0]}"].fillna(0) + m[f"{n_col}{suffixes[1]}"].fillna(0)
    m["sign_consistent"] = (
            np.sign(m[f"{median_col}{suffixes[0]}"].fillna(0.0))
            == np.sign(m[f"{median_col}{suffixes[1]}"].fillna(0.0))
    ).astype(int)

    return m.sort_values(["sign_consistent", f"{n_col}_total"], ascending=[False, False]).reset_index(drop=True)