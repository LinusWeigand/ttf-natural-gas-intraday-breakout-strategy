import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.constants import (
    DEFAULT_MAX_GAP_MINUTES,
    REGIME_ORDER_THREE_STATE,
    SESSION_END,
    SESSION_START, CRISIS_START, CRISIS_END,
)
from utils.data import load_session_df
from utils.plotting import (
    add_first_hour_marker,
    add_mean_legend_entry,
    bucket_to_minutes,
    style_axes,
)
from utils.regimes import label_crisis_regime
from utils.time_features import add_date_and_weekday_cols, between_session, minute_of_session

MODE = "volatility"

MIN_PERC_FIRST_HOUR = 0.01
MIN_PERC_TOTAL = 0.01

DATA_RESOLUTION_MINUTES = 1

MAX_GAP_MINUTES = DEFAULT_MAX_GAP_MINUTES

INTRADAY_PER_DAY_AGG = "median"
ACROSS_DAYS_AGG = "median"

MIN_DAYS_PER_BUCKET = 100

MIN_BUCKET_COVERAGE_PERC_VOLUME = 0.80

WEEKDAYS = [0, 1, 2, 3, 4]
WEEKDAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}

BUCKET_WEEKDAY = DATA_RESOLUTION_MINUTES
BUCKET_REGIME = DATA_RESOLUTION_MINUTES


def _agg_series(s: pd.Series, how: str) -> float:
    if how == "mean":
        return float(pd.to_numeric(s, errors="coerce").mean())
    if how == "median":
        return float(pd.to_numeric(s, errors="coerce").median())
    raise ValueError("how must be 'mean' or 'median'.")

def _resample_ohlcv(df: pd.DataFrame, rule: str, mode: str) -> pd.DataFrame:
    agg: dict[str, str] = {}
    if "open" in df.columns:
        agg["open"] = "first"
    if "high" in df.columns:
        agg["high"] = "max"
    if "low" in df.columns:
        agg["low"] = "min"
    agg["close"] = "last"
    if "volume" in df.columns:
        agg["volume"] = "sum"

    out = df.resample(rule).agg(agg).dropna(subset=["close"]).sort_index()

    if mode == "volume":
        out = out.dropna(subset=["volume"]).copy()
        out = out[out["volume"] >= 0].copy()

    return out

def _prepare_signal_df(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, str, str]:
    d = df.copy()

    d = between_session(d, SESSION_START, SESSION_END, inclusive="both").copy()
    d = add_date_and_weekday_cols(
        d,
        date_col="date",
        weekday_col="weekday",
        weekday_name_col="weekday_name",
        weekday_names=WEEKDAY_NAMES,
    )
    d = d[d["weekday"].isin(WEEKDAYS)].copy()

    d["minute_of_session"] = minute_of_session(pd.DatetimeIndex(d.index), SESSION_START).astype(int)

    if mode == "volatility":
        dt = d.index.to_series().diff()
        d["dt_minutes"] = dt.dt.total_seconds() / 60.0
        d["logp"] = np.log(pd.to_numeric(d["close"], errors="coerce"))
        d["log_ret"] = d["logp"].diff()

        is_new_day = d["date"] != d["date"].shift(1)
        d.loc[is_new_day, ["log_ret", "dt_minutes"]] = np.nan

        d = d[d["dt_minutes"] > 0].copy()
        d = d.dropna(subset=["log_ret", "dt_minutes"]).copy()
        if MAX_GAP_MINUTES is not None:
            d = d[d["dt_minutes"] <= float(MAX_GAP_MINUTES)].copy()

        d["signal_rate"] = (d["log_ret"] ** 2) / d["dt_minutes"]
        return d, "Relative Volatility (Normalized)", "Volatility"

    if mode == "volume":
        d["signal_rate"] = pd.to_numeric(d["volume"], errors="coerce").astype(float)
        return d, "Relative Volume (Normalized)", "Volume"

    raise ValueError("MODE must be 'volatility' or 'volume'.")

def _compute_bucket_curves_equal_day_weight(
        dsub: pd.DataFrame,
        *,
        bucket_minutes: int,
        mode: str,
        data_resolution_minutes: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    out = dsub.copy()
    out["bucket"] = (out["minute_of_session"] // int(bucket_minutes)).astype(int)

    if mode == "volume":
        obs_per_bucket = int(np.ceil(float(bucket_minutes) / float(data_resolution_minutes)))
        obs_per_bucket = max(obs_per_bucket, 1)
        min_obs = int(np.ceil(obs_per_bucket * float(MIN_BUCKET_COVERAGE_PERC_VOLUME)))
        min_obs = max(min_obs, 1)

        g = out.groupby(["date", "bucket"])["signal_rate"]
        per_day_bucket = g.apply(lambda s: _agg_series(s, INTRADAY_PER_DAY_AGG)).reset_index(name="signal")
        n_obs = g.size().reset_index(name="n_obs")
        per_day_bucket = per_day_bucket.merge(n_obs, on=["date", "bucket"], how="left")
        per_day_bucket = per_day_bucket[per_day_bucket["n_obs"] >= min_obs].copy()

        g2 = per_day_bucket.groupby("bucket")["signal"]
        primary_curve = g2.mean() if ACROSS_DAYS_AGG == "mean" else g2.median()
        mean_curve = g2.mean()
        day_counts = g2.size()

        if MIN_DAYS_PER_BUCKET is not None:
            mask = day_counts < int(MIN_DAYS_PER_BUCKET)
            primary_curve = primary_curve.copy()
            mean_curve = mean_curve.copy()
            primary_curve[mask] = np.nan
            mean_curve[mask] = np.nan

        return primary_curve, mean_curve, day_counts

    g = out.groupby(["date", "bucket"])["signal_rate"]
    per_day_bucket = g.apply(lambda s: _agg_series(s, INTRADAY_PER_DAY_AGG)).reset_index(name="rate")
    per_day_bucket["signal"] = np.sqrt(pd.to_numeric(per_day_bucket["rate"], errors="coerce"))

    g2 = per_day_bucket.groupby("bucket")["signal"]
    primary_curve = g2.mean() if ACROSS_DAYS_AGG == "mean" else g2.median()
    mean_curve = g2.mean()
    day_counts = g2.size()

    if MIN_DAYS_PER_BUCKET is not None:
        mask = day_counts < int(MIN_DAYS_PER_BUCKET)
        primary_curve = primary_curve.copy()
        mean_curve = mean_curve.copy()
        primary_curve[mask] = np.nan
        mean_curve[mask] = np.nan

    return primary_curve, mean_curve, day_counts


def _normalize_curves(primary_curve: pd.Series, mean_curve: pd.Series) -> tuple[pd.Series, pd.Series]:
    m = float(np.nanmean(primary_curve.to_numpy(dtype=float)))
    if np.isfinite(m) and m > 0:
        return primary_curve / m, mean_curve / m
    return primary_curve, mean_curve


require_cols = ["close"] if MODE == "volatility" else ["close", "volume"]
dropna_cols = ["close"] if MODE == "volatility" else ["close", "volume"]

df = load_session_df(
    min_perc_first_hour=MIN_PERC_FIRST_HOUR,
    min_perc_total=MIN_PERC_TOTAL,
    require_cols=require_cols,
    dropna_cols=dropna_cols,
    allow_negative_volume=False,
).copy()

if DATA_RESOLUTION_MINUTES is None or int(DATA_RESOLUTION_MINUTES) < 1:
    raise ValueError("DATA_RESOLUTION_MINUTES must be an integer >= 1")

DATA_RESOLUTION_MINUTES = int(DATA_RESOLUTION_MINUTES)

if DATA_RESOLUTION_MINUTES > 1:
    df = _resample_ohlcv(df, f"{DATA_RESOLUTION_MINUTES}min", MODE)

df, SIGNAL_LABEL, TITLE_ENTITY = _prepare_signal_df(df, MODE)

df["regime"] = label_crisis_regime(
    pd.DatetimeIndex(df.index),
    crisis_start=CRISIS_START,
    crisis_end=CRISIS_END,
    mode="three",
    three_labels=tuple(REGIME_ORDER_THREE_STATE),
    ordered=True,
)
df["regime"] = pd.Categorical(df["regime"], categories=REGIME_ORDER_THREE_STATE, ordered=True)

bucket_minutes = BUCKET_WEEKDAY
plt.figure(figsize=(12, 6))

counts_by_wd: dict[str, pd.Series] = {}
for wd in WEEKDAYS:
    wd_name = WEEKDAY_NAMES.get(wd, str(wd))
    dsub = df[df["weekday"] == wd].copy()

    primary_curve, mean_curve, day_counts = _compute_bucket_curves_equal_day_weight(
        dsub,
        bucket_minutes=bucket_minutes,
        mode=MODE,
        data_resolution_minutes=DATA_RESOLUTION_MINUTES,
    )
    primary_curve, mean_curve = _normalize_curves(primary_curve, mean_curve)
    counts_by_wd[wd_name] = day_counts

    x = bucket_to_minutes(primary_curve.index.to_numpy(), bucket_minutes)
    (line,) = plt.plot(x, primary_curve.to_numpy(), label=wd_name)
    plt.plot(
        x,
        mean_curve.to_numpy(),
        linestyle=":",
        linewidth=1.5,
        color=line.get_color(),
        label=None,
    )

add_first_hour_marker()
add_mean_legend_entry()
style_axes(
    title=f"Intraday {TITLE_ENTITY} Curve ({bucket_minutes}m) by Weekday",
    xlabel="Minute of Session",
    ylabel=SIGNAL_LABEL,
    grid_alpha=0.3,
    tight=True,
)
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for wd_name, c in counts_by_wd.items():
    x = bucket_to_minutes(c.index.to_numpy(), bucket_minutes)
    plt.plot(x, c.to_numpy(), label=wd_name)

add_first_hour_marker()
style_axes(
    title=f"Coverage ({bucket_minutes}m) by Weekday: Contributing Day-Count per Bucket",
    xlabel="Minute of Session",
    ylabel="Number of (day, bucket) values",
    grid_alpha=0.3,
    tight=True,
)
plt.legend()
plt.show()

bucket_minutes = BUCKET_REGIME
plt.figure(figsize=(12, 6))

counts_by_regime: dict[str, pd.Series] = {}
for regime in REGIME_ORDER_THREE_STATE:
    dsub = df[df["regime"] == regime].copy()

    primary_curve, mean_curve, day_counts = _compute_bucket_curves_equal_day_weight(
        dsub,
        bucket_minutes=bucket_minutes,
        mode=MODE,
        data_resolution_minutes=DATA_RESOLUTION_MINUTES,
    )
    primary_curve, mean_curve = _normalize_curves(primary_curve, mean_curve)
    counts_by_regime[str(regime)] = day_counts

    x = bucket_to_minutes(primary_curve.index.to_numpy(), bucket_minutes)
    (line,) = plt.plot(x, primary_curve.to_numpy(), label=str(regime))
    plt.plot(
        x,
        mean_curve.to_numpy(),
        linestyle=":",
        linewidth=1.5,
        color=line.get_color(),
        label=None,
    )

add_first_hour_marker()
add_mean_legend_entry()
style_axes(
    title=f"Intraday {TITLE_ENTITY} Curve ({bucket_minutes}m) by Regime (pre/crisis/post)",
    xlabel="Minute of Session",
    ylabel=SIGNAL_LABEL,
    grid_alpha=0.3,
    tight=True,
)
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for regime, c in counts_by_regime.items():
    x = bucket_to_minutes(c.index.to_numpy(), bucket_minutes)
    plt.plot(x, c.to_numpy(), label=regime)

add_first_hour_marker()
style_axes(
    title=f"Coverage ({bucket_minutes}m) by Regime: Contributing Day-Count per Bucket",
    xlabel="Minute of Session",
    ylabel="Number of (day, bucket) values",
    grid_alpha=0.3,
    tight=True,
)
plt.legend()
plt.show()