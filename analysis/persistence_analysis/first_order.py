import numpy as np
import pandas as pd
from utils.constants import (
    CRISIS_END,
    CRISIS_START,
    DEFAULT_MAX_GAP_MINUTES,
    SESSION_END,
    SESSION_START,
)
from utils.data import load_session_df
from utils.events import (
    compute_excursions,
    first_confluence_cross_time,
    first_true_cross_time,
)
from utils.regimes import label_crisis_regime
from utils.summary import breakout_event_summary, sign_stability_compare
from utils.time_features import add_date_and_weekday_cols, ensure_time
from utils.volatility import (
    gap_aware_sigma_log_per_minute,
    choose_sigma_ref_price,
    sigma_px_over_horizon,
    first_hour_mask,
)

USE_CONFLUENCE = False

MAX_GAP_MINUTES = DEFAULT_MAX_GAP_MINUTES

ATR_DAYS = 14
ATR_R_VALUES = [0.25, 0.5, 0.75, 1.0]

VWAP_SIGMA_VALUES = [0.5, 1.0, 1.5, 2.0]

EPS_SIG = 0.1
MIN_EPS_PX = 0.01

CONFLUENCE_DEFS = [
    {"name": "OR_HIGH_AND_VWAP_UP", "legs": [("OR_HIGH", "up"), ("VWAP", "up")]},
    {"name": "OR_HIGH_AND_VWAP_PLUS_0.5SIG_UP", "legs": [("OR_HIGH", "up"), ("VWAP_PLUS_0.5SIG", "up")]},
    {"name": "OR_LOW_AND_VWAP_MINUS_0.5SIG_DOWN", "legs": [("OR_LOW", "down"), ("VWAP_MINUS_0.5SIG", "down")]},
]


def _fmt(x: float) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else str(x)


def compute_daily_atr(df: pd.DataFrame, atr_days: int = 14) -> pd.Series:
    daily = (
        df.dropna(subset=["high", "low", "close"])
        .groupby("date")
        .agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .sort_index()
    )
    if daily.empty or len(daily) < 2:
        return pd.Series(dtype=float)

    prev_close = daily["close"].shift(1)
    tr = np.maximum(
        daily["high"] - daily["low"],
        np.maximum((daily["high"] - prev_close).abs(), (daily["low"] - prev_close).abs()),
    )

    atr = tr.rolling(int(atr_days), min_periods=int(atr_days)).mean()
    atr.index = pd.to_datetime(atr.index)
    return atr


def first_hour_levels(
    day: pd.DataFrame,
    *,
    atr_px: float,
    atr_r_values: list[float],
    vwap_sigma_values: list[float],
    sigma_px_hour: float,
) -> dict[str, float]:
    ss = ensure_time(SESSION_START)
    fh = day.loc[first_hour_mask(day.index, start=ss)]
    fh = fh[fh.index.time < pd.to_datetime("08:00").time()]
    if fh.empty:
        return {}

    levels: dict[str, float] = {}
    levels["OR_HIGH"] = float(pd.to_numeric(fh["high"], errors="coerce").max())
    levels["OR_LOW"] = float(pd.to_numeric(fh["low"], errors="coerce").min())

    vwap = np.nan
    if "volume" in fh.columns:
        tmp = fh.dropna(subset=["close", "volume"]).copy()
        tmp["close"] = pd.to_numeric(tmp["close"], errors="coerce")
        tmp["volume"] = pd.to_numeric(tmp["volume"], errors="coerce")
        tmp = tmp[(tmp["volume"] > 0) & tmp["close"].notna()]
        if not tmp.empty:
            vwap = float((tmp["close"] * tmp["volume"]).sum() / tmp["volume"].sum())
            if np.isfinite(vwap):
                levels["VWAP"] = vwap

    if np.isfinite(vwap) and np.isfinite(sigma_px_hour) and float(sigma_px_hour) > 0:
        for k in vwap_sigma_values:
            if not (np.isfinite(k) and float(k) > 0):
                continue
            kk = _fmt(float(k))
            levels[f"VWAP_PLUS_{kk}SIG"] = float(vwap) + float(k) * float(sigma_px_hour)
            levels[f"VWAP_MINUS_{kk}SIG"] = float(vwap) - float(k) * float(sigma_px_hour)

    anchor = float(vwap) if np.isfinite(vwap) else choose_sigma_ref_price(day, prefer_vwap=False)
    if np.isfinite(anchor) and np.isfinite(atr_px) and float(atr_px) > 0:
        for r in atr_r_values:
            if not (np.isfinite(r) and float(r) > 0):
                continue
            rr = _fmt(float(r))
            levels[f"ATR_{ATR_DAYS}D_PLUS_{rr}R"] = float(anchor) + float(r) * float(atr_px)
            levels[f"ATR_{ATR_DAYS}D_MINUS_{rr}R"] = float(anchor) - float(r) * float(atr_px)

    return levels


def main():
    df = load_session_df(
        min_perc_first_hour=0.20,
        min_perc_total=0.20,
        require_cols=["close", "high", "low"],
        allow_negative_volume=True,
    )
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.between_time(SESSION_START, SESSION_END, inclusive="both").copy()
    df = add_date_and_weekday_cols(df, date_col="date").copy()

    atr_by_date = compute_daily_atr(df, atr_days=ATR_DAYS)

    rows: list[dict] = []

    for d, day in df.groupby("date"):
        day = day.sort_index()
        date_ts = pd.Timestamp(d)

        sigma_per_min, total_min = gap_aware_sigma_log_per_minute(
            day,
            close_col="close",
            fh_start=ensure_time(SESSION_START),
            fh_end=pd.to_datetime("08:00").time(),
            max_gap_minutes=MAX_GAP_MINUTES,
        )
        if not (np.isfinite(sigma_per_min) and sigma_per_min > 0 and np.isfinite(total_min) and total_min > 0):
            continue

        ref_price = choose_sigma_ref_price(
            day,
            prefer_vwap=True,
            fh_start=ensure_time(SESSION_START),
            fh_end=pd.to_datetime("08:00").time(),
            close_col="close",
            volume_col="volume",
        )
        if not (np.isfinite(ref_price) and ref_price > 0):
            continue

        sigma_px_hour = sigma_px_over_horizon(ref_price, sigma_per_min, total_min)
        if not (np.isfinite(sigma_px_hour) and sigma_px_hour > 1e-12):
            continue

        epsilon = max(float(EPS_SIG) * float(sigma_px_hour), float(MIN_EPS_PX))

        atr_px = float(atr_by_date.get(date_ts, np.nan))

        levels = first_hour_levels(
            day,
            atr_px=atr_px,
            atr_r_values=ATR_R_VALUES,
            vwap_sigma_values=VWAP_SIGMA_VALUES,
            sigma_px_hour=sigma_px_hour,
        )
        if not levels:
            continue

        if not USE_CONFLUENCE:
            for level_name, lvl in levels.items():
                if lvl is None or not np.isfinite(lvl):
                    continue

                for direction in ("up", "down"):
                    t0 = first_true_cross_time(
                        day,
                        float(lvl),
                        direction,
                        after_time=pd.to_datetime("08:00").time(),
                        close_col="close",
                    )
                    if t0 is None:
                        continue

                    ex = compute_excursions(
                        day,
                        float(lvl),
                        pd.Timestamp(t0),
                        direction,
                        float(epsilon),
                        high_col="high",
                        low_col="low",
                        close_col="close",
                        fail_window_minutes=30,
                    )
                    if not (np.isfinite(ex.mfe) and np.isfinite(ex.mae)):
                        continue

                    rows.append(
                        {
                            "date": date_ts,
                            "level": level_name,
                            "direction": direction,
                            "mfe_sig": float(ex.mfe) / float(sigma_px_hour),
                            "mae_sig": float(ex.mae) / float(sigma_px_hour),
                            "asym": (float(ex.mfe) - float(ex.mae)) / float(sigma_px_hour),
                            "t_mfe": float(ex.t_mfe_min),
                            "t_mae": float(ex.t_mae_min),
                            "fail_30m": float(ex.fail_within_window),
                        }
                    )
        else:
            for spec in CONFLUENCE_DEFS:
                t0 = first_confluence_cross_time(
                    day,
                    levels,
                    spec["legs"],
                    after_time=pd.to_datetime("08:00").time(),
                    close_col="close",
                )
                if t0 is None:
                    continue

                ref_level_name, ref_dir = spec["legs"][0]
                ref_level = levels.get(ref_level_name, np.nan)
                if ref_level is None or not np.isfinite(ref_level):
                    continue

                ex = compute_excursions(
                    day,
                    float(ref_level),
                    pd.Timestamp(t0),
                    ref_dir,
                    float(epsilon),
                    high_col="high",
                    low_col="low",
                    close_col="close",
                    fail_window_minutes=30,
                )
                if not (np.isfinite(ex.mfe) and np.isfinite(ex.mae)):
                    continue

                rows.append(
                    {
                        "date": date_ts,
                        "level": spec["name"],
                        "direction": ref_dir,
                        "mfe_sig": float(ex.mfe) / float(sigma_px_hour),
                        "mae_sig": float(ex.mae) / float(sigma_px_hour),
                        "asym": (float(ex.mfe) - float(ex.mae)) / float(sigma_px_hour),
                        "t_mfe": float(ex.t_mfe_min),
                        "t_mae": float(ex.t_mae_min),
                        "fail_30m": float(ex.fail_within_window),
                    }
                )

    events = pd.DataFrame(rows)

    events["regime"] = label_crisis_regime(
        events["date"],
        crisis_start=CRISIS_START,
        crisis_end=CRISIS_END,
        mode="binary",
        binary_labels=("crisis", "non_crisis"),
        ordered=False,
    ).astype(str)

    summary_all = breakout_event_summary(
        events,
        group_cols=["level", "direction"],
        mfe_col="mfe_sig",
        mae_col="mae_sig",
        asym_col="asym",
        fail_col="fail_30m",
        t_mfe_col="t_mfe",
        quantiles=(0.10, 0.90),
    ).sort_values("med_asym", ascending=False)

    summary_crisis = breakout_event_summary(
        events[events["regime"] == "crisis"],
        group_cols=["level", "direction"],
        mfe_col="mfe_sig",
        mae_col="mae_sig",
        asym_col="asym",
        fail_col="fail_30m",
        t_mfe_col="t_mfe",
        quantiles=(0.10, 0.90),
    ).sort_values("med_asym", ascending=False)

    summary_non_crisis = breakout_event_summary(
        events[events["regime"] == "non_crisis"],
        group_cols=["level", "direction"],
        mfe_col="mfe_sig",
        mae_col="mae_sig",
        asym_col="asym",
        fail_col="fail_30m",
        t_mfe_col="t_mfe",
        quantiles=(0.10, 0.90),
    ).sort_values("med_asym", ascending=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 80)

    print("\nFULL")
    print(summary_all.to_string(index=False))

    print(f"\nCRISIS REGIME ({pd.Timestamp(CRISIS_START).date()} to {pd.Timestamp(CRISIS_END).date()})")
    print(summary_crisis.to_string(index=False))

    print("\nNON-CRISIS REGIME")
    print(summary_non_crisis.to_string(index=False))

    merged = sign_stability_compare(
        summary_crisis,
        summary_non_crisis,
        keys=["level", "direction"],
        n_col="n",
        median_col="med_asym",
        suffixes=("_crisis", "_noncrisis"),
    )

    print("\nSIGN STABILITY (crisis vs non-crisis)")
    print(
        merged.sort_values(
            ["sign_consistent", "n_total"],
            ascending=[False, False],
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
