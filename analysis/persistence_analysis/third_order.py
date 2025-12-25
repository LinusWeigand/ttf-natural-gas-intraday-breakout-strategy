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
from utils.events import compute_excursions, first_true_cross_time
from utils.regimes import label_crisis_regime
from utils.summary import breakout_event_summary, sign_stability_compare
from utils.volatility import (
    choose_sigma_ref_price,
    first_hour_mask,
    gap_aware_sigma_log_per_minute,
    sigma_px_over_horizon,
)

EPS_SIG = 0.1
MIN_EPS_PX = 0.01

VOL_SPLIT_QUANTILE = 0.50
THIRD_ORDER_SPLIT_QUANTILE = 0.50

MIN_N_WARN = 100


def compute_first_hour_levels(day: pd.DataFrame) -> dict[str, float]:
    fh = day.loc[first_hour_mask(day.index)].copy()
    if fh.empty or ("high" not in fh.columns) or ("low" not in fh.columns):
        return {}
    hi = pd.to_numeric(fh["high"], errors="coerce").dropna()
    lo = pd.to_numeric(fh["low"], errors="coerce").dropna()
    if hi.empty or lo.empty:
        return {}
    return {"OR_HIGH": float(hi.max()), "OR_LOW": float(lo.min())}


def first_hour_open_close(day: pd.DataFrame, close_col: str = "close") -> tuple[float, float]:
    if close_col not in day.columns:
        return np.nan, np.nan
    fh = day.loc[first_hour_mask(day.index)].copy()
    fh[close_col] = pd.to_numeric(fh[close_col], errors="coerce")
    fh = fh.dropna(subset=[close_col]).sort_index()
    if fh.empty:
        return np.nan, np.nan
    return float(fh[close_col].iloc[0]), float(fh[close_col].iloc[-1])


def median_split_bucket(x: float, thr: float, low_label: str, high_label: str) -> str:
    if not np.isfinite(x) or not np.isfinite(thr):
        return "missing"
    return low_label if float(x) <= float(thr) else high_label


def value_at_asof(day: pd.DataFrame, col: str, t: pd.Timestamp) -> float:
    if col not in day.columns:
        return np.nan
    s = pd.to_numeric(day[col], errors="coerce").dropna().sort_index()
    if s.empty:
        return np.nan
    if t in s.index:
        v = s.loc[t]
        try:
            return float(v)
        except Exception:
            return np.nan
    try:
        i = s.index.get_indexer([t], method="pad")[0]
        if i >= 0:
            return float(s.iloc[i])
    except Exception:
        pass
    try:
        i = s.index.get_indexer([t], method="nearest")[0]
        return float(s.iloc[i])
    except Exception:
        return np.nan


def volume_confirmation_metric(day: pd.DataFrame, t0: pd.Timestamp) -> tuple[float, float, float]:
    if "volume" not in day.columns:
        return np.nan, np.nan, np.nan
    fh = day.loc[first_hour_mask(day.index)].copy()
    fh["volume"] = pd.to_numeric(fh["volume"], errors="coerce")
    fh = fh.dropna(subset=["volume"]).sort_index()
    if fh.empty:
        return np.nan, np.nan, np.nan
    fh_med = float(fh["volume"].median())
    if not np.isfinite(fh_med) or fh_med <= 0:
        fh_med = np.nan
    v0 = value_at_asof(day, "volume", t0)
    if not np.isfinite(v0) or v0 < 0:
        v0 = np.nan
    conf = (v0 / fh_med) if (np.isfinite(v0) and np.isfinite(fh_med) and fh_med > 0) else np.nan
    return float(v0), float(fh_med), float(conf) if np.isfinite(conf) else np.nan


def build_events(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for d, day in df.groupby("date"):
        day = day.sort_index()
        date_ts = pd.Timestamp(d)

        sigma_per_min, total_min = gap_aware_sigma_log_per_minute(
            day,
            close_col="close",
            max_gap_minutes=DEFAULT_MAX_GAP_MINUTES,
        )
        if not (np.isfinite(sigma_per_min) and sigma_per_min > 0 and np.isfinite(total_min) and total_min > 0):
            continue

        ref_price = choose_sigma_ref_price(day, prefer_vwap=False, close_col="close", volume_col="volume")
        if not (np.isfinite(ref_price) and ref_price > 0):
            continue

        sigma_px_hour = sigma_px_over_horizon(ref_price, sigma_per_min, total_min)
        if not (np.isfinite(sigma_px_hour) and sigma_px_hour > 1e-12):
            continue

        epsilon = max(float(EPS_SIG) * float(sigma_px_hour), float(MIN_EPS_PX))

        levels = compute_first_hour_levels(day)
        if not levels:
            continue

        or_high = levels.get("OR_HIGH", np.nan)
        or_low = levels.get("OR_LOW", np.nan)
        or_rng = float(or_high - or_low) if (np.isfinite(or_high) and np.isfinite(or_low)) else np.nan

        fh_open, fh_close = first_hour_open_close(day, close_col="close")
        drift_sig = ((fh_close - fh_open) / sigma_px_hour) if (np.isfinite(fh_open) and np.isfinite(fh_close) and sigma_px_hour > 0) else np.nan

        candidates = [
            ("OR_LOW_SINGLE", "down", float(levels["OR_LOW"])),
            ("OR_HIGH_SINGLE", "up", float(levels["OR_HIGH"])),
        ]

        for level_name, direction, lvl in candidates:
            t0 = first_true_cross_time(day, float(lvl), direction, close_col="close")
            if t0 is None:
                continue

            ex = compute_excursions(
                day,
                level=float(lvl),
                t0=pd.Timestamp(t0),
                direction=direction,
                epsilon=float(epsilon),
                high_col="high",
                low_col="low",
                close_col="close",
                fail_window_minutes=30,
            )
            if not (np.isfinite(ex.mfe) and np.isfinite(ex.mae)):
                continue

            close_t0 = value_at_asof(day, "close", pd.Timestamp(t0))
            dist_sig = (abs(close_t0 - float(lvl)) / sigma_px_hour) if (np.isfinite(close_t0) and sigma_px_hour > 0) else np.nan

            t_since_0800 = float(
                (pd.Timestamp(t0) - (pd.Timestamp(t0).normalize() + pd.Timedelta(hours=8))).total_seconds() / 60.0
            )

            vol_t0, fh_vol_med, vol_conf = volume_confirmation_metric(day, pd.Timestamp(t0))

            rows.append(
                {
                    "date": date_ts,
                    "level": level_name,
                    "direction": direction,
                    "t0": pd.Timestamp(t0),
                    "mfe_sig": float(ex.mfe) / float(sigma_px_hour),
                    "mae_sig": float(ex.mae) / float(sigma_px_hour),
                    "asym": (float(ex.mfe) - float(ex.mae)) / float(sigma_px_hour),
                    "t_mfe": float(ex.t_mfe_min),
                    "t_mae": float(ex.t_mae_min),
                    "fail_30m": float(ex.fail_within_window) if np.isfinite(ex.fail_within_window) else np.nan,
                    "sigma_px_hour": float(sigma_px_hour),
                    "or_range": float(or_rng) if np.isfinite(or_rng) else np.nan,
                    "drift_sig": float(drift_sig) if np.isfinite(drift_sig) else np.nan,
                    "dist_at_cross_sig": float(dist_sig) if np.isfinite(dist_sig) else np.nan,
                    "t_since_0800_min": float(t_since_0800) if np.isfinite(t_since_0800) else np.nan,
                    "vol_t0": float(vol_t0) if np.isfinite(vol_t0) else np.nan,
                    "fh_vol_med": float(fh_vol_med) if np.isfinite(fh_vol_med) else np.nan,
                    "vol_conf": float(vol_conf) if np.isfinite(vol_conf) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def add_third_order_buckets(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()

    out["range_metric"] = out["or_range"] / out["sigma_px_hour"].replace(0.0, np.nan)
    thr_range = float(out["range_metric"].dropna().quantile(THIRD_ORDER_SPLIT_QUANTILE)) if out["range_metric"].notna().any() else np.nan
    out["range_bucket"] = out["range_metric"].apply(lambda x: median_split_bucket(x, thr_range, "compression", "expansion"))

    thr_dist = float(out["dist_at_cross_sig"].dropna().quantile(THIRD_ORDER_SPLIT_QUANTILE)) if out["dist_at_cross_sig"].notna().any() else np.nan
    out["dist_bucket"] = out["dist_at_cross_sig"].apply(lambda x: median_split_bucket(x, thr_dist, "small", "large"))

    thr_time = float(out["t_since_0800_min"].dropna().quantile(THIRD_ORDER_SPLIT_QUANTILE)) if out["t_since_0800_min"].notna().any() else np.nan
    out["time_bucket"] = out["t_since_0800_min"].apply(lambda x: median_split_bucket(x, thr_time, "early", "late"))

    def drift_align_bucket(row: pd.Series) -> str:
        d = row.get("drift_sig", np.nan)
        if not np.isfinite(d):
            return "missing"
        if row.get("direction") == "up":
            return "aligned" if float(d) > 0 else "not_aligned"
        return "aligned" if float(d) < 0 else "not_aligned"

    out["drift_align_bucket"] = out.apply(drift_align_bucket, axis=1)

    thr_volconf = float(out["vol_conf"].dropna().quantile(THIRD_ORDER_SPLIT_QUANTILE)) if out["vol_conf"].notna().any() else np.nan
    out["vol_conf_bucket"] = out["vol_conf"].apply(lambda x: median_split_bucket(x, thr_volconf, "low_volconf", "high_volconf"))

    return out


def run_third_order(events: pd.DataFrame, cond_col: str, tag: str) -> None:
    group_cols = ["level", "direction", "vol_bucket", cond_col]

    summ = breakout_event_summary(
        events,
        group_cols=group_cols,
        mfe_col="mfe_sig",
        mae_col="mae_sig",
        asym_col="asym",
        fail_col="fail_30m",
        t_mfe_col="t_mfe",
        quantiles=(0.10, 0.90),
    )

    sort_cols = ["level", "direction", "vol_bucket", cond_col, "med_asym"]
    for c in sort_cols:
        if c not in summ.columns:
            summ[c] = np.nan
    summ = summ.sort_values(sort_cols, ascending=[True, True, True, True, False])

    print(f"\nTHIRD-ORDER CONDITIONING: {tag} (filtered: high-vol OR_LOW down + high-vol OR_HIGH up)")
    print(summ.to_string(index=False))

    s_cr = breakout_event_summary(
        events[events["regime"] == "crisis"],
        group_cols=group_cols,
        mfe_col="mfe_sig",
        mae_col="mae_sig",
        asym_col="asym",
        fail_col="fail_30m",
        t_mfe_col="t_mfe",
        quantiles=(0.10, 0.90),
    )
    s_nc = breakout_event_summary(
        events[events["regime"] == "non_crisis"],
        group_cols=group_cols,
        mfe_col="mfe_sig",
        mae_col="mae_sig",
        asym_col="asym",
        fail_col="fail_30m",
        t_mfe_col="t_mfe",
        quantiles=(0.10, 0.90),
    )

    merged = sign_stability_compare(
        s_cr,
        s_nc,
        keys=group_cols,
        n_col="n",
        median_col="med_asym",
        suffixes=("_crisis", "_noncrisis"),
    )

    print(f"\nSIGN STABILITY (crisis vs non-crisis) | {tag}")
    print(merged.to_string(index=False))

    out_summ = f"summary_third_order_{tag}.csv"
    out_stab = f"sign_stability_third_order_{tag}.csv"
    # summ.to_csv(out_summ, index=False)
    # merged.to_csv(out_stab, index=False)


def main() -> None:
    df = load_session_df(
        min_perc_first_hour=0.20,
        min_perc_total=0.20,
        require_cols=["close", "high", "low"],
        dropna_cols=None,
        allow_negative_volume=False,
    )
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.between_time(SESSION_START, SESSION_END, inclusive="both")
    df["date"] = pd.to_datetime(pd.DatetimeIndex(df.index).date)

    events = build_events(df)

    events["regime"] = label_crisis_regime(
        events["date"],
        crisis_start=CRISIS_START,
        crisis_end=CRISIS_END,
        mode="binary",
        binary_labels=("crisis", "non_crisis"),
        ordered=False,
    ).astype(str)

    sig_vals = events[["date", "sigma_px_hour"]].drop_duplicates().dropna()

    sig_thr = float(sig_vals["sigma_px_hour"].quantile(VOL_SPLIT_QUANTILE))
    events["vol_bucket"] = events["sigma_px_hour"].apply(lambda x: median_split_bucket(x, sig_thr, "low_vol", "high_vol"))

    keep_mask = (
        ((events["level"] == "OR_LOW_SINGLE") & (events["direction"] == "down") & (events["vol_bucket"] == "low_vol"))
        | ((events["level"] == "OR_HIGH_SINGLE") & (events["direction"] == "up") & (events["vol_bucket"] == "high_vol"))
    )
    events = events.loc[keep_mask].copy()


    events = add_third_order_buckets(events)

    # events.to_csv("events_third_order.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 180)

    print("\nSECOND-ORDER VOL SPLIT (median)")
    print(f"median sigma_px_hour threshold: {sig_thr:.6f}")
    print(events["vol_bucket"].value_counts(dropna=False).to_string())


    run_third_order(events, "range_bucket", "range_regime")
    run_third_order(events, "dist_bucket", "distance_at_cross")
    run_third_order(events, "time_bucket", "time_of_cross")
    run_third_order(events, "drift_align_bucket", "drift_alignment")
    run_third_order(events, "vol_conf_bucket", "volume_confirmation")


if __name__ == "__main__":
    main()
