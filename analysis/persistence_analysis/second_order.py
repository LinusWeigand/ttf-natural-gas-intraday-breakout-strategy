import numpy as np
import pandas as pd
from utils.constants import (
    CRISIS_END,
    CRISIS_START,
    DEFAULT_MAX_GAP_MINUTES,
    SESSION_END,
    SESSION_START,
)
from utils.data import load_filtered_df
from utils.events import combo_cross_time, compute_excursions, first_true_cross_time
from utils.regimes import label_crisis_regime
from utils.summary import breakout_event_summary, sign_stability_compare
from utils.time_features import add_date_and_weekday_cols, between_session
from utils.volatility import (
    choose_sigma_ref_price,
    first_hour_mask,
    gap_aware_sigma_log_per_minute,
    sigma_px_over_horizon,
)

MAX_GAP_MINUTES = DEFAULT_MAX_GAP_MINUTES

OBJECTS_OF_INTEREST = [
    {
        "name": "OR_HIGH_AND_VWAP_UP",
        "primary_level": "OR_HIGH",
        "confirm_level": "VWAP",
        "direction": "up",
    },
    {
        "name": "OR_HIGH_AND_VWAP_PLUS_0.5SIG_UP",
        "primary_level": "OR_HIGH",
        "confirm_level": "VWAP_PLUS_0.5SIG",
        "direction": "up",
    },
    {
        "name": "OR_LOW_AND_VWAP_MINUS_0.5SIG_DOWN",
        "primary_level": "OR_LOW",
        "confirm_level": "VWAP_MINUS_0.5SIG",
        "direction": "down",
    },
]

INCLUDE_SINGLES = True

EPS_SIG = 0.1
MIN_EPS_PX = 0.01

VOL_SPLIT_QUANTILE = 0.50


def compute_first_hour_levels(day: pd.DataFrame, sigma_px_hour: float) -> dict[str, float]:
    fh = day.loc[first_hour_mask(day.index)].copy()
    if fh.empty:
        return {}

    levels: dict[str, float] = {}

    if "high" in fh.columns and fh["high"].notna().any():
        levels["OR_HIGH"] = float(pd.to_numeric(fh["high"], errors="coerce").max())
    if "low" in fh.columns and fh["low"].notna().any():
        levels["OR_LOW"] = float(pd.to_numeric(fh["low"], errors="coerce").min())

    vwap = np.nan
    if "volume" in fh.columns and "close" in fh.columns:
        tmp = fh.dropna(subset=["close", "volume"]).copy()
        tmp["close"] = pd.to_numeric(tmp["close"], errors="coerce")
        tmp["volume"] = pd.to_numeric(tmp["volume"], errors="coerce")
        tmp = tmp[(tmp["close"] > 0) & (tmp["volume"] > 0)]
        if not tmp.empty:
            vwap = float((tmp["close"] * tmp["volume"]).sum() / tmp["volume"].sum())
            if np.isfinite(vwap):
                levels["VWAP"] = vwap

    if np.isfinite(vwap) and np.isfinite(sigma_px_hour) and float(sigma_px_hour) > 0:
        levels["VWAP_PLUS_0.5SIG"] = float(vwap) + 0.5 * float(sigma_px_hour)
        levels["VWAP_MINUS_0.5SIG"] = float(vwap) - 0.5 * float(sigma_px_hour)

    return levels


def vol_bucketize(vol_by_date: pd.Series, split_q: float = 0.50) -> pd.Series:
    v = pd.to_numeric(vol_by_date, errors="coerce").dropna().astype(float)
    if v.empty:
        return pd.Series(dtype=str)
    thr = float(v.quantile(float(split_q)))
    out = pd.Series(index=v.index, dtype=object)
    out.loc[v <= thr] = "low_vol"
    out.loc[v > thr] = "high_vol"
    return out


def make_singular_requests(objects: list[dict], include: bool) -> list[dict]:
    if not include:
        return []
    req = []
    for obj in objects:
        req.append(
            {
                "name": f"{obj['primary_level']}_SINGLE",
                "level_key": obj["primary_level"],
                "direction": obj["direction"],
            }
        )
        req.append(
            {
                "name": f"{obj['confirm_level']}_SINGLE",
                "level_key": obj["confirm_level"],
                "direction": obj["direction"],
            }
        )
    seen = set()
    uniq = []
    for r in req:
        k = (r["name"], r["level_key"], r["direction"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


def sign_stability_low_vs_high(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame(
            columns=[
                "level",
                "direction",
                "n_low",
                "med_asym_low",
                "n_high",
                "med_asym_high",
                "n_total",
                "sign_consistent",
            ]
        )

    low = summary_df[summary_df["vol_bucket"] == "low_vol"].copy()
    high = summary_df[summary_df["vol_bucket"] == "high_vol"].copy()

    keys = ["level", "direction"]
    merged = sign_stability_compare(
        low,
        high,
        keys=keys,
        n_col="n",
        median_col="med_asym",
        suffixes=("_low", "_high"),
    )

    merged = merged.rename(
        columns={
            "n_low": "n_low",
            "med_asym_low": "med_asym_low",
            "n_high": "n_high",
            "med_asym_high": "med_asym_high",
        }
    )

    keep = [c for c in ["level", "direction", "n_low", "med_asym_low", "n_high", "med_asym_high", "n_total", "sign_consistent"] if c in merged.columns]
    return merged[keep].sort_values(["sign_consistent", "n_total"], ascending=[False, False]).reset_index(drop=True)


def compute_sigma_px_hour_by_date(df: pd.DataFrame) -> pd.Series:
    out: dict[pd.Timestamp, float] = {}
    for d, day in df.groupby("date"):
        day = day.sort_index()
        date_ts = pd.Timestamp(d)

        sigma_per_min, total_min = gap_aware_sigma_log_per_minute(
            day,
            close_col="close",
            max_gap_minutes=MAX_GAP_MINUTES,
        )
        if not (np.isfinite(sigma_per_min) and float(sigma_per_min) > 0 and np.isfinite(total_min) and float(total_min) > 0):
            continue

        ref_price = choose_sigma_ref_price(day, prefer_vwap=True, close_col="close", volume_col="volume")
        if not (np.isfinite(ref_price) and float(ref_price) > 0):
            continue

        sigma_px_hour = sigma_px_over_horizon(float(ref_price), float(sigma_per_min), float(total_min))
        if not (np.isfinite(sigma_px_hour) and float(sigma_px_hour) > 1e-12):
            continue

        out[date_ts] = float(sigma_px_hour)

    return pd.Series(out).sort_index()


def build_events(
    df: pd.DataFrame,
    sigma_px_by_date: pd.Series,
    vol_bucket_by_date: pd.Series,
    singular_requests: list[dict],
) -> pd.DataFrame:
    rows: list[dict] = []

    for d, day in df.groupby("date"):
        day = day.sort_index()
        date_ts = pd.Timestamp(d)

        if date_ts not in sigma_px_by_date.index or date_ts not in vol_bucket_by_date.index:
            continue

        sigma_px_hour = float(sigma_px_by_date.loc[date_ts])
        vol_bucket = str(vol_bucket_by_date.loc[date_ts])

        epsilon = max(EPS_SIG * sigma_px_hour, MIN_EPS_PX)

        levels = compute_first_hour_levels(day, sigma_px_hour=sigma_px_hour)
        if not levels:
            continue

        for obj in OBJECTS_OF_INTEREST:
            primary_key = obj["primary_level"]
            confirm_key = obj["confirm_level"]
            direction = obj["direction"]
            label = obj["name"]

            if primary_key not in levels or confirm_key not in levels:
                continue

            primary_lvl = float(levels[primary_key])
            confirm_lvl = float(levels[confirm_key])

            t0 = combo_cross_time(day, primary_lvl, confirm_lvl, direction)
            if t0 is None:
                continue

            ex = compute_excursions(day, primary_lvl, pd.Timestamp(t0), direction, epsilon, fail_window_minutes=30)
            if not (np.isfinite(ex.mfe) and np.isfinite(ex.mae)):
                continue

            rows.append(
                {
                    "date": date_ts,
                    "vol_bucket": vol_bucket,
                    "level": label,
                    "direction": direction,
                    "mfe_sig": ex.mfe / sigma_px_hour,
                    "mae_sig": ex.mae / sigma_px_hour,
                    "asym": (ex.mfe - ex.mae) / sigma_px_hour,
                    "t_mfe": ex.t_mfe_min,
                    "t_mae": ex.t_mae_min,
                    "fail_30m": ex.fail_within_window,
                    "sigma_px_hour": sigma_px_hour,
                }
            )

        for s in singular_requests:
            label = s["name"]
            level_key = s["level_key"]
            direction = s["direction"]

            if level_key not in levels:
                continue

            lvl = float(levels[level_key])
            t0 = first_true_cross_time(day, lvl, direction)
            if t0 is None:
                continue

            ex = compute_excursions(day, lvl, pd.Timestamp(t0), direction, epsilon, fail_window_minutes=30)
            if not (np.isfinite(ex.mfe) and np.isfinite(ex.mae)):
                continue

            rows.append(
                {
                    "date": date_ts,
                    "vol_bucket": vol_bucket,
                    "level": label,
                    "direction": direction,
                    "mfe_sig": ex.mfe / sigma_px_hour,
                    "mae_sig": ex.mae / sigma_px_hour,
                    "asym": (ex.mfe - ex.mae) / sigma_px_hour,
                    "t_mfe": ex.t_mfe_min,
                    "t_mae": ex.t_mae_min,
                    "fail_30m": ex.fail_within_window,
                    "sigma_px_hour": sigma_px_hour,
                }
            )

    return pd.DataFrame(rows)


def main():
    df = load_filtered_df(min_perc_first_hour=0.20, min_perc_total=0.20)
    df.columns = [c.strip().lower() for c in df.columns]
    df = between_session(df, SESSION_START, SESSION_END, inclusive="both")
    df = add_date_and_weekday_cols(df, date_col="date")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    sigma_px_by_date = compute_sigma_px_hour_by_date(df)

    vol_bucket_by_date = vol_bucketize(sigma_px_by_date, split_q=VOL_SPLIT_QUANTILE)

    singular_requests = make_singular_requests(OBJECTS_OF_INTEREST, INCLUDE_SINGLES)
    events = build_events(df, sigma_px_by_date, vol_bucket_by_date, singular_requests)

    events["regime"] = label_crisis_regime(
        events["date"],
        crisis_start=CRISIS_START,
        crisis_end=CRISIS_END,
        mode="binary",
        binary_labels=("crisis", "non_crisis"),
        ordered=False,
    ).astype(str)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 140)

    thr = float(pd.to_numeric(sigma_px_by_date, errors="coerce").quantile(VOL_SPLIT_QUANTILE))
    print("\nVOL BUCKET SPLIT")
    print(f"Split quantile: {VOL_SPLIT_QUANTILE:.2f}   sigma_hour threshold: {thr:.6f}")
    print(f"Dates in low_vol:  {(vol_bucket_by_date == 'low_vol').sum()}")
    print(f"Dates in high_vol: {(vol_bucket_by_date == 'high_vol').sum()}")

    print("\nREGIME SPLIT")
    print(f"Crisis: {pd.Timestamp(CRISIS_START).date()} to {pd.Timestamp(CRISIS_END).date()}")
    print("Non-crisis: all other dates")
    print("Events in crisis: ", int((events["regime"] == "crisis").sum()))
    print("Events in noncrisis: ", int((events["regime"] == "non_crisis").sum()))

    group_cols = ["level", "direction", "vol_bucket"]

    summary_full = breakout_event_summary(events, group_cols=group_cols)
    summary_crisis = breakout_event_summary(events[events["regime"] == "crisis"], group_cols=group_cols)
    summary_noncrisis = breakout_event_summary(events[events["regime"] == "non_crisis"], group_cols=group_cols)

    print("\nSECOND-ORDER: FULL")
    print(summary_full.to_string(index=False))

    print("\nSECOND-ORDER: CRISIS")
    print(summary_crisis.to_string(index=False))

    print("\nSECOND-ORDER: NON-CRISIS")
    print(summary_noncrisis.to_string(index=False))

    stab_full = sign_stability_low_vs_high(summary_full)
    stab_crisis = sign_stability_low_vs_high(summary_crisis)
    stab_noncrisis = sign_stability_low_vs_high(summary_noncrisis)

    print("\nSIGN STABILITY (low_vol vs high_vol): FULL")
    print(stab_full.to_string(index=False))

    print("\nSIGN STABILITY (low_vol vs high_vol) CRISIS")
    print(stab_crisis.to_string(index=False))

    print("\nSIGN STABILITY (low_vol vs high_vol) NON-CRISIS")
    print(stab_noncrisis.to_string(index=False))

    # events.to_csv("events_second_order_vol.csv", index=False)
    # summary_full.to_csv("summary_second_order_vol_full.csv", index=False)
    # summary_crisis.to_csv("summary_second_order_vol_crisis.csv", index=False)
    # summary_noncrisis.to_csv("summary_second_order_vol_noncrisis.csv", index=False)
    # stab_full.to_csv("summary_second_order_vol_sign_stability_full.csv", index=False)
    # stab_crisis.to_csv("summary_second_order_vol_sign_stability_crisis.csv", index=False)
    # stab_noncrisis.to_csv("summary_second_order_vol_sign_stability_noncrisis.csv", index=False)

if __name__ == "__main__":
    main()
