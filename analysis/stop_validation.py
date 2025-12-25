from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.constants import DEFAULT_MAX_GAP_MINUTES, SESSION_END, SESSION_START
from utils.data import load_session_df
from utils.events import first_true_cross_time
from utils.plotting import style_axes
from utils.regimes import label_crisis_regime
from utils.time_features import add_date_and_weekday_cols, between_session
from utils.volatility import (
    choose_sigma_ref_price,
    first_hour_mask,
    gap_aware_sigma_log_per_minute,
    sigma_px_over_horizon,
)

VOL_SPLIT_QUANTILE = 0.50
EPS_FAIL_SIG = 0.10
HARD_STOP_DELAY_MIN = 30

TP_K_LIST = [0.5, 1.0, 2.0]
SL_S = 1.25
STOP_SWEEP_LIST = [0.75, 1.0, 1.25]


def compute_or_levels(day: pd.DataFrame) -> tuple[float | None, float | None]:
    fh = day.loc[first_hour_mask(day.index)].dropna(subset=["high", "low"])
    if fh.empty:
        return None, None
    or_high = float(pd.to_numeric(fh["high"], errors="coerce").max())
    or_low = float(pd.to_numeric(fh["low"], errors="coerce").min())
    if not (np.isfinite(or_high) and np.isfinite(or_low) and or_high > or_low):
        return None, None
    return or_high, or_low


def pick_first_signal(day: pd.DataFrame, or_high: float, or_low: float) -> tuple[str | None, pd.Timestamp | None]:
    t_long = first_true_cross_time(day, or_high, "up")
    t_short = first_true_cross_time(day, or_low, "down")
    if t_long is None and t_short is None:
        return None, None
    if t_long is None:
        return "short", t_short
    if t_short is None:
        return "long", t_long
    return ("long", t_long) if t_long <= t_short else ("short", t_short)


def next_bar_open_or_close(day: pd.DataFrame, t_signal: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    d = day.sort_index()
    if t_signal not in d.index:
        try:
            pos = d.index.get_indexer([t_signal], method="pad")[0]
            if pos < 0:
                return None, None
            t_signal = d.index[pos]
        except Exception:
            return None, None

    pos = d.index.get_loc(t_signal)
    if isinstance(pos, slice):
        pos = pos.start
    nxt = pos + 1
    if 0 <= nxt < len(d.index) and "open" in d.columns:
        t_entry = d.index[nxt]
        px = pd.to_numeric(d.at[t_entry, "open"], errors="coerce")
        if np.isfinite(px):
            return t_entry, float(px)

    px = pd.to_numeric(d.at[t_signal, "close"], errors="coerce")
    return (t_signal, float(px)) if np.isfinite(px) else (None, None)


def end_of_day_time(day: pd.DataFrame) -> pd.Timestamp | None:
    d = day.dropna(subset=["close"]).sort_index()
    return None if d.empty else pd.Timestamp(d.index[-1])


def find_structure_failure_time(
    day: pd.DataFrame,
    t_entry: pd.Timestamp,
    side: str,
    or_high: float,
    or_low: float,
    eps_fail_px: float,
    window_min: int = 30,
) -> pd.Timestamp | None:
    w = day.loc[t_entry : t_entry + pd.Timedelta(minutes=int(window_min))].dropna(subset=["close"]).sort_index()
    if w.empty:
        return None
    close = pd.to_numeric(w["close"], errors="coerce")
    if side == "long":
        trig = close <= float(or_high - eps_fail_px)
    else:
        trig = close >= float(or_low + eps_fail_px)
    idx = w.index[trig.fillna(False)]
    return None if len(idx) == 0 else pd.Timestamp(idx[0])


def first_hit_time_barriers(
    day: pd.DataFrame,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
    side: str,
    entry_px: float,
    sigma_px: float,
    k_tp: float,
    s_sl: float,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    if not (np.isfinite(entry_px) and np.isfinite(sigma_px) and sigma_px > 0 and np.isfinite(k_tp) and np.isfinite(s_sl)):
        return None, None
    w = day.loc[t_start:t_end].dropna(subset=["high", "low"]).sort_index()
    if w.empty:
        return None, None

    high = pd.to_numeric(w["high"], errors="coerce")
    low = pd.to_numeric(w["low"], errors="coerce")

    tp_dist = float(k_tp) * float(sigma_px)
    sl_dist = float(s_sl) * float(sigma_px)

    if side == "long":
        tp_level = float(entry_px + tp_dist)
        sl_level = float(entry_px - sl_dist)
        hit_tp = high >= tp_level
        hit_sl = low <= sl_level
    else:
        tp_level = float(entry_px - tp_dist)
        sl_level = float(entry_px + sl_dist)
        hit_tp = low <= tp_level
        hit_sl = high >= sl_level

    t_tp = None
    t_sl = None

    idx_tp = w.index[hit_tp.fillna(False)]
    if len(idx_tp) > 0:
        t_tp = pd.Timestamp(idx_tp[0])

    idx_sl = w.index[hit_sl.fillna(False)]
    if len(idx_sl) > 0:
        t_sl = pd.Timestamp(idx_sl[0])

    return t_tp, t_sl


def mins_between(t0: pd.Timestamp, t1: pd.Timestamp | None) -> float:
    return np.nan if t1 is None else float((pd.Timestamp(t1) - pd.Timestamp(t0)).total_seconds() / 60.0)


def mfe_mae_to_eod_sig(
    day: pd.DataFrame,
    t_entry: pd.Timestamp,
    t_eod: pd.Timestamp,
    side: str,
    entry_px: float,
    sigma_px_hour: float,
) -> tuple[float, float]:
    if not (np.isfinite(entry_px) and np.isfinite(sigma_px_hour) and sigma_px_hour > 0 and t_eod >= t_entry):
        return np.nan, np.nan
    w = day.loc[t_entry:t_eod].dropna(subset=["high", "low"]).sort_index()
    if w.empty:
        return np.nan, np.nan

    high = pd.to_numeric(w["high"], errors="coerce")
    low = pd.to_numeric(w["low"], errors="coerce")

    if side == "long":
        fav = high - float(entry_px)
        adv = float(entry_px) - low
    else:
        fav = float(entry_px) - low
        adv = high - float(entry_px)

    mfe_px = float(fav.max()) if len(fav) else np.nan
    mae_px = float(adv.max()) if len(adv) else np.nan
    return (mfe_px / float(sigma_px_hour), mae_px / float(sigma_px_hour))


def barrier_race_summary(diag_df: pd.DataFrame, k_tp: float) -> pd.DataFrame:
    col_order = f"race_k{str(k_tp).replace('.', '_')}"
    col_ttp = f"time_to_tp_k{str(k_tp).replace('.', '_')}_min"
    col_tsl = f"time_to_sl_k{str(k_tp).replace('.', '_')}_min"
    if diag_df.empty or col_order not in diag_df.columns:
        return pd.DataFrame()

    d = diag_df.copy()
    total = int(len(d))
    counts = d[col_order].value_counts(dropna=False)

    med_t_tp = float(d.loc[d[col_order] == "tp_first", col_ttp].median()) if (d[col_order] == "tp_first").any() else np.nan
    med_t_sl = float(d.loc[d[col_order] == "sl_first", col_tsl].median()) if (d[col_order] == "sl_first").any() else np.nan

    return pd.DataFrame(
        [
            {
                "k_tp": float(k_tp),
                "n": total,
                "tp_first_n": int(counts.get("tp_first", 0)),
                "sl_first_n": int(counts.get("sl_first", 0)),
                "no_hit_n": int(counts.get("no_hit", 0)),
                "tp_first_rate": float(counts.get("tp_first", 0) / total) if total else np.nan,
                "sl_first_rate": float(counts.get("sl_first", 0) / total) if total else np.nan,
                "no_hit_rate": float(counts.get("no_hit", 0) / total) if total else np.nan,
                "median_time_to_tp_if_tp_first_min": med_t_tp,
                "median_time_to_sl_if_sl_first_min": med_t_sl,
            }
        ]
    )


def latent_winner_summary(diag_df: pd.DataFrame, k_tp: float) -> pd.DataFrame:
    col = f"latent_tp_k{str(k_tp).replace('.', '_')}"
    if diag_df.empty or col not in diag_df.columns:
        return pd.DataFrame()
    sf = diag_df[diag_df["would_structure_fail"] == 1].copy()
    n = int(len(sf))
    if n == 0:
        return pd.DataFrame([{"k_tp": float(k_tp), "n_structure_fail": 0, "latent_tp_rate": np.nan}])
    rate = float(pd.to_numeric(sf[col], errors="coerce").mean())
    return pd.DataFrame([{"k_tp": float(k_tp), "n_structure_fail": n, "latent_tp_rate": rate}])


def sf_vs_nosf_path_summary(diag_df: pd.DataFrame) -> pd.DataFrame:
    if diag_df.empty:
        return pd.DataFrame()
    d = diag_df.dropna(subset=["would_structure_fail", "mfe_sig_to_eod", "mae_sig_to_eod"]).copy()
    if d.empty:
        return pd.DataFrame()

    rows = []
    for val, name in [(1, "would_structure_fail"), (0, "no_structure_fail")]:
        g = d[d["would_structure_fail"] == val]
        if g.empty:
            continue
        rows.append(
            {
                "group": name,
                "n": int(len(g)),
                "mfe_med": float(pd.to_numeric(g["mfe_sig_to_eod"], errors="coerce").median()),
                "mae_med": float(pd.to_numeric(g["mae_sig_to_eod"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def would_hit_hard_stop_after_delay(
    day: pd.DataFrame,
    t_entry: pd.Timestamp,
    t_eod: pd.Timestamp,
    side: str,
    entry_px: float,
    sigma_px_hour: float,
    stop_sigma: float,
    delay_min: int,
) -> int | None:
    if not (np.isfinite(entry_px) and np.isfinite(sigma_px_hour) and sigma_px_hour > 0 and np.isfinite(stop_sigma)):
        return None
    if t_eod is None or t_eod < t_entry:
        return None

    t0 = t_entry + pd.Timedelta(minutes=int(delay_min)) if delay_min and delay_min > 0 else t_entry
    if t0 > t_eod:
        return None

    w = day.loc[t0:t_eod].dropna(subset=["high", "low"]).sort_index()
    if w.empty:
        return None

    high = pd.to_numeric(w["high"], errors="coerce")
    low = pd.to_numeric(w["low"], errors="coerce")

    stop_dist = float(stop_sigma) * float(sigma_px_hour)
    if side == "long":
        level = float(entry_px - stop_dist)
        hit = bool((low <= level).any())
    else:
        level = float(entry_px + stop_dist)
        hit = bool((high >= level).any())

    return 1 if hit else 0


def hard_stop_path_classifier_table(entries_df: pd.DataFrame, label_col: str, mfe_col: str, mae_col: str) -> pd.DataFrame:
    d = entries_df.dropna(subset=[label_col, mfe_col, mae_col]).copy()
    if d.empty:
        return pd.DataFrame()
    rows = []
    for val, name in [(1, "would_hit_hard_stop"), (0, "no_hard_stop_hit")]:
        g = d[d[label_col] == val]
        if g.empty:
            continue
        rows.append(
            {
                "group": name,
                "n": int(len(g)),
                "mfe_med": float(pd.to_numeric(g[mfe_col], errors="coerce").median()),
                "mae_med": float(pd.to_numeric(g[mae_col], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def compute_sigma_px_hour(df: pd.DataFrame) -> pd.Series:
    out = {}
    for d, day in df.groupby("date"):
        day = day.sort_index()
        sigma_per_min, total_min = gap_aware_sigma_log_per_minute(day, max_gap_minutes=DEFAULT_MAX_GAP_MINUTES)
        if not (np.isfinite(sigma_per_min) and sigma_per_min > 0 and np.isfinite(total_min) and total_min > 0):
            continue
        ref = choose_sigma_ref_price(day, prefer_vwap=True)
        sigma_px = sigma_px_over_horizon(ref, sigma_per_min, total_min)
        if np.isfinite(sigma_px) and sigma_px > 1e-12:
            out[pd.Timestamp(d)] = float(sigma_px)
    return pd.Series(out).sort_index()


def build_diagnostics_entries(df: pd.DataFrame, sig: pd.Series, vol_thr: float) -> pd.DataFrame:
    rows = []
    for d, day in df.groupby("date"):
        day = day.sort_index()
        date_ts = pd.Timestamp(d)

        if date_ts not in sig.index:
            continue
        sigma_px_hour = float(sig.loc[date_ts])
        if not (np.isfinite(sigma_px_hour) and sigma_px_hour >= vol_thr):
            continue

        or_high, or_low = compute_or_levels(day)
        if or_high is None or or_low is None:
            continue

        side, t_signal = pick_first_signal(day, float(or_high), float(or_low))
        if side is None or t_signal is None:
            continue

        t_entry, entry_px = next_bar_open_or_close(day, pd.Timestamp(t_signal))
        if t_entry is None or entry_px is None or not np.isfinite(entry_px):
            continue

        t_eod = end_of_day_time(day)
        if t_eod is None or t_eod < t_entry:
            continue

        eps_fail_px = float(EPS_FAIL_SIG) * float(sigma_px_hour)
        t_sf = find_structure_failure_time(day, t_entry, str(side), float(or_high), float(or_low), eps_fail_px)
        would_sf = 1 if t_sf is not None else 0

        mfe_sig, mae_sig = mfe_mae_to_eod_sig(day, t_entry, t_eod, str(side), float(entry_px), float(sigma_px_hour))

        race_cols = {}
        latent_cols = {}

        for k in TP_K_LIST:
            t_tp, t_sl = first_hit_time_barriers(
                day=day,
                t_start=t_entry,
                t_end=t_eod,
                side=str(side),
                entry_px=float(entry_px),
                sigma_px=float(sigma_px_hour),
                k_tp=float(k),
                s_sl=float(SL_S),
            )

            if t_tp is None and t_sl is None:
                order = "no_hit"
            elif t_tp is None:
                order = "sl_first"
            elif t_sl is None:
                order = "tp_first"
            else:
                order = "tp_first" if pd.Timestamp(t_tp) <= pd.Timestamp(t_sl) else "sl_first"

            k_tag = str(k).replace(".", "_")
            race_cols[f"race_k{k_tag}"] = order
            race_cols[f"time_to_tp_k{k_tag}_min"] = mins_between(t_entry, t_tp)
            race_cols[f"time_to_sl_k{k_tag}_min"] = mins_between(t_entry, t_sl)

            if would_sf and t_sf is not None and t_sf <= t_eod:
                t_tp_after, _ = first_hit_time_barriers(
                    day=day,
                    t_start=pd.Timestamp(t_sf),
                    t_end=t_eod,
                    side=str(side),
                    entry_px=float(entry_px),
                    sigma_px=float(sigma_px_hour),
                    k_tp=float(k),
                    s_sl=1e9,
                )
                latent_cols[f"latent_tp_k{k_tag}"] = 1 if (t_tp_after is not None) else 0
            else:
                latent_cols[f"latent_tp_k{k_tag}"] = np.nan

        rows.append(
            {
                "date": date_ts,
                "side": str(side),
                "signal_time": pd.Timestamp(t_signal),
                "entry_time": pd.Timestamp(t_entry),
                "entry_px": float(entry_px),
                "eod_time": pd.Timestamp(t_eod),
                "sigma_px_hour": float(sigma_px_hour),
                "vol_threshold": float(vol_thr),
                "or_high": float(or_high),
                "or_low": float(or_low),
                "would_structure_fail": int(would_sf),
                "sf_time": pd.Timestamp(t_sf) if t_sf is not None else pd.NaT,
                "mfe_sig_to_eod": float(mfe_sig),
                "mae_sig_to_eod": float(mae_sig),
                **race_cols,
                **latent_cols,
            }
        )

    diag = pd.DataFrame(rows).sort_values(["date", "entry_time"]).reset_index(drop=True)
    if not diag.empty:
        diag["regime"] = label_crisis_regime(diag["date"], mode="binary", binary_labels=("crisis", "non_crisis"))
    return diag


def plot_barrier_race_rates(race_tbl: pd.DataFrame) -> None:
    if race_tbl.empty:
        return
    x = race_tbl["k_tp"].astype(float).to_numpy()
    plt.figure()
    plt.plot(x, race_tbl["tp_first_rate"].astype(float).to_numpy(), marker="o", label="TP first")
    plt.plot(x, race_tbl["sl_first_rate"].astype(float).to_numpy(), marker="o", label="SL first")
    plt.plot(x, race_tbl["no_hit_rate"].astype(float).to_numpy(), marker="o", label="No hit")
    plt.legend()
    plt.ylim(0, 1)
    style_axes(title="Barrier Race Rates by TP Barrier", xlabel="TP barrier k (σ units)", ylabel="Rate", tight=True)
    plt.show()


def plot_latent_winner_rates(latent_tbl: pd.DataFrame) -> None:
    if latent_tbl.empty:
        return
    x = latent_tbl["k_tp"].astype(float).to_numpy()
    y = latent_tbl["latent_tp_rate"].astype(float).to_numpy()
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.ylim(0, 1)
    style_axes(title="Latent Winner Rate among Would-Structure-Fail Trades", xlabel="TP barrier k (σ units)", ylabel="Rate", tight=True)
    plt.show()


def run_stop_sweep_path_classifier(diag_df: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if diag_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    sweep = list(dict.fromkeys([float(x) for x in STOP_SWEEP_LIST]))
    sweep = sorted(sweep)

    classifier_rows = []
    summary_rows = []

    diag_by_date = {pd.Timestamp(d): g.sort_index() for d, g in df.groupby("date")}

    for s in sweep:
        d = diag_df.copy()
        labels = []
        for _, row in d.iterrows():
            date_ts = pd.Timestamp(row["date"])
            day = diag_by_date.get(date_ts)
            if day is None:
                labels.append(None)
                continue
            hit = would_hit_hard_stop_after_delay(
                day=day,
                t_entry=pd.Timestamp(row["entry_time"]),
                t_eod=pd.Timestamp(row["eod_time"]),
                side=str(row["side"]),
                entry_px=float(row["entry_px"]),
                sigma_px_hour=float(row["sigma_px_hour"]),
                stop_sigma=float(s),
                delay_min=int(HARD_STOP_DELAY_MIN),
            )
            labels.append(hit)

        d["would_hit_hard_stop_after_delay"] = labels

        clf_tbl = hard_stop_path_classifier_table(
            entries_df=d,
            label_col="would_hit_hard_stop_after_delay",
            mfe_col="mfe_sig_to_eod",
            mae_col="mae_sig_to_eod",
        )

        if not clf_tbl.empty:
            clf_tbl2 = clf_tbl.copy()
            clf_tbl2.insert(0, "stop_sigma", float(s))
            classifier_rows.append(clf_tbl2)

        def get_val(group: str, col: str) -> float:
            if clf_tbl.empty:
                return np.nan
            sub = clf_tbl[clf_tbl["group"] == group]
            return np.nan if sub.empty else float(sub[col].iloc[0])

        summary_rows.append(
            {
                "stop_sigma": float(s),
                "hs_hit_n": get_val("would_hit_hard_stop", "n"),
                "hs_nohit_n": get_val("no_hard_stop_hit", "n"),
                "hs_hit_mfe_med_sig": get_val("would_hit_hard_stop", "mfe_med"),
                "hs_hit_mae_med_sig": get_val("would_hit_hard_stop", "mae_med"),
                "hs_nohit_mfe_med_sig": get_val("no_hard_stop_hit", "mfe_med"),
                "hs_nohit_mae_med_sig": get_val("no_hard_stop_hit", "mae_med"),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("stop_sigma").reset_index(drop=True)
    clf_all = pd.concat(classifier_rows, ignore_index=True) if classifier_rows else pd.DataFrame()
    return summary, clf_all


def main() -> None:
    df = load_session_df(
        min_perc_first_hour=0.20,
        min_perc_total=0.20,
        require_cols=["open", "high", "low", "close"],
        dropna_cols=None,
        allow_negative_volume=False,
    )
    df.columns = [c.strip().lower() for c in df.columns]
    df = between_session(df, SESSION_START, SESSION_END, inclusive="both").copy()
    df = add_date_and_weekday_cols(df, date_col="date")

    sig = compute_sigma_px_hour(df)
    vol_thr = float(sig.quantile(VOL_SPLIT_QUANTILE))

    diag_df = build_diagnostics_entries(df, sig, vol_thr)


    race_tbl = pd.concat([barrier_race_summary(diag_df, float(k)) for k in TP_K_LIST], ignore_index=True)
    print("\nbarrier race summary")
    print(race_tbl.to_string(index=False))
    plot_barrier_race_rates(race_tbl)

    latent_tbl = pd.concat([latent_winner_summary(diag_df, float(k)) for k in TP_K_LIST], ignore_index=True)
    print("\nlatent winner summary")
    print(latent_tbl.to_string(index=False))
    plot_latent_winner_rates(latent_tbl)

    sf_tbl = sf_vs_nosf_path_summary(diag_df)
    print("\nsf vs nosf path summary")
    print(sf_tbl.to_string(index=False))

    stop_sweep_summary, stop_sweep_classifier = run_stop_sweep_path_classifier(diag_df, df)

    print("\nstop sweep summary")
    print(stop_sweep_summary.to_string(index=False))

    print("\nstop sweep hardstop path classifier")
    print(stop_sweep_classifier.to_string(index=False))


if __name__ == "__main__":
    main()
