from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.constants import (
    DEFAULT_MAX_GAP_MINUTES,
    SESSION_END,
    SESSION_START,
    MIN_PERC_FIRST_HOUR,
    MIN_PERC_TOTAL,
)
from utils.data import load_filtered_df
from utils.events import first_true_cross_time
from utils.regimes import label_crisis_regime
from utils.time_features import minute_of_session
from utils.volatility import (
    choose_sigma_ref_price,
    gap_aware_sigma_log_per_minute,
    sigma_px_over_horizon,
)

# =========================
# CONFIG
# =========================
VOL_SPLIT_QUANTILE = 0.50
EPS_FAIL_SIG = 0.1

STOP_SIGMA = 1.25
HARD_STOP_DELAY_MIN = 0

TAKE_PROFIT_SIGMA = 2.16  # for reversal_fade_tp only

USE_NEXT_BAR_OPEN_FOR_ENTRY = True
SLIPPAGE_BPS = 0.0
COMMISSION_PER_UNIT = 0.0
ONE_TRADE_PER_DAY = True

# Choose which strategy to backtest:
#   "breakout"                   = original OR breakout strategy
#   "reversal_fade"              = trade the *same* breakout; if its hard stop is hit (touch), enter reversal
#                                 immediately at the stop price; exits: structure_fail / hard_stop / eod
#   "reversal_fade_no_sf"        = same, but NO early structure-fail stop (only hard_stop / eod)
#   "reversal_fade_tp"           = same entry as reversal_fade, take profit at TAKE_PROFIT_SIGMA*sigma (eod fallback)
#   "stop_sweep_asym"            = conditioned on breakout; wait for 1.25σ stop sweep, enter in stop direction at stop;
#                                 exits: hard_stop / eod
#   "breakout_stop_reversal"     = (not implemented here; keep your prior block if you use it)
STRATEGY = "stop_sweep_asym"  # "breakout" | "reversal_fade" | "reversal_fade_no_sf" | "reversal_fade_tp" | "stop_sweep_asym"


def _apply_slippage(price: float, side: str, bps: float) -> float:
    if not np.isfinite(price) or float(bps) <= 0:
        return float(price)
    mult = 1.0 + (float(bps) / 1e4) if side == "buy" else 1.0 - (float(bps) / 1e4)
    return float(price * mult)


def _compute_or_levels(day: pd.DataFrame) -> tuple[float | None, float | None]:
    fh = day.between_time("07:00", "07:59", inclusive="both").dropna(subset=["high", "low"])
    if fh.empty:
        return None, None
    or_high = float(pd.to_numeric(fh["high"], errors="coerce").max())
    or_low = float(pd.to_numeric(fh["low"], errors="coerce").min())
    if not (np.isfinite(or_high) and np.isfinite(or_low) and or_high > or_low):
        return None, None
    return or_high, or_low


def _pick_first_breakout(day: pd.DataFrame, or_high: float, or_low: float) -> tuple[str | None, pd.Timestamp | None]:
    t_long = first_true_cross_time(day, float(or_high), "up")
    t_short = first_true_cross_time(day, float(or_low), "down")
    if t_long is None and t_short is None:
        return None, None
    if t_long is None:
        return "short", pd.Timestamp(t_short)
    if t_short is None:
        return "long", pd.Timestamp(t_long)
    return ("long", pd.Timestamp(t_long)) if pd.Timestamp(t_long) <= pd.Timestamp(t_short) else ("short", pd.Timestamp(t_short))


def _next_bar_open_or_close(day: pd.DataFrame, t_signal: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    day = day.sort_index()
    if t_signal not in day.index:
        try:
            pos = day.index.get_indexer([t_signal], method="pad")[0]
            if pos < 0:
                return None, None
            t_signal = day.index[pos]
        except Exception:
            return None, None

    if USE_NEXT_BAR_OPEN_FOR_ENTRY:
        pos = day.index.get_loc(t_signal)
        if isinstance(pos, slice):
            pos = pos.start
        nxt = int(pos) + 1
        if 0 <= nxt < len(day.index):
            t_entry = day.index[nxt]
            px = day.at[t_entry, "open"] if "open" in day.columns else np.nan
            if np.isfinite(px):
                return pd.Timestamp(t_entry), float(px)

    px = day.at[t_signal, "close"]
    return (pd.Timestamp(t_signal), float(px)) if np.isfinite(px) else (None, None)


def _end_of_day_exit(day: pd.DataFrame) -> tuple[pd.Timestamp | None, float | None]:
    d = day.dropna(subset=["close"]).sort_index()
    if d.empty:
        return None, None
    return pd.Timestamp(d.index[-1]), float(pd.to_numeric(d["close"], errors="coerce").iloc[-1])


def _choose_earliest_exit(
        candidates: list[tuple[pd.Timestamp | None, float | None, str]]
) -> tuple[pd.Timestamp | None, float | None, str | None]:
    c = [(pd.Timestamp(t), float(px), r) for (t, px, r) in candidates if t is not None and px is not None and np.isfinite(px)]
    if not c:
        return None, None, None
    c.sort(key=lambda x: x[0])
    return c[0]


def _find_stop_exit(
        day: pd.DataFrame,
        t_entry: pd.Timestamp,
        side: str,
        stop_price: float,
        delay_minutes: int,
) -> tuple[pd.Timestamp | None, float | None]:
    t0 = pd.Timestamp(t_entry) + pd.Timedelta(minutes=int(delay_minutes)) if int(delay_minutes) > 0 else pd.Timestamp(t_entry)
    fwd = day.loc[t0:].dropna(subset=["high", "low"]).sort_index()
    if fwd.empty:
        return None, None

    hi = pd.to_numeric(fwd["high"], errors="coerce").astype(float)
    lo = pd.to_numeric(fwd["low"], errors="coerce").astype(float)

    if side == "long":
        trig = lo <= float(stop_price)
    else:
        trig = hi >= float(stop_price)

    idx = fwd.index[trig.fillna(False)]
    if len(idx) == 0:
        return None, None

    return pd.Timestamp(idx[0]), float(stop_price)


def _find_take_profit_exit(
        day: pd.DataFrame,
        t_entry: pd.Timestamp,
        side: str,
        tp_price: float,
) -> tuple[pd.Timestamp | None, float | None]:
    fwd = day.loc[pd.Timestamp(t_entry):].dropna(subset=["high", "low"]).sort_index()
    if fwd.empty:
        return None, None

    hi = pd.to_numeric(fwd["high"], errors="coerce").astype(float)
    lo = pd.to_numeric(fwd["low"], errors="coerce").astype(float)

    if side == "long":
        trig = hi >= float(tp_price)
    else:
        trig = lo <= float(tp_price)

    idx = fwd.index[trig.fillna(False)]
    if len(idx) == 0:
        return None, None

    return pd.Timestamp(idx[0]), float(tp_price)


def _find_structure_failure_exit_reversal_touch(
        day: pd.DataFrame,
        t_entry: pd.Timestamp,
        side: str,
        breakout_boundary: float,
        eps_fail_px: float,
        window_minutes: int = 30,
) -> tuple[pd.Timestamp | None, float | None]:
    w_end = pd.Timestamp(t_entry) + pd.Timedelta(minutes=int(window_minutes))
    w = day.loc[pd.Timestamp(t_entry):w_end].dropna(subset=["high", "low"]).sort_index()
    if w.empty:
        return None, None

    hi = pd.to_numeric(w["high"], errors="coerce").astype(float)
    lo = pd.to_numeric(w["low"], errors="coerce").astype(float)

    if side == "short":
        level = float(breakout_boundary) + float(eps_fail_px)
        trig = hi >= level
        px = level
    else:
        level = float(breakout_boundary) - float(eps_fail_px)
        trig = lo <= level
        px = level

    idx = w.index[trig.fillna(False)]
    if len(idx) == 0:
        return None, None

    return pd.Timestamp(idx[0]), float(px)


def _sharpe_daily_annualized(daily_pnl: pd.Series) -> float:
    x = pd.to_numeric(daily_pnl, errors="coerce").dropna().astype(float)
    if len(x) < 2:
        return np.nan
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if not (np.isfinite(sd) and sd > 0):
        return np.nan
    return float((mu / sd) * np.sqrt(252.0))


def _safe_quantile(s: pd.Series, q: float) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    return float(x.quantile(float(q))) if len(x) else np.nan


def _safe_median(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    return float(x.median()) if len(x) else np.nan


def _safe_max(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    return float(x.max()) if len(x) else np.nan


def compute_sigma_by_date(df: pd.DataFrame) -> pd.Series:
    out: dict[pd.Timestamp, float] = {}
    for d, day in df.groupby("date", sort=True):
        day = day.sort_index()
        sigma_per_min, total_min = gap_aware_sigma_log_per_minute(day, max_gap_minutes=DEFAULT_MAX_GAP_MINUTES)
        if not (np.isfinite(sigma_per_min) and float(sigma_per_min) > 0 and float(total_min) > 0):
            continue
        ref_price = choose_sigma_ref_price(day, prefer_vwap=True)
        sigma_px = sigma_px_over_horizon(float(ref_price), float(sigma_per_min), float(total_min))
        if np.isfinite(sigma_px) and float(sigma_px) > 1e-12:
            out[pd.Timestamp(d).normalize()] = float(sigma_px)
    return pd.Series(out).sort_index()


def _leg_mfe_mae_sig(
        day: pd.DataFrame,
        t_entry: pd.Timestamp,
        t_exit: pd.Timestamp,
        side: str,
        entry_px_eff: float,
        sigma_px_hour: float,
) -> tuple[float, float]:
    w = day.loc[pd.Timestamp(t_entry): pd.Timestamp(t_exit)].dropna(subset=["high", "low", "close"]).sort_index()
    if w.empty:
        return np.nan, np.nan

    high = pd.to_numeric(w["high"], errors="coerce").astype(float)
    low = pd.to_numeric(w["low"], errors="coerce").astype(float)

    if side == "long":
        mfe_px = float((high - float(entry_px_eff)).max())
        mae_px = float((float(entry_px_eff) - low).max())
    else:
        mfe_px = float((float(entry_px_eff) - low).max())
        mae_px = float((high - float(entry_px_eff)).max())

    mfe_sig = float(mfe_px) / float(sigma_px_hour) if np.isfinite(mfe_px) else np.nan
    mae_sig = float(mae_px) / float(sigma_px_hour) if np.isfinite(mae_px) else np.nan
    return mfe_sig, mae_sig


def run_backtest(
        df: pd.DataFrame,
        sig: pd.Series,
        vol_thr: float,
        *,
        stop_sigma: float,
        strategy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades: list[dict] = []
    strategy = str(strategy).strip().lower()

    if strategy not in {"breakout", "reversal_fade", "reversal_fade_no_sf", "reversal_fade_tp", "stop_sweep_asym"}:
        raise ValueError(
            f"Unknown strategy='{strategy}'. Use 'breakout', 'reversal_fade', 'reversal_fade_no_sf', 'reversal_fade_tp', or 'stop_sweep_asym'."
        )

    for d, day in df.groupby("date", sort=True):
        day = day.sort_index()
        date_ts = pd.Timestamp(d).normalize()
        if date_ts not in sig.index:
            continue

        sigma_px_hour = float(sig.loc[date_ts])
        if not (np.isfinite(sigma_px_hour) and sigma_px_hour >= float(vol_thr)):
            continue

        or_high, or_low = _compute_or_levels(day)
        if or_high is None or or_low is None:
            continue

        breakout_dir, t_break = _pick_first_breakout(day, float(or_high), float(or_low))
        if breakout_dir is None or t_break is None:
            continue

        eps_fail_px = float(EPS_FAIL_SIG) * float(sigma_px_hour)
        stop_dist = float(stop_sigma) * float(sigma_px_hour)
        tp_dist = float(TAKE_PROFIT_SIGMA) * float(sigma_px_hour)

        if strategy == "breakout":
            side = str(breakout_dir)
            t_entry, entry_px = _next_bar_open_or_close(day, pd.Timestamp(t_break))
            if t_entry is None or entry_px is None or not np.isfinite(entry_px):
                continue

            entry_side = "buy" if side == "long" else "sell"
            entry_px_eff = _apply_slippage(float(entry_px), entry_side, SLIPPAGE_BPS)

            stop_price = (float(entry_px_eff) - float(stop_dist)) if side == "long" else (float(entry_px_eff) + float(stop_dist))

            exits: list[tuple[pd.Timestamp | None, float | None, str]] = []

            w_end = pd.Timestamp(t_entry) + pd.Timedelta(minutes=30)
            w = day.loc[pd.Timestamp(t_entry):w_end].dropna(subset=["close"]).sort_index()
            if not w.empty:
                cl = pd.to_numeric(w["close"], errors="coerce").astype(float)
                if side == "long":
                    trig = cl <= float(or_high) - float(eps_fail_px)
                else:
                    trig = cl >= float(or_low) + float(eps_fail_px)
                idx = w.index[trig.fillna(False)]
                if len(idx) > 0:
                    t_sf = pd.Timestamp(idx[0])
                    exits.append((t_sf, float(w.at[t_sf, "close"]), "structure_fail_30m"))

            t_st, px_st = _find_stop_exit(day, pd.Timestamp(t_entry), side, float(stop_price), int(HARD_STOP_DELAY_MIN))
            if t_st is not None and px_st is not None:
                exits.append((t_st, px_st, "hard_stop"))

            t_eod, px_eod = _end_of_day_exit(day)
            if t_eod is not None and px_eod is not None:
                exits.append((t_eod, px_eod, "eod"))

            t_exit, exit_px, reason = _choose_earliest_exit(exits)
            if t_exit is None or exit_px is None or reason is None:
                continue

            exit_side = "sell" if side == "long" else "buy"
            exit_px_eff = _apply_slippage(float(exit_px), exit_side, SLIPPAGE_BPS)

            pnl_px = (float(exit_px_eff) - float(entry_px_eff)) if side == "long" else (float(entry_px_eff) - float(exit_px_eff))
            pnl_px_net = float(pnl_px) - 2.0 * float(COMMISSION_PER_UNIT)
            pnl_sig = float(pnl_px_net) / float(sigma_px_hour)

            hold_min = float((pd.Timestamp(t_exit) - pd.Timestamp(t_entry)).total_seconds() / 60.0)
            mfe_sig, mae_sig = _leg_mfe_mae_sig(day, pd.Timestamp(t_entry), pd.Timestamp(t_exit), side, float(entry_px_eff), float(sigma_px_hour))

            trades.append(
                dict(
                    date=date_ts,
                    strategy=strategy,
                    leg=1,
                    breakout_dir=str(breakout_dir),
                    side=side,
                    entry_time=pd.Timestamp(t_entry),
                    exit_time=pd.Timestamp(t_exit),
                    exit_reason=str(reason),
                    sigma_px_hour=float(sigma_px_hour),
                    entry_px_eff=float(entry_px_eff),
                    exit_px_eff=float(exit_px_eff),
                    pnl_sig=float(pnl_sig),
                    hold_min=float(hold_min),
                    mfe_sig=float(mfe_sig) if np.isfinite(mfe_sig) else np.nan,
                    mae_sig=float(mae_sig) if np.isfinite(mae_sig) else np.nan,
                )
            )
            if ONE_TRADE_PER_DAY:
                continue

        else:
            breakout_side = str(breakout_dir)
            t_b_entry, b_entry_px = _next_bar_open_or_close(day, pd.Timestamp(t_break))
            if t_b_entry is None or b_entry_px is None or not np.isfinite(b_entry_px):
                continue

            b_entry_side = "buy" if breakout_side == "long" else "sell"
            b_entry_px_eff = _apply_slippage(float(b_entry_px), b_entry_side, SLIPPAGE_BPS)

            b_stop_price = (float(b_entry_px_eff) - float(stop_dist)) if breakout_side == "long" else (float(b_entry_px_eff) + float(stop_dist))

            t_stop_hit, stop_hit_px = _find_stop_exit(day, pd.Timestamp(t_b_entry), breakout_side, float(b_stop_price), int(HARD_STOP_DELAY_MIN))
            if t_stop_hit is None or stop_hit_px is None or not np.isfinite(stop_hit_px):
                continue

            side = "short" if breakout_side == "long" else "long"
            t_entry = pd.Timestamp(t_stop_hit)
            entry_px = float(stop_hit_px)

            entry_side = "buy" if side == "long" else "sell"
            entry_px_eff = _apply_slippage(float(entry_px), entry_side, SLIPPAGE_BPS)

            stop_price = (float(entry_px_eff) - float(stop_dist)) if side == "long" else (float(entry_px_eff) + float(stop_dist))
            tp_price = (float(entry_px_eff) + float(tp_dist)) if side == "long" else (float(entry_px_eff) - float(tp_dist))

            breakout_boundary = float(or_high) if str(breakout_dir) == "long" else float(or_low)

            exits: list[tuple[pd.Timestamp | None, float | None, str]] = []

            if strategy == "reversal_fade" or strategy == "reversal_fade_tp":
                t_sf, px_sf = _find_structure_failure_exit_reversal_touch(
                    day=day,
                    t_entry=pd.Timestamp(t_entry),
                    side=str(side),
                    breakout_boundary=float(breakout_boundary),
                    eps_fail_px=float(eps_fail_px),
                    window_minutes=30,
                )
                if t_sf is not None and px_sf is not None:
                    exits.append((t_sf, px_sf, "structure_fail_30m"))

            t_st, px_st = _find_stop_exit(day, pd.Timestamp(t_entry), str(side), float(stop_price), int(HARD_STOP_DELAY_MIN))
            if t_st is not None and px_st is not None:
                exits.append((t_st, px_st, "hard_stop"))

            if strategy == "reversal_fade_tp":
                t_tp, px_tp = _find_take_profit_exit(day, pd.Timestamp(t_entry), str(side), float(tp_price))
                if t_tp is not None and px_tp is not None:
                    exits.append((t_tp, px_tp, "take_profit_2.16sigma"))
                t_eod, px_eod = _end_of_day_exit(day)
                if t_eod is not None and px_eod is not None:
                    exits.append((t_eod, px_eod, "eod_fallback"))
            else:
                t_eod, px_eod = _end_of_day_exit(day)
                if t_eod is not None and px_eod is not None:
                    exits.append((t_eod, px_eod, "eod"))

            t_exit, exit_px, reason = _choose_earliest_exit(exits)
            if t_exit is None or exit_px is None or reason is None:
                continue

            exit_side = "sell" if side == "long" else "buy"
            exit_px_eff = _apply_slippage(float(exit_px), exit_side, SLIPPAGE_BPS)

            pnl_px = (float(exit_px_eff) - float(entry_px_eff)) if side == "long" else (float(entry_px_eff) - float(exit_px_eff))
            pnl_px_net = float(pnl_px) - 2.0 * float(COMMISSION_PER_UNIT)
            pnl_sig = float(pnl_px_net) / float(sigma_px_hour)

            hold_min = float((pd.Timestamp(t_exit) - pd.Timestamp(t_entry)).total_seconds() / 60.0)
            mfe_sig, mae_sig = _leg_mfe_mae_sig(day, pd.Timestamp(t_entry), pd.Timestamp(t_exit), str(side), float(entry_px_eff), float(sigma_px_hour))

            trades.append(
                dict(
                    date=date_ts,
                    strategy=strategy,
                    leg=1,
                    breakout_dir=str(breakout_dir),
                    side=str(side),
                    entry_time=pd.Timestamp(t_entry),
                    exit_time=pd.Timestamp(t_exit),
                    exit_reason=str(reason),
                    sigma_px_hour=float(sigma_px_hour),
                    entry_px_eff=float(entry_px_eff),
                    exit_px_eff=float(exit_px_eff),
                    pnl_sig=float(pnl_sig),
                    hold_min=float(hold_min),
                    mfe_sig=float(mfe_sig) if np.isfinite(mfe_sig) else np.nan,
                    mae_sig=float(mae_sig) if np.isfinite(mae_sig) else np.nan,
                    breakout_entry_time=pd.Timestamp(t_b_entry),
                    breakout_entry_px_eff=float(b_entry_px_eff),
                    breakout_stop_price=float(b_stop_price),
                    breakout_stop_hit_time=pd.Timestamp(t_stop_hit),
                )
            )

            if ONE_TRADE_PER_DAY:
                continue

    trades_df = pd.DataFrame(trades).sort_values(["date", "entry_time", "leg"]).reset_index(drop=True)
    if trades_df.empty:
        return trades_df, pd.DataFrame()

    daily = (
        trades_df.groupby("date", sort=True)
        .agg(trades=("pnl_sig", "size"), pnl_sig=("pnl_sig", "sum"))
        .sort_index()
    )
    daily["equity_sig"] = daily["pnl_sig"].cumsum()
    daily["cummax_sig"] = daily["equity_sig"].cummax()
    daily["dd_sig"] = daily["equity_sig"] - daily["cummax_sig"]
    return trades_df, daily


def build_metrics_table(trades_df: pd.DataFrame, daily: pd.DataFrame, sig: pd.Series) -> pd.DataFrame:
    if trades_df.empty or daily.empty:
        return pd.DataFrame({"Metric": ["Sample (sigma days)"], "Value": [int(len(sig))]})

    regimes = label_crisis_regime(
        trades_df["date"],
        mode="binary",
        binary_labels=("crisis", "non_crisis"),
        ordered=False,
    )
    trades_df = trades_df.copy()
    trades_df["regime"] = regimes.to_numpy()

    total_pnl_sig = float(daily["pnl_sig"].sum())
    sharpe = _sharpe_daily_annualized(daily["pnl_sig"])
    max_dd = float(daily["dd_sig"].min())
    win_rate = float((pd.to_numeric(trades_df["pnl_sig"], errors="coerce") > 0).mean())

    exit_counts = trades_df["exit_reason"].value_counts()
    n_sf = int(exit_counts.get("structure_fail_30m", 0))
    n_hs = int(exit_counts.get("hard_stop", 0))
    n_eod = int(exit_counts.get("eod", 0)) + int(exit_counts.get("eod_fallback", 0))

    pnl_by_side = trades_df.groupby("side")["pnl_sig"].sum()
    pnl_long = float(pnl_by_side.get("long", 0.0))
    pnl_short = float(pnl_by_side.get("short", 0.0))

    pnl_by_regime = trades_df.groupby("regime")["pnl_sig"].sum()
    pnl_crisis = float(pnl_by_regime.get("crisis", 0.0))
    pnl_non = float(pnl_by_regime.get("non_crisis", 0.0))

    trade_med = float(pd.to_numeric(trades_df["pnl_sig"], errors="coerce").dropna().median()) if len(trades_df) else np.nan
    trade_p90 = _safe_quantile(trades_df["pnl_sig"], 0.90)
    trade_max = _safe_max(trades_df["pnl_sig"])

    hold_med = _safe_median(trades_df["hold_min"])
    hold_p75 = _safe_quantile(trades_df["hold_min"], 0.75)
    hold_max = _safe_max(trades_df["hold_min"])

    entry_min = minute_of_session(pd.DatetimeIndex(trades_df["entry_time"]), SESSION_START).astype(float)
    entry_med = float(entry_min.median()) if len(entry_min) else np.nan
    entry_p75 = float(entry_min.quantile(0.75)) if len(entry_min) else np.nan
    entry_max = float(entry_min.max()) if len(entry_min) else np.nan

    rows = [
        ("Strategy", str(trades_df["strategy"].iloc[0]) if "strategy" in trades_df.columns and len(trades_df) else ""),
        ("Sample (high vol days)", int(len(sig))),
        ("Trading days / trades", f"{int(len(daily))} / {int(len(trades_df))}"),
        ("Total PnL (cumulative, sigma)", float(total_pnl_sig)),
        ("Sharpe (daily sigma, annualized)", float(sharpe)),
        ("Max drawdown (sigma)", float(max_dd)),
        ("Win rate (trades)", float(win_rate)),
        ("Exit reason: structure fail (30m)", int(n_sf)),
        ("Exit reason: hard stop", int(n_hs)),
        ("Exit reason: eod", int(n_eod)),
        ("PnL (sigma): long / short", f"{pnl_long:.3f} / {pnl_short:.3f}"),
        ("PnL (sigma): crisis / non-crisis", f"{pnl_crisis:.3f} / {pnl_non:.3f}"),
        ("Trade PnL (sigma): med / p90 / max", f"{trade_med:.3f} / {trade_p90:.3f} / {trade_max:.3f}"),
        ("Hold duration (min): med / p75 / max", f"{hold_med:.1f} / {hold_p75:.1f} / {hold_max:.0f}"),
        ("Entry time (min): med / p75 / max", f"{entry_med:.1f} / {entry_p75:.2f} / {entry_max:.0f}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def plot_equity_curve(daily: pd.DataFrame, title_suffix: str = "") -> None:
    plt.figure()
    plt.plot(daily.index, pd.to_numeric(daily["equity_sig"], errors="coerce").astype(float))
    plt.title(f"Equity curve (PnL in σ units){title_suffix}")
    plt.xlabel("Date")
    plt.ylabel("Equity (σ units)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = load_filtered_df(
        min_perc_first_hour=MIN_PERC_FIRST_HOUR,
        min_perc_total=MIN_PERC_TOTAL,
    )
    df["date"] = pd.to_datetime(df.index.date)

    sig = compute_sigma_by_date(df)
    vol_thr = float(sig.quantile(VOL_SPLIT_QUANTILE))

    trades_df, daily = run_backtest(
        df=df,
        sig=sig,
        vol_thr=vol_thr,
        stop_sigma=float(STOP_SIGMA),
        strategy=str(STRATEGY),
    )

    metrics_tbl = build_metrics_table(trades_df=trades_df, daily=daily, sig=sig)
    print(metrics_tbl.to_string(index=False))

    if not daily.empty:
        plot_equity_curve(daily, title_suffix=f" — {STRATEGY}")


if __name__ == "__main__":
    main()
