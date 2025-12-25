import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.constants import SESSION_START, SESSION_END
from utils.data import load_filtered_df
from utils.summary import summarize_expansion
from utils.time_features import add_date_and_weekday_cols, between_session, expected_bars_in_session, minute_of_session
from utils.regimes import regime_for_date

COVERAGE_LOADER = 0.01

WINDOW_LENGTHS = [30, 35, 40, 45, 60, 90]

MIN_OR_WINDOW_COVERAGE = 0.20
MIN_SESSION_COVERAGE = 0.20

REGIMES = {
    "Pre-Crisis": ("2019-06-01", "2020-12-31"),
    "Crisis": ("2021-01-01", "2023-07-31"),
    "Post-Crisis": ("2023-08-01", "2024-12-31"),
}

DAY_MAP = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri"]
DAY_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

REGIME_COLORS = {
    "Pre-Crisis": "tab:orange",
    "Crisis": "tab:red",
    "Post-Crisis": "tab:purple",
}

def _load_base_df() -> pd.DataFrame:
    df = load_filtered_df(min_perc_first_hour=COVERAGE_LOADER, min_perc_total=COVERAGE_LOADER).copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].copy()
    df.index = df.index.tz_localize(None)
    df = df.sort_index()
    df = between_session(df, SESSION_START, SESSION_END, inclusive="both").copy()
    return df


def _add_day_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = add_date_and_weekday_cols(df, date_col="Date", weekday_col="Day_Num", weekday_name_col="_wd_name")
    out["Segment"] = out["Day_Num"].map(DAY_MAP)
    out = out[out["Segment"].isin(DAY_ORDER)].copy()
    out["Mins_From_Open"] = minute_of_session(out.index, SESSION_START).astype(int)
    out = out[out["Mins_From_Open"] >= 0].copy()
    return out


def _daily_session_table(df: pd.DataFrame, expected_session_bars: int) -> pd.DataFrame:
    daily = df.groupby("Date").agg(
        day_high=("high", "max"),
        day_low=("low", "min"),
        n_bars=("close", lambda s: int(s.notna().sum())),
        Segment=("Segment", "first"),
    )
    daily["Total_Range"] = daily["day_high"] - daily["day_low"]
    daily["session_cov"] = daily["n_bars"] / float(expected_session_bars)
    daily = daily[
        (daily["Total_Range"] > 0)
        & daily["Total_Range"].notna()
        & (daily["session_cov"] >= float(MIN_SESSION_COVERAGE))
    ].copy()
    return daily


def _or_stats_by_window(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, pd.Series], dict[int, pd.Series]]:
    valid_by_len = {}
    orwidth_by_len = {}
    orcov_by_len = {}

    for length in WINDOW_LENGTHS:
        in_window = (df["Mins_From_Open"] >= 0) & (df["Mins_From_Open"] < int(length))
        w = df.loc[in_window]

        per_day_or = w.groupby("Date").agg(
            or_high=("high", "max"),
            or_low=("low", "min"),
            or_valid=("close", lambda s: int(s.notna().sum())),
        )
        per_day_or["OR_Width"] = per_day_or["or_high"] - per_day_or["or_low"]

        expected_or_bars = int(np.ceil(int(length) / 1))
        per_day_or["or_cov"] = per_day_or["or_valid"] / float(expected_or_bars)

        valid = (per_day_or["OR_Width"] > 0) & (per_day_or["or_cov"] >= float(MIN_OR_WINDOW_COVERAGE))

        valid_by_len[int(length)] = valid
        orwidth_by_len[int(length)] = per_day_or["OR_Width"]
        orcov_by_len[int(length)] = per_day_or["or_cov"]

    valid_df = pd.DataFrame(valid_by_len).fillna(False)
    return valid_df, orwidth_by_len, orcov_by_len


def _build_day_table(daily: pd.DataFrame, valid_dates: pd.Index, orwidth_by_len: dict[int, pd.Series]) -> pd.DataFrame:
    day_table = daily.loc[daily.index.intersection(valid_dates)].copy()
    day_table["Regime"] = [
        regime_for_date(d, REGIMES, inclusive=True) for d in day_table.index
    ]
    day_table = day_table.dropna(subset=["Regime"]).copy()

    for length in WINDOW_LENGTHS:
        length = int(length)
        day_table[f"OR_Width_{length}"] = orwidth_by_len[length].reindex(day_table.index)
        day_table[f"Exp_{length}"] = day_table["Total_Range"] / day_table[f"OR_Width_{length}"]

    return day_table


def _print_diagnostics(day_table: pd.DataFrame) -> tuple[dict, dict, dict]:
    print(f"Loader coverage: {COVERAGE_LOADER}")
    print(f"Min OR-window coverage: {MIN_OR_WINDOW_COVERAGE}")
    print(f"Min full-session coverage: {MIN_SESSION_COVERAGE}")
    print(f"Valid day count (all windows): {len(day_table)}")

    print("\nDays contributing per regime (pooled weekdays):")
    for r in REGIMES.keys():
        n = int((day_table["Regime"] == r).sum())
        print(f"  {r}: {n}")

    print("\nDays contributing per weekday (pooled regimes):")
    weekday_counts = (
        day_table["Segment"].value_counts().reindex(DAY_ORDER).fillna(0).astype(int).to_dict()
    )
    for d in DAY_ORDER:
        print(f"  {d}: {weekday_counts[d]}")

    print("\nDays contributing per regime * weekday:")
    reg_wd = (
        day_table.groupby(["Regime", "Segment"]).size()
        .reindex(pd.MultiIndex.from_product([list(REGIMES.keys()), DAY_ORDER], names=["Regime", "Segment"]))
        .fillna(0)
        .astype(int)
    )
    for r in REGIMES.keys():
        row = reg_wd.loc[r].to_dict()
        print(f"  {r}: {row}")

    n_days_regime_weekday = {r: reg_wd.loc[r].to_dict() for r in REGIMES.keys()}
    n_days_regime = {r: int((day_table["Regime"] == r).sum()) for r in REGIMES.keys()}
    n_days_weekday = weekday_counts
    return n_days_regime_weekday, n_days_regime, n_days_weekday


def _stats_regime_weekday(day_table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for regime_name in REGIMES.keys():
        rows = []
        for day in DAY_ORDER:
            sub = day_table[(day_table["Regime"] == regime_name) & (day_table["Segment"] == day)]
            for length in WINDOW_LENGTHS:
                s = summarize_expansion(sub[f"Exp_{int(length)}"])
                rows.append({"Regime": regime_name, "Segment": day, "Window_Mins": int(length), **s})
        out[regime_name] = pd.DataFrame(rows)
    return out


def _plot_regime_by_weekday(stats_by_regime: dict[str, pd.DataFrame], n_days_regime_weekday: dict) -> None:
    x = np.arange(len(WINDOW_LENGTHS))
    width = 0.14

    for regime_name in REGIMES.keys():
        stats = stats_by_regime[regime_name]
        fig, ax = plt.subplots(figsize=(12, 7))

        for i, day in enumerate(DAY_ORDER):
            day_data = (
                stats[stats["Segment"] == day]
                .set_index("Window_Mins")
                .reindex([int(w) for w in WINDOW_LENGTHS])
            )

            med = day_data["Median"].to_numpy()
            mean = day_data["Mean"].to_numpy()
            p25 = day_data["P25"].to_numpy()
            p75 = day_data["P75"].to_numpy()
            xpos = x + (i - 2) * width

            ax.bar(
                xpos,
                med,
                width,
                label=f"{day} (N={n_days_regime_weekday[regime_name][day]})",
                color=DAY_COLORS[i],
                alpha=0.35,
            )

            yerr = np.vstack([med - p25, p75 - med])
            ax.errorbar(
                xpos,
                med,
                yerr=yerr,
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=2,
                alpha=0.7,
            )

            ax.scatter(
                xpos,
                mean,
                marker="o",
                s=18,
                color="black",
                alpha=0.85,
                zorder=3,
            )

        ax.set_xlabel("Opening Range Window (Minutes)", fontsize=12)
        ax.set_ylabel("Expansion Ratio", fontsize=12, labelpad=18)
        ax.set_title(
            f"Expansion Ratio by Weekday ({regime_name})\n(Bars=Median, IQR=P25–P75, Dots=Mean)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(w)}m" for w in WINDOW_LENGTHS])
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)

        ax.legend(loc="upper left", ncol=2, fontsize=9)
        ax.set_ylim(0, 5)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.show()


def _plot_regime_only(day_table: pd.DataFrame, n_days_regime: dict) -> None:
    rows = []
    for regime_name in REGIMES.keys():
        sub = day_table[day_table["Regime"] == regime_name]
        for length in WINDOW_LENGTHS:
            s = summarize_expansion(sub[f"Exp_{int(length)}"])
            rows.append({"Regime": regime_name, "Window_Mins": int(length), **s})

    regime_only = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(WINDOW_LENGTHS))
    width = 0.22

    for i, regime_name in enumerate(REGIMES.keys()):
        data = (
            regime_only[regime_only["Regime"] == regime_name]
            .set_index("Window_Mins")
            .reindex([int(w) for w in WINDOW_LENGTHS])
        )

        med = data["Median"].to_numpy()
        mean = data["Mean"].to_numpy()
        p25 = data["P25"].to_numpy()
        p75 = data["P75"].to_numpy()
        xpos = x + (i - 1) * width

        ax.bar(
            xpos,
            med,
            width,
            label=f"{regime_name} (N={n_days_regime[regime_name]})",
            color=REGIME_COLORS[regime_name],
            alpha=0.35,
        )

        yerr = np.vstack([med - p25, p75 - med])
        ax.errorbar(
            xpos,
            med,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=2,
            alpha=0.7,
        )

        ax.scatter(
            xpos,
            mean,
            marker="o",
            s=22,
            color="black",
            alpha=0.85,
            zorder=3,
        )

    ax.set_xlabel("Opening Range Window (Minutes)", fontsize=12)
    ax.set_ylabel("Expansion Ratio", fontsize=12, labelpad=18)
    ax.set_title(
        "Expansion Ratio by Regime\n(Bars=Median, IQR=P25–P75, Dots=Mean)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(w)}m" for w in WINDOW_LENGTHS])
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)

    ax.legend(loc="upper left", ncol=1, fontsize=10)
    ax.set_ylim(0, 5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_weekday_only(day_table: pd.DataFrame, n_days_weekday: dict) -> None:
    rows = []
    for day in DAY_ORDER:
        sub = day_table[day_table["Segment"] == day]
        for length in WINDOW_LENGTHS:
            s = summarize_expansion(sub[f"Exp_{int(length)}"])
            rows.append({"Segment": day, "Window_Mins": int(length), **s})

    weekday_only = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(WINDOW_LENGTHS))
    width = 0.14

    for i, day in enumerate(DAY_ORDER):
        data = (
            weekday_only[weekday_only["Segment"] == day]
            .set_index("Window_Mins")
            .reindex([int(w) for w in WINDOW_LENGTHS])
        )

        med = data["Median"].to_numpy()
        mean = data["Mean"].to_numpy()
        p25 = data["P25"].to_numpy()
        p75 = data["P75"].to_numpy()
        xpos = x + (i - 2) * width

        ax.bar(
            xpos,
            med,
            width,
            label=f"{day} (N={n_days_weekday[day]})",
            color=DAY_COLORS[i],
            alpha=0.35,
        )

        yerr = np.vstack([med - p25, p75 - med])
        ax.errorbar(
            xpos,
            med,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=2,
            alpha=0.7,
        )

        ax.scatter(
            xpos,
            mean,
            marker="o",
            s=18,
            color="black",
            alpha=0.85,
            zorder=3,
        )

    ax.set_xlabel("Opening Range Window (Minutes)", fontsize=12)
    ax.set_ylabel("Expansion Ratio", fontsize=12, labelpad=18)
    ax.set_title(
        "Expansion Ratio by Weekday\n(Bars=Median, IQR=P25–P75, Dots=Mean)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(w)}m" for w in WINDOW_LENGTHS])
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)

    ax.legend(loc="upper left", ncol=2, fontsize=10)
    ax.set_ylim(0, 5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    df = _load_base_df()
    df = _add_day_fields(df)

    expected_session_bars = expected_bars_in_session(SESSION_START, SESSION_END, bar_minutes=1, inclusive=True)

    daily = _daily_session_table(df, expected_session_bars=expected_session_bars)

    valid_df, orwidth_by_len, _orcov_by_len = _or_stats_by_window(df)
    valid_dates_all_windows = valid_df.index[valid_df.all(axis=1)]

    day_table = _build_day_table(daily, valid_dates_all_windows, orwidth_by_len)

    n_days_regime_weekday, n_days_regime, n_days_weekday = _print_diagnostics(day_table)

    stats_by_regime = _stats_regime_weekday(day_table)
    _plot_regime_by_weekday(stats_by_regime, n_days_regime_weekday)
    _plot_regime_only(day_table, n_days_regime)
    _plot_weekday_only(day_table, n_days_weekday)


if __name__ == "__main__":
    main()
