import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.data import load_filtered_df

RESOLUTION = "minute"
WEEKDAYS_ONLY = True

audit_df = load_filtered_df(min_perc_first_hour=0.0, min_perc_total=0.0)

audit_df = audit_df.copy()
audit_df["Is_Missing"] = audit_df["close"].isna().astype(int)
audit_df["Date_Only"] = audit_df.index.date

if RESOLUTION == "minute":
    audit_df["X_Key"] = audit_df.index.hour * 60 + audit_df.index.minute
    x_label = "Time of Day"
    cbar_label = "Missing (1 = Gap)"
    xtick_step = 60
    expected_columns = np.arange(7 * 60, 17 * 60)

elif RESOLUTION == "hour":
    audit_df["X_Key"] = audit_df.index.hour
    x_label = "Hour of Day"
    cbar_label = "Fraction Missing"
    xtick_step = 1
    expected_columns = np.arange(7, 17)

heatmap_pivot = (
    audit_df
    .groupby(["Date_Only", "X_Key"])["Is_Missing"]
    .mean()
    .unstack()
    .reindex(columns=expected_columns)
)

plt.figure(figsize=(22, 12))
sns.set_theme(style="white")

ax = sns.heatmap(
    heatmap_pivot,
    cmap="YlOrRd",
    cbar_kws={"label": cbar_label},
    yticklabels=100,
    xticklabels=xtick_step,
)

week_label = "Mon–Fri" if WEEKDAYS_ONLY else "All days"
plt.title(f"Missingness Heatmap ({week_label})", fontsize=16, fontweight="bold")
plt.xlabel(x_label)
plt.ylabel("Observation Date", labelpad=18)

xticks = ax.get_xticks()
labels = []
for t in xticks:
    idx = int(round(t))
    if 0 <= idx < len(expected_columns):
        val = expected_columns[idx]
        if RESOLUTION == "minute":
            labels.append(f"{val // 60:02d}:{val % 60:02d}")
        else:
            labels.append(f"{val:02d}:00")
    else:
        labels.append("")
ax.set_xticklabels(labels, rotation=0)

plt.tight_layout()
plt.show()

total_expected = len(audit_df)
total_missing = int(audit_df["Is_Missing"].sum())
completeness = 100 * (1 - total_missing / total_expected) if total_expected else float("nan")

print(f"Total Expected Bars: {total_expected}")
print(f"Total Missing Bars: {total_missing}")
print(f"Data Completeness: {completeness}%")
