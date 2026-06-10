import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

RESULTS_FILE = "../results.csv"
SCENARIOS_FILE = "../res/scenarios_extended.json"

VMIN = -1
VMAX = 1

choice_map = {
    "A": -2.5,
    "B": -1.5,
    "C": -0.5,
    "D":  0.5,
    "E":  1.5,
    "F":  2.5
}

# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(RESULTS_FILE)

df = df[df["parsed_choice"].isin(choice_map.keys())].copy()

df["choice_value"] = df["parsed_choice"].map(choice_map)

# ============================================================
# LOAD METADATA
# ============================================================

with open(SCENARIOS_FILE, "r", encoding="utf-8") as f:
    scenarios = json.load(f)

meta = pd.DataFrame([
    {
        "scenario_index": s["scenario_id"] - 1,
        "social_context": s["social_context"],
        "schwarz_value": s["schwarz_value"]
    }
    for s in scenarios
])

# ============================================================
# ATTACH METADATA
# ============================================================

df = df.merge(
    meta,
    on="scenario_index",
    how="left"
)

# ============================================================
# BASELINE STD
# ============================================================

baseline_std = (
    df[df["baseline"] == True]
    .groupby(
        ["model", "scenario_index"]
    )["choice_value"]
    .std()
    .reset_index()
)

baseline_std = baseline_std.rename(
    columns={
        "choice_value": "baseline_std"
    }
)

# ============================================================
# PROFILE STD
# ============================================================

profile_std = (
    df[df["baseline"] == False]
    .groupby(
        ["model", "profile_id", "scenario_index"]
    )["choice_value"]
    .std()
    .reset_index()
)

profile_std = profile_std.rename(
    columns={
        "choice_value": "profile_std"
    }
)

# ============================================================
# MERGE
# ============================================================

merged = profile_std.merge(
    baseline_std,
    on=["model", "scenario_index"],
    how="left"
)

# ============================================================
# PRECISION GAIN
# ============================================================

def precision_gain(row):

    baseline = row["baseline_std"]
    profile = row["profile_std"]

    if pd.isna(baseline):
        return np.nan

    if baseline == 0:
        return 0

    return (baseline - profile) / baseline

merged["precision_gain"] = merged.apply(
    precision_gain,
    axis=1
)

# ============================================================
# ADD METADATA
# ============================================================

merged = merged.merge(
    meta,
    on="scenario_index",
    how="left"
)

# ============================================================
# PROFILES
# ============================================================

profile_ids = sorted(
    merged["profile_id"].unique()
)

# ============================================================
# GENERATE HEATMAPS
# ============================================================

for profile_id in profile_ids:

    print(
        f"Generating precision heatmap for {profile_id}"
    )

    profile_df = merged[
        merged["profile_id"] == profile_id
    ]

    # ============================================================
    # COUNT IMPROVEMENTS VS DEGRADATIONS
    # ============================================================

    THRESHOLD = 0.05

    improved_count = (
        profile_df["precision_gain"] > THRESHOLD
    ).sum()

    degraded_count = (
        profile_df["precision_gain"] < -THRESHOLD
    ).sum()

    unchanged_count = (
        profile_df["precision_gain"].between(
            -THRESHOLD,
            THRESHOLD
        )
    ).sum()

    positive = profile_df["precision_gain"][
        profile_df["precision_gain"] > THRESHOLD
    ]

    negative = profile_df["precision_gain"][
        profile_df["precision_gain"] < -THRESHOLD
    ]

    total_improvement = positive.sum()
    total_degradation = abs(negative.sum())

    net_gain = total_improvement - total_degradation

    improvement_ratio = (
        total_improvement / total_degradation
        if total_degradation > 0
        else float("inf")
    )

    print(
        f"{profile_id}: "
        f"improved={improved_count}, "
        f"degraded={degraded_count}, "
        f"unchanged={unchanged_count}, "
        f"total_improvement={total_improvement:.2f}, "
        f"total_degradation={total_degradation:.2f}, "
        f"net_gain={net_gain:.2f}, "
        f"ratio={improvement_ratio:.2f}"
    )

    heatmap_df = (
        profile_df
        .groupby(
            ["social_context", "schwarz_value"]
        )["precision_gain"]
        .mean()
        .unstack()
    )

    heatmap_df = heatmap_df.sort_index()

    heatmap_df = heatmap_df[
        sorted(heatmap_df.columns)
    ]

    fig, ax = plt.subplots(
        figsize=(28, 6)
    )

    mesh = ax.pcolormesh(

        heatmap_df.values,

        cmap="RdBu_r",

        edgecolors="white",

        linewidth=1.0,

        vmin=VMIN,

        vmax=VMAX

    )

    ax.set_aspect("equal")

    ax.set_xticks(
        np.arange(
            len(heatmap_df.columns)
        ) + 0.5
    )

    ax.set_yticks(
        np.arange(
            len(heatmap_df.index)
        ) + 0.5
    )

    ax.set_xticklabels(

        heatmap_df.columns,

        rotation=90,

        fontsize=12,

        fontweight="medium"

    )

    ax.set_yticklabels(

        heatmap_df.index,

        fontsize=12,

        fontweight="medium"

    )

    cbar = plt.colorbar(mesh)

    cbar.set_label(

        "Relative Reduction in Variability",

        rotation=270,

        labelpad=25

    )

    cbar.ax.text(
        0.5,
        1.05,
        "Much More\nPrecise",
        transform=cbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="semibold",
        color="#b24d2e"
    )

    cbar.ax.text(
        0.5,
        -0.05,
        "Much Less\nPrecise",
        transform=cbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="semibold",
        color="#3a6f8f"
    )

    plt.title(
        f"Profile {profile_id}: Precision Gain Relative to Baseline\n"
        f"Improved: {improved_count}   "
        f"Degraded: {degraded_count}   "
        f"Unchanged: {unchanged_count}\n"
        f"Total Improvement: {total_improvement:.2f}   "
        f"Total Degradation: {total_degradation:.2f}   "
        f"Net: {net_gain:.2f}",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel(
        "56 Schwartz Values",
        fontsize=12,
        fontweight="semibold"
    )

    plt.ylabel(
        "11 Social Contexts",
        fontsize=12,
        fontweight="semibold"
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    output_path = (
        f"../generated_heatmaps/"
        f"profile_{profile_id}_precision_heatmap.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")

print("Done.")