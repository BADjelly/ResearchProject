import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

RESULTS_FILE = "results.csv"
SCENARIOS_FILE = "res/scenarios_extended.json"

CHOICE_MAP = {
    "A": -2.5,
    "B": -1.5,
    "C": -0.5,
    "D":  0.5,
    "E":  1.5,
    "F":  2.5
}

STDMIN = 0
STDMAX = 2

VMIN = -3
VMAX = 3


# ============================================================
# HELPERS
# ============================================================

def load_metadata(scenarios_file):
    with open(scenarios_file, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    return pd.DataFrame([
        {
            "scenario_index": s["scenario_id"] - 1,
            "social_context": s["social_context"],
            "schwarz_value": s["schwarz_value"]
        }
        for s in scenarios
    ])


def attach_metadata(df, meta):
    return df.merge(meta, on="scenario_index", how="left")


def build_std_heatmap(df, value_column):
    heatmap = (
        df.groupby(
            ["social_context", "schwarz_value"]
        )[value_column]
        .std()
        .unstack()
    )

    return (
        heatmap
        .sort_index()
        .reindex(sorted(heatmap.columns), axis=1)
    )


def build_mean_heatmap(df, value_column):
    heatmap = (
        df.groupby(
            ["social_context", "schwarz_value"]
        )[value_column]
        .mean()
        .unstack()
    )

    return (
        heatmap
        .sort_index()
        .reindex(sorted(heatmap.columns), axis=1)
    )


def plot_heatmap(
        heatmap_df,
        title,
        colorbar_label,
        top_label,
        bottom_label,
        output_path,
        vmin,
        vmax):

    fig, ax = plt.subplots(figsize=(28, 6))

    mesh = ax.pcolormesh(
        heatmap_df.values,
        cmap="RdBu_r",
        edgecolors="white",
        linewidth=1.0,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_aspect("equal")

    ax.set_xticks(np.arange(len(heatmap_df.columns)) + 0.5)
    ax.set_yticks(np.arange(len(heatmap_df.index)) + 0.5)

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
        colorbar_label,
        rotation=270,
        labelpad=25
    )

    cbar.ax.text(
        0.5, 1.05, top_label,
        transform=cbar.ax.transAxes,
        ha="center", va="bottom",
        fontsize=11, fontweight="semibold",
        color="#b24d2e"
    )

    cbar.ax.text(
        0.5, -0.05, bottom_label,
        transform=cbar.ax.transAxes,
        ha="center", va="top",
        fontsize=11, fontweight="semibold",
        color="#3a6f8f"
    )

    plt.title(title, fontsize=14, fontweight="bold")

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
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# LOAD DATA ONCE
# ============================================================

df = pd.read_csv(RESULTS_FILE)

df = df[df["parsed_choice"].isin(CHOICE_MAP.keys())].copy()
df["choice_value"] = df["parsed_choice"].map(CHOICE_MAP)

meta = load_metadata(SCENARIOS_FILE)
df = attach_metadata(df, meta)

# ============================================================
# BASELINE STD HEATMAP
# ============================================================

baseline_df = df[df["baseline"] == True]

baseline_heatmap = build_std_heatmap(
    baseline_df,
    "choice_value"
)

plot_heatmap(
    heatmap_df=baseline_heatmap,
    title="Behavioral Variability (Standard Deviation) Across Values and Social Contexts",
    colorbar_label="Behavioral Instability",
    top_label="Highly\nVariable",
    bottom_label="Highly\nConsistent",
    output_path="generated_heatmaps/baseline_stddev_heatmap.png",
    vmin=STDMIN,
    vmax=STDMAX
)

# ============================================================
# PROFILE DEVIATIONS
# ============================================================

baseline_means = (
    df[df["baseline"] == True]
    .groupby(["model", "scenario_index"])["choice_value"]
    .mean()
    .reset_index(name="baseline_mean")
)

profile_means = (
    df[df["baseline"] == False]
    .groupby(["model", "profile_id", "scenario_index"])["choice_value"]
    .mean()
    .reset_index()
)

merged = profile_means.merge(
    baseline_means,
    on=["model", "scenario_index"],
    how="left"
)

merged["delta"] = (
    merged["choice_value"]
    -
    merged["baseline_mean"]
)

merged = attach_metadata(merged, meta)

profile_ids = sorted(merged["profile_id"].unique())

for profile_id in profile_ids:

    print(f"Generating heatmaps for profile {profile_id}")

    profile_delta_df = merged[
        merged["profile_id"] == profile_id
    ]

    deviation_heatmap = build_mean_heatmap(
        profile_delta_df,
        "delta"
    )

    plot_heatmap(
        heatmap_df=deviation_heatmap,
        title=f"Profile {profile_id}: Deviation From Baseline",
        colorbar_label="Deviation From Baseline",
        top_label="Strongly\nMisaligned",
        bottom_label="Strongly\nAligned",
        output_path=f"generated_heatmaps/profile_{profile_id}_heatmap.png",
        vmin=VMIN,
        vmax=VMAX
    )

    profile_choices = df[
        (df["baseline"] == False)
        &
        (df["profile_id"] == profile_id)
    ]

    std_heatmap = build_std_heatmap(
        profile_choices,
        "choice_value"
    )

    plot_heatmap(
        heatmap_df=std_heatmap,
        title=f"Profile {profile_id}: Response Variability",
        colorbar_label="Response Standard Deviation",
        top_label="Highly\nVariable",
        bottom_label="Highly\nConsistent",
        output_path=f"generated_heatmaps/profile_{profile_id}_response_std_heatmap.png",
        vmin=STDMIN,
        vmax=STDMAX
    )

print("Done.")
