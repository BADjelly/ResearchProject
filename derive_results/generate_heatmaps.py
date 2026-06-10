import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

RESULTS_FILE = "../results.csv"
SCENARIOS_FILE = "../res/scenarios_extended.json"

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

AVGMIN = -2.5
AVGMAX = 2.5

# SHIFTMIN = -5
# SHIFTMAX = 5
SHIFTMIN = -2
SHIFTMAX = 2

RANGEMIN = 0
RANGEMAX = 5

PRECISIONMIN = -1
PRECISIONMAX = 1


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


def build_range_heatmap(df, value_column):
    heatmap = (
        df.groupby(
            ["social_context", "schwarz_value"]
        )[value_column]
        .agg(lambda x: x.max() - x.min())
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
        output_paths,
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

    for output_path in output_paths:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Saved: {output_path}")

    plt.close()


# ============================================================
# LOAD DATA ONCE
# ============================================================

df = pd.read_csv(RESULTS_FILE)

df = df[df["parsed_choice"].isin(CHOICE_MAP.keys())].copy()
df["choice_value"] = df["parsed_choice"].map(CHOICE_MAP)

meta = load_metadata(SCENARIOS_FILE)
df = attach_metadata(df, meta)

models = sorted(df["model"].unique())

for model in models:

    model_df = df[df["model"] == model].copy()

    safe_model = model.replace("/", "_")

    # ============================================================
    # BASELINE STD HEATMAP
    # ============================================================

    baseline_df = model_df[model_df["baseline"] == True]

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
        output_paths=[
            f"../generated_heatmaps/by_metric/baseline_stddev_heatmap_{safe_model}.png",
            f"../generated_heatmaps/by_profile/baseline_stddev_heatmap_{safe_model}.png"
        ],
        vmin=STDMIN,
        vmax=STDMAX
    )

    #
    # ============================================================
    # BASELINE AVERAGE CHOICE HEATMAP
    # ============================================================

    baseline_average_heatmap = build_mean_heatmap(
        baseline_df,
        "choice_value"
    )

    plot_heatmap(
        heatmap_df=baseline_average_heatmap,
        title="Average Choice Across Values and Social Contexts",
        colorbar_label="Average Choice Value",
        top_label="Strongly\nMisaligned",
        bottom_label="Strongly\nAligned",
        output_paths=[
            f"../generated_heatmaps/by_metric/baseline_average_choice_heatmap_{safe_model}.png",
            f"../generated_heatmaps/by_profile/baseline_average_choice_heatmap_{safe_model}.png"
        ],
        vmin=AVGMIN,
        vmax=AVGMAX
    )

    # ============================================================
    # BASELINE RANGE HEATMAP
    # ============================================================

    range_choice_map = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6
    }

    baseline_range_df = baseline_df.copy()
    baseline_range_df["range_choice_value"] = (
        baseline_range_df["parsed_choice"]
        .map(range_choice_map)
    )

    baseline_range_heatmap = build_range_heatmap(
        baseline_range_df,
        "range_choice_value"
    )

    plot_heatmap(
        heatmap_df=baseline_range_heatmap,
        title="Behavioral Response Range Across Values and Social Contexts",
        colorbar_label="Maximum Response Spread",
        top_label="A ↔ F\nObserved",
        bottom_label="All Runs\nIdentical",
        output_paths=[
            f"../generated_heatmaps/by_metric/baseline_range_heatmap_{safe_model}.png",
            f"../generated_heatmaps/by_profile/baseline_range_heatmap_{safe_model}.png"
        ],
        vmin=RANGEMIN,
        vmax=RANGEMAX
    )

    # ============================================================
    # PROFILE DEVIATIONS
    # ============================================================

    baseline_means = (
        model_df[model_df["baseline"] == True]
        .groupby(["model", "scenario_index"])["choice_value"]
        .mean()
        .reset_index(name="baseline_mean")
    )

    profile_means = (
        model_df[model_df["baseline"] == False]
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
            output_paths=[
                f"../generated_heatmaps/by_profile/{profile_id}/deviation_from_baseline_{safe_model}.png",
                f"../generated_heatmaps/by_metric/deviation_from_baseline/{profile_id}_{safe_model}.png"
            ],
            vmin=VMIN,
            vmax=VMAX
        )

        profile_choices = model_df[
            (model_df["baseline"] == False)
            &
            (model_df["profile_id"] == profile_id)
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
            output_paths=[
                f"../generated_heatmaps/by_profile/{profile_id}/response_std_{safe_model}.png",
                f"../generated_heatmaps/by_metric/response_std/{profile_id}_{safe_model}.png"
            ],
            vmin=STDMIN,
            vmax=STDMAX
        )

        # ============================================================
        # PROFILE AVERAGE CHOICE HEATMAP
        # ============================================================

        profile_average_heatmap = build_mean_heatmap(
            profile_choices,
            "choice_value"
        )

        plot_heatmap(
            heatmap_df=profile_average_heatmap,
            title=f"Profile {profile_id}: Average Choice",
            colorbar_label="Average Choice Value",
            top_label="Strongly\nMisaligned",
            bottom_label="Strongly\nAligned",
            output_paths=[
                f"../generated_heatmaps/by_profile/{profile_id}/average_choice_{safe_model}.png",
                f"../generated_heatmaps/by_metric/average_choice/{profile_id}_{safe_model}.png"
            ],
            vmin=AVGMIN,
            vmax=AVGMAX
        )

        # ============================================================
        # PROFILE AVERAGE CHOICE SHIFT FROM BASELINE
        # ============================================================

        baseline_average_heatmap_aligned = baseline_average_heatmap.reindex(
            index=profile_average_heatmap.index,
            columns=profile_average_heatmap.columns
        )

        average_shift_heatmap = (
            profile_average_heatmap
            -
            baseline_average_heatmap_aligned
        )

        plot_heatmap(
            heatmap_df=average_shift_heatmap,
            title=f"Profile {profile_id}: Average Choice Shift From Baseline",
            colorbar_label="Average Choice Shift",
            top_label="More\nMisaligned",
            bottom_label="More\nAligned",
            output_paths=[
                f"../generated_heatmaps/by_profile/{profile_id}/average_shift_{safe_model}.png",
                f"../generated_heatmaps/by_metric/average_shift/{profile_id}_{safe_model}.png"
            ],
            vmin=SHIFTMIN,
            vmax=SHIFTMAX
        )

        # ============================================================
        # PROFILE RANGE HEATMAP
        # ============================================================

        profile_range_choices = profile_choices.copy()

        profile_range_choices["range_choice_value"] = (
            profile_range_choices["parsed_choice"]
            .map(range_choice_map)
        )

        profile_range_heatmap = build_range_heatmap(
            profile_range_choices,
            "range_choice_value"
        )

        plot_heatmap(
            heatmap_df=profile_range_heatmap,
            title=f"Profile {profile_id}: Response Range",
            colorbar_label="Maximum Response Spread",
            top_label="A ↔ F\nObserved",
            bottom_label="All Runs\nIdentical",
            output_paths=[
                f"../generated_heatmaps/by_profile/{profile_id}/response_range_{safe_model}.png",
                f"../generated_heatmaps/by_metric/response_range/{profile_id}_{safe_model}.png"
            ],
            vmin=RANGEMIN,
            vmax=RANGEMAX
        )

        # ============================================================
        # PROFILE PRECISION GAIN HEATMAP
        # ============================================================

        baseline_std = (
            model_df[model_df["baseline"] == True]
            .groupby(
                ["model", "scenario_index"]
            )["choice_value"]
            .std()
            .reset_index(name="baseline_std")
        )

        profile_std = (
            profile_choices
            .groupby(
                ["model", "profile_id", "scenario_index"]
            )["choice_value"]
            .std()
            .reset_index(name="profile_std")
        )

        precision_df = profile_std.merge(
            baseline_std,
            on=["model", "scenario_index"],
            how="left"
        )

        def precision_gain(row):

            baseline = row["baseline_std"]
            profile = row["profile_std"]

            if pd.isna(baseline):
                return np.nan

            if baseline == 0:
                return 0

            return (baseline - profile) / baseline

        precision_df["precision_gain"] = precision_df.apply(
            precision_gain,
            axis=1
        )

        precision_df = attach_metadata(
            precision_df,
            meta
        )

        THRESHOLD = 0.05

        improved_count = (
            precision_df["precision_gain"] > THRESHOLD
        ).sum()

        degraded_count = (
            precision_df["precision_gain"] < -THRESHOLD
        ).sum()

        unchanged_count = (
            precision_df["precision_gain"].between(
                -THRESHOLD,
                THRESHOLD
            )
        ).sum()

        positive = precision_df[
            "precision_gain"
        ][precision_df["precision_gain"] > THRESHOLD]

        negative = precision_df[
            "precision_gain"
        ][precision_df["precision_gain"] < -THRESHOLD]

        total_improvement = positive.sum()
        total_degradation = abs(negative.sum())
        net_gain = total_improvement - total_degradation

        precision_heatmap = (
            precision_df
            .groupby(
                ["social_context", "schwarz_value"]
            )["precision_gain"]
            .mean()
            .unstack()
        )

        precision_heatmap = precision_heatmap.sort_index()
        precision_heatmap = precision_heatmap[
            sorted(precision_heatmap.columns)
        ]

        plot_heatmap(
            heatmap_df=precision_heatmap,
            title=(
                f"Profile {profile_id}: Precision Gain Relative to Baseline\n"
                f"Improved: {improved_count}   "
                f"Degraded: {degraded_count}   "
                f"Unchanged: {unchanged_count}\n"
                f"Total Improvement: {total_improvement:.2f}   "
                f"Total Degradation: {total_degradation:.2f}   "
                f"Net: {net_gain:.2f}"
            ),
            colorbar_label="Relative Reduction in Variability",
            top_label="Much More\nPrecise",
            bottom_label="Much Less\nPrecise",
            output_paths=[
                f"../generated_heatmaps/by_profile/{profile_id}/precision_gain_{safe_model}.png",
                f"../generated_heatmaps/by_metric/precision_gain/{profile_id}_{safe_model}.png"
            ],
            vmin=PRECISIONMIN,
            vmax=PRECISIONMAX
        )

print("Done.")
