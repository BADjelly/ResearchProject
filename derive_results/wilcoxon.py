import pandas as pd
from scipy.stats import wilcoxon
import json
import numpy as np

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

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(RESULTS_FILE)
df = df[df["parsed_choice"].isin(CHOICE_MAP.keys())].copy()
df["choice_value"] = (
    df["parsed_choice"]
    .map(CHOICE_MAP)
)
meta = load_metadata(SCENARIOS_FILE)

model_value_counter = {}
model_context_counter = {}
model_pair_counter = {}

overall_value_counter = {}
overall_context_counter = {}
overall_pair_counter = {}

models = sorted(df["model"].unique())
# ============================================================
# WILCOXON TESTS
# ============================================================
for model in models:
    print()
    print("=" * 80)
    print(f"MODEL: {model}")
    print("=" * 80)
    model_df = df[df["model"] == model]

    model_value_counter[model] = {}
    model_context_counter[model] = {}
    model_pair_counter[model] = {}

    baseline_runs = (
        model_df[
            model_df["baseline"] == True
        ]
        .groupby("scenario_index")["choice_value"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "baseline_mean",
            "std": "baseline_std"
        })
    )
    baseline_means = baseline_runs[["scenario_index", "baseline_mean"]]
    profile_ids = sorted(
        model_df[
            model_df["baseline"] == False
        ]["profile_id"].unique()
    )
    for profile_id in profile_ids:
        profile_means = (
            model_df[
                (model_df["baseline"] == False)
                &
                (model_df["profile_id"] == profile_id)
            ]
            .groupby("scenario_index")["choice_value"]
            .mean()
            .reset_index(name="profile_mean")
        )
        profile_stats = (
            model_df[
                (model_df["baseline"] == False)
                &
                (model_df["profile_id"] == profile_id)
            ]
            .groupby("scenario_index")["choice_value"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={
                "mean": "profile_mean",
                "std": "profile_std"
            })
        )
        merged = baseline_runs.merge(
            profile_stats,
            on="scenario_index",
            how="inner"
        )
        merged = merged.merge(
            meta,
            on="scenario_index",
            how="left"
        )

        merged["difference"] = (
            merged["profile_mean"]
            -
            merged["baseline_mean"]
        )

        differences = merged["difference"]

        baseline_values = merged["baseline_mean"]
        profile_values = merged["profile_mean"]

        stat, p_value = wilcoxon(
            baseline_values,
            profile_values,
            zero_method="wilcox"
        )

        mean_shift = differences.mean()
        median_shift = differences.median()

        changed_differences = differences[differences != 0]

        if len(changed_differences) > 0:
            median_changed_shift = changed_differences.median()
        else:
            median_changed_shift = 0

        median_abs_shift = differences.abs().median()

        nonzero_count = (differences != 0).sum()
        zero_count = (differences == 0).sum()
        large_shift_count = (differences.abs() >= 1).sum()

        top_shifts = (
            merged.loc[differences.abs() >= 1]
            .copy()
        )

        top_shifts["abs_difference"] = (
            top_shifts["difference"]
            .abs()
        )

        top_shifts = (
            top_shifts
            .sort_values("abs_difference", ascending=False)
            .head(5)
        )

        large_shift_scenarios = (
            merged.loc[differences.abs() >= 1]
            .copy()
        )

        top_values = (
            large_shift_scenarios["schwarz_value"]
            .value_counts()
            .head(5)
        )

        top_contexts = (
            large_shift_scenarios["social_context"]
            .value_counts()
            .head(5)
        )

        for value, count in large_shift_scenarios["schwarz_value"].value_counts().items():
            model_value_counter[model][value] = (
                model_value_counter[model].get(value, 0)
                + count
            )
            overall_value_counter[value] = (
                overall_value_counter.get(value, 0)
                + count
            )

        for context, count in large_shift_scenarios["social_context"].value_counts().items():
            model_context_counter[model][context] = (
                model_context_counter[model].get(context, 0)
                + count
            )
            overall_context_counter[context] = (
                overall_context_counter.get(context, 0)
                + count
            )

        pair_counts = (
            large_shift_scenarios
            .groupby(["schwarz_value", "social_context"])
            .size()
        )

        for pair, count in pair_counts.items():
            model_pair_counter[model][pair] = (
                model_pair_counter[model].get(pair, 0)
                + count
            )
            overall_pair_counter[pair] = (
                overall_pair_counter.get(pair, 0)
                + count
            )

        analysis_df = merged.copy()
        analysis_df["abs_difference"] = analysis_df["difference"].abs()
        analysis_df = analysis_df.sort_values(
            "abs_difference",
            ascending=False
        )

        percentage_summaries = []

        bins = [
            ("Top 10%", 0.0, 0.10),
            ("10-40%", 0.10, 0.40),
            ("40-80%", 0.40, 0.80),
            ("80-100%", 0.80, 1.00),
        ]

        for label, start_frac, end_frac in bins:

            start_idx = int(len(analysis_df) * start_frac)
            end_idx = max(start_idx + 1, int(len(analysis_df) * end_frac))
            subset = analysis_df.iloc[start_idx:end_idx].copy()

            clean_count = 0
            mixed_count = 0
            confusion_count = 0

            precision_gains = []

            for _, row in subset.iterrows():

                baseline_std = row["baseline_std"]
                profile_std = row["profile_std"]

                baseline_std = 0 if pd.isna(baseline_std) else baseline_std
                profile_std = 0 if pd.isna(profile_std) else profile_std

                precision_gain = baseline_std - profile_std

                precision_gains.append(precision_gain)

                if precision_gain < -0.25:
                    confusion_count += 1
                elif precision_gain < 0:
                    mixed_count += 1
                else:
                    clean_count += 1

            percentage_summaries.append(
                (
                    label,
                    subset["abs_difference"].mean(),
                    np.mean(precision_gains),
                    clean_count / len(subset),
                    mixed_count / len(subset),
                    confusion_count / len(subset)
                )
            )

        print()
        print(f"Profile: {profile_id}")
        print(f"Scenarios: {len(merged)}")
        print(f"Mean shift: {mean_shift:.4f}")
        print(f"Median shift: {median_shift:.4f}")
        print(f"Median shift (changed scenarios): {median_changed_shift:.4f}")
        print(f"Median absolute shift: {median_abs_shift:.4f}")
        print(f"Changed scenarios: {nonzero_count}")
        print(f"Unchanged scenarios: {zero_count}")
        print(f"Scenarios shifted by >=1 action category: {large_shift_count}")

        if len(top_shifts) > 0:
            print("Top 5 largest shifts:")

            for _, row in top_shifts.iterrows():
                baseline_std = row['baseline_std'] if not pd.isna(row['baseline_std']) else 0
                profile_std = row['profile_std'] if not pd.isna(row['profile_std']) else 0

                precision_gain = baseline_std - profile_std

                shift_type = "CLEAN SHIFT"
                if precision_gain < -0.25:
                    shift_type = "CONFUSION"
                elif precision_gain < 0:
                    shift_type = "MIXED"

                print(
                    f"  {row['schwarz_value']} | "
                    f"{row['social_context']} | "
                    f"shift={row['difference']:+.2f} | "
                    f"baseline_std={baseline_std:.2f} | "
                    f"profile_std={profile_std:.2f} | "
                    f"precision_gain={precision_gain:+.2f} | "
                    f"{shift_type}"
                )

        if len(large_shift_scenarios) > 0:
            print("Top 5 values among scenarios shifted by >=1 action category:")

            for value, count in top_values.items():
                print(f"  {value}: {count}")

            print("Top 5 social contexts among scenarios shifted by >=1 action category:")

            for context, count in top_contexts.items():
                print(f"  {context}: {count}")

        print(f"Wilcoxon statistic: {stat:.4f}")
        print(f"p-value: {p_value:.8f}")
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
            significance = "**"
        elif p_value < 0.05:
            significance = "*"
        else:
            significance = "ns"
        print(f"Significance: {significance}")

        print()
        print("Shift quality by affected-scenario percentage:")

        for (
            label,
            mean_abs_shift,
            avg_precision_gain,
            clean_ratio,
            mixed_ratio,
            confusion_ratio
        ) in percentage_summaries:

            print(
                f"  {label} | "
                f"mean_abs_shift={mean_abs_shift:.2f} | "
                f"precision_gain={avg_precision_gain:+.2f} | "
                f"clean={clean_ratio * 100:.0f}% | "
                f"mixed={mixed_ratio * 100:.0f}% | "
                f"confusion={confusion_ratio * 100:.0f}%"
            )

for model in models:

    print()
    print("=" * 80)
    print(f"AGGREGATED LARGE-SHIFT STATISTICS FOR {model}")
    print("=" * 80)

    top_values = sorted(
        model_value_counter[model].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    print("Top values:")
    for value, count in top_values:
        print(f"  {value}: {count}")

    top_contexts = sorted(
        model_context_counter[model].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    print("Top contexts:")
    for context, count in top_contexts:
        print(f"  {context}: {count}")

    top_pairs = sorted(
        model_pair_counter[model].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    print("Top value-context pairs:")
    for (value, context), count in top_pairs:
        print(f"  {value} | {context}: {count}")

print()
print("=" * 80)
print("AGGREGATED LARGE-SHIFT STATISTICS ACROSS BOTH MODELS")
print("=" * 80)

print("Top values:")
for value, count in sorted(overall_value_counter.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {value}: {count}")

print("Top contexts:")
for context, count in sorted(overall_context_counter.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {context}: {count}")

print("Top value-context pairs:")
for (value, context), count in sorted(overall_pair_counter.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {value} | {context}: {count}")

print()
print("Done.")