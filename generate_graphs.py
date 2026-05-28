import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

RESULTS_FILE = "results.csv"

#MIN_REPETITIONS_WARNING = 5

# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(RESULTS_FILE)

# ============================================================
# KEEP BASELINE ONLY
# ============================================================

df = df[df["baseline"] == True].copy()

# ============================================================
# VALID CHOICES
# ============================================================

VALID = ["A", "B", "C", "D", "E", "F"]

df = df[df["parsed_choice"].isin(VALID)]

# ============================================================
# WARNING
# ============================================================

rep_count = df["repetition"].nunique()

# if rep_count < MIN_REPETITIONS_WARNING:
#
#     print(
#         f"WARNING: Only {rep_count} repetitions detected. "
#         f"Entropy estimates may be noisy."
#     )

# ============================================================
# ENTROPY FUNCTION
# ============================================================

def entropy(choices):

    counts = pd.Series(choices).value_counts(normalize=True)

    probs = counts.values

    return -np.sum(probs * np.log2(probs))

# ============================================================
# LOAD social_context + schwarz_value FROM scenarios.json
# ============================================================

import json

SCENARIOS_FILE = "res/scenarios_extended.json"

with open(SCENARIOS_FILE, "r", encoding="utf-8") as f:
    scenarios_data = json.load(f)

# Build lookup:
# scenario_id -> metadata
scenario_metadata = {}

for scenario in scenarios_data:
    scenario_metadata[scenario["scenario_id"]-1] = {
        "social_context": scenario["social_context"],
        "schwarz_value": scenario["schwarz_value"]
    }

# ============================================================
# ATTACH METADATA FROM scenarios.json
# ============================================================

df["social_context"] = df["scenario_index"].map(
    lambda x: scenario_metadata[x]["social_context"]
)

df["schwarz_value"] = df["scenario_index"].map(
    lambda x: scenario_metadata[x]["schwarz_value"]
)

ROW_COLUMN = "social_context"
COL_COLUMN = "schwarz_value"

# ============================================================
# COMPUTE ENTROPY MATRIX
# ============================================================

heatmap_df = (
    df.groupby(
        [ROW_COLUMN, COL_COLUMN]
    )["parsed_choice"]
    .apply(entropy)
    .unstack()
)

# ============================================================
# SORT
# ============================================================

heatmap_df = heatmap_df.sort_index()

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(22, 8))

im = plt.imshow(
    heatmap_df,
    aspect="auto",
    interpolation="nearest"
)

# ============================================================
# AXES
# ============================================================

plt.yticks(
    range(len(heatmap_df.index)),
    heatmap_df.index,
    fontsize=10
)

plt.xticks(
    range(len(heatmap_df.columns)),
    heatmap_df.columns,
    rotation=90,
    fontsize=8
)

# ============================================================
# COLORBAR
# ============================================================

cbar = plt.colorbar(im)

cbar.set_label(
    "Choice Entropy\n(Behavioral Instability)",
    rotation=270,
    labelpad=25
)

# ============================================================
# LABELS
# ============================================================

plt.xlabel(
    "56 Schwartz Values",
    fontsize=12,
    fontweight="bold"
)

plt.ylabel(
    "Social Contexts",
    fontsize=12,
    fontweight="bold"
)

plt.title(
    "Behavioral Instability Across Values and Social Contexts\n"
    "(Baseline Condition)",
    fontsize=14,
    fontweight="bold"
)

# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

plt.savefig(
    "baseline_entropy_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()