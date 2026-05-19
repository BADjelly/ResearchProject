import csv
import sys
from collections import Counter

INPUT_FILE = "results.csv"   # ← change to your input CSV path
OUTPUT_FILE = "majority_results.csv"

def majority_vote(choices):
    """Return the majority choice, or None if there's no clear majority."""
    counts = Counter(choices)
    top_two = counts.most_common(2)
    if len(top_two) == 1:
        return top_two[0][0]  # Only one distinct value
    if top_two[0][1] > top_two[1][1]:
        return top_two[0][0]  # Clear majority
    return None  # Tie

def main():
    # Group rows by scenario_index, baseline-filtered
    scenarios = {}  # scenario_index -> list of rows
    fieldnames = None

    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            if row["baseline"].strip() != "True":
                continue

            idx = row["scenario_index"].strip()
            scenarios.setdefault(idx, []).append(row)

    if not scenarios:
        print("No baseline=True rows found.")
        sys.exit(1)

    output_rows = []
    skipped = []

    for scenario_index, rows in sorted(scenarios.items(), key=lambda x: int(x[0])):
        choices = [r["parsed_choice"].strip() for r in rows]
        winner = majority_vote(choices)

        if winner is None:
            counts = Counter(choices)
            print(
                f"No majority for scenario_index={scenario_index} "
                f"(choices: {dict(counts)}) — skipping."
            )
            skipped.append(scenario_index)
            continue

        # Use the first row that has the winning choice as the representative
        representative = next(r for r in rows if r["parsed_choice"].strip() == winner)
        output_rows.append(representative)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nDone. {len(output_rows)} scenario(s) written to '{OUTPUT_FILE}'.", end="")
    if skipped:
        print(f" {len(skipped)} skipped: scenario_index in {skipped}.")
        answer = input(
            f"\nRemove skipped scenario(s) {skipped} (baseline=True rows only) "
            f"from '{INPUT_FILE}'? [y/n] "
        ).strip().lower()
        if answer in ("y", "yes"):
            skipped_set = set(skipped)
            kept_rows = []
            with open(INPUT_FILE, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    is_skipped_baseline = (
                        row["scenario_index"].strip() in skipped_set
                        and row["baseline"].strip() == "True"
                    )
                    if not is_skipped_baseline:
                        kept_rows.append(row)

            with open(INPUT_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(kept_rows)

            removed = len([r for r in kept_rows if r not in kept_rows])  # for message
            print(f"Removed baseline=True rows for scenario_index {skipped} from '{INPUT_FILE}'.")
        else:
            print("Input file left unchanged.")
    else:
        print()

if __name__ == "__main__":
    main()