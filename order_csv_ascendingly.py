import csv
import sys

INPUT_FILE = "results.csv"   # ← change to your input CSV path

def main():
    with open(INPUT_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    rows.sort(key=lambda r: int(r["trial_index"]))

    with open(INPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Sorted {len(rows)} rows by trial_index and saved to '{INPUT_FILE}'.")

if __name__ == "__main__":
    main()