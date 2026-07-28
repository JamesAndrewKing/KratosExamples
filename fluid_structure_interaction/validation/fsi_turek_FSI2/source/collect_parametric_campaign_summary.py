import csv
import json
import sys
from pathlib import Path


FIELDNAMES = [
    "label",
    "kind",
    "controller",
    "amplitude",
    "frequency",
    "frequency_start",
    "frequency_end",
    "amplitude_start",
    "amplitude_end",
    "end_time",
    "last_time",
    "max_abs_tip_y",
    "rms_tip_y",
    "final_tip_y",
    "max_abs_control",
    "rms_control",
    "number_of_samples",
    "run_directory",
    "input_timeseries",
    "identification_snapshots",
]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 collect_parametric_campaign_summary.py run_outputs/<campaign>")

    campaign_directory = Path(sys.argv[1])
    results_directory = campaign_directory / "case_results"
    results = []
    for result_path in sorted(results_directory.glob("*.json")):
        results.append(json.loads(result_path.read_text()))

    if not results:
        raise RuntimeError(f"No case result JSON files found in {results_directory}.")

    (campaign_directory / "summary.json").write_text(json.dumps(results, indent=4))
    with (campaign_directory / "summary.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"wrote {campaign_directory / 'summary.csv'}")
    print(f"wrote {campaign_directory / 'summary.json'}")


if __name__ == "__main__":
    main()
