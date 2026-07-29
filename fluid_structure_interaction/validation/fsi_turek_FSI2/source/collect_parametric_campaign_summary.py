import csv
import json
import sys
from collections import Counter
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

    manifest_path = campaign_directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        case_order = {
            case["label"]: index
            for index, case in enumerate(manifest.get("cases", []))
        }
        results.sort(key=lambda result: case_order.get(result["label"], len(case_order)))

    run_directory_counts = Counter(result.get("run_directory", "") for result in results)
    duplicate_run_directories = [
        run_directory
        for run_directory, count in run_directory_counts.items()
        if run_directory and count > 1
    ]
    if duplicate_run_directories:
        print("WARNING: multiple case results point to the same run directory:")
        for run_directory in duplicate_run_directories:
            print(f"  {run_directory}: {run_directory_counts[run_directory]} cases")

    (campaign_directory / "summary.json").write_text(json.dumps(results, indent=4))
    with (campaign_directory / "summary.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"wrote {campaign_directory / 'summary.csv'}")
    print(f"wrote {campaign_directory / 'summary.json'}")


if __name__ == "__main__":
    main()
