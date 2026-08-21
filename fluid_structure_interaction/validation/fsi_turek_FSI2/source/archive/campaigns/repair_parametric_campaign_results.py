"""Repair a Slurm-array campaign whose case results point to wrong run folders.

Older versions of run_parametric_actuation_library.py detected the created run
folder by comparing timestamps. That is racy in a Slurm array because other
tasks create run folders at the same time. This script reconstructs the results
from each case log, where MainKratos.py prints the actual output directory.
"""

import csv
import json
import re
import sys
from pathlib import Path

from collect_parametric_campaign_summary import FIELDNAMES
from run_parametric_actuation_library import (
    collect_metrics,
    write_identification_snapshots_csv,
    write_input_timeseries_csv,
)


RUN_DIRECTORY_PATTERN = re.compile(r"Run outputs written to:\s*(.+)\s*$")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 repair_parametric_campaign_results.py run_outputs/<campaign>")

    campaign_directory = Path(sys.argv[1])
    manifest_path = campaign_directory / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing campaign manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    cases = manifest["cases"]
    end_time = float(manifest["end_time"])
    signal_dt = float(manifest["signal_dt"])
    results = []

    results_directory = campaign_directory / "case_results"
    results_directory.mkdir(parents=True, exist_ok=True)

    for case in cases:
        label = case["label"]
        log_path = campaign_directory / "logs" / f"{label}.log"
        run_directory = find_run_directory(log_path, campaign_directory, label)
        input_path = campaign_directory / "inputs" / f"{label}.csv"
        if not input_path.exists():
            write_input_timeseries_csv(input_path, case, end_time, signal_dt)

        run_input_path = run_directory / "input_timeseries.csv"
        run_snapshot_path = run_directory / "identification_snapshots.csv"
        run_input_path.write_text(input_path.read_text())
        write_identification_snapshots_csv(
            run_directory / "beam_displacement_timeseries.csv",
            input_path,
            run_snapshot_path,
        )

        metadata = {
            "case": case,
            "input_path": str(input_path.resolve()),
            "signal_path": str(input_path.resolve()) if case["controller"] == "csv" else "",
            "identification_snapshots_path": str(run_snapshot_path.resolve()),
            "return_code": 0,
            "log_path": str(log_path.resolve()),
        }
        (run_directory / "case_metadata.json").write_text(json.dumps(metadata, indent=4))

        result = {
            **case,
            **collect_metrics(run_directory),
            "run_directory": str(run_directory.resolve()),
            "input_timeseries": str(run_input_path.resolve()),
            "identification_snapshots": str(run_snapshot_path.resolve()),
        }
        (results_directory / f"{label}.json").write_text(json.dumps(result, indent=4))
        results.append(result)
        print(f"repaired {label}: {run_directory}")

    (campaign_directory / "summary.json").write_text(json.dumps(results, indent=4))
    with (campaign_directory / "summary.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"wrote {campaign_directory / 'summary.csv'}")
    print(f"wrote {campaign_directory / 'summary.json'}")


def find_run_directory(log_path, campaign_directory, label):
    if log_path.exists():
        for line in reversed(log_path.read_text(errors="replace").splitlines()):
            match = RUN_DIRECTORY_PATTERN.search(line)
            if match:
                run_directory = Path(match.group(1))
                if run_directory.exists():
                    return run_directory
                raise RuntimeError(f"{log_path} points to missing run directory {run_directory}")

    fallback = sorted(
        campaign_directory.parent.glob(f"run_*_{label}"),
        key=lambda path: path.stat().st_mtime,
    )
    if fallback:
        return fallback[-1]

    raise RuntimeError(f"Could not find run directory for {label}. Check {log_path}.")


if __name__ == "__main__":
    main()
