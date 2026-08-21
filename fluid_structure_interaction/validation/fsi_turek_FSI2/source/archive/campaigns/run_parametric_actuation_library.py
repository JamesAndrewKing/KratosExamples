"""Run the Turek FSI2 parametric actuator library.

The generated identification data uses
    value = amplitude * cos(theta_rad)
with Delta = [amplitude, omega_rad_s] and Theta = [theta_rad].
"""

import csv
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PYTHONPATH = os.environ.get("KRATOS_FSI_PYTHONPATH")
DYLD_LIBRARY_PATH = os.environ.get("KRATOS_FSI_DYLD_LIBRARY_PATH")
LD_LIBRARY_PATH = os.environ.get("KRATOS_FSI_LD_LIBRARY_PATH")

# The uncontrolled FSI2 response we observed is in the few-Hz range. The grid below
# is deliberately centered near 3.8 Hz, with moderate excursions to expose detuning.
NATURAL_FREQUENCY_ESTIMATE_HZ = 3.8
STEADY_FREQUENCIES_HZ = [2.4, 3.0, 3.4, 3.8, 4.2, 4.8, 5.6]
STEADY_AMPLITUDES = [0.02, 0.04, 0.06, 0.08]

DEFAULT_END_TIME = 2.0
DEFAULT_OUTPUT_INTERVAL = 0.01
DEFAULT_SIGNAL_DT = 0.002
DEFAULT_WRITE_PARAVIEW = False
TWO_PI = 2.0 * math.pi


def main():
    source_directory = Path(__file__).resolve().parent
    os.chdir(source_directory)

    campaign_label = os.environ.get(
        "KRATOS_FSI_LIBRARY_LABEL",
        datetime.now().strftime("parametric_actuation_%Y%m%d_%H%M%S"),
    )
    campaign_directory = Path("run_outputs") / campaign_label
    campaign_directory.mkdir(parents=True, exist_ok=True)

    end_time = read_float("KRATOS_FSI_LIBRARY_END_TIME", DEFAULT_END_TIME)
    output_interval = read_float("KRATOS_FSI_LIBRARY_OUTPUT_INTERVAL", DEFAULT_OUTPUT_INTERVAL)
    signal_dt = read_float("KRATOS_FSI_LIBRARY_SIGNAL_DT", DEFAULT_SIGNAL_DT)
    write_paraview = read_bool("KRATOS_FSI_LIBRARY_WRITE_PARAVIEW", DEFAULT_WRITE_PARAVIEW)
    limit = read_optional_int("KRATOS_FSI_LIBRARY_LIMIT")
    case_index = read_optional_int("KRATOS_FSI_LIBRARY_CASE_INDEX")

    all_cases = build_case_library(end_time)
    cases = all_cases
    if limit is not None:
        cases = cases[:limit]
    if case_index is not None:
        if case_index < 0 or case_index >= len(all_cases):
            raise IndexError(
                f"KRATOS_FSI_LIBRARY_CASE_INDEX={case_index} is outside "
                f"the case range 0..{len(all_cases) - 1}."
            )
        cases = [all_cases[case_index]]

    manifest = {
        "campaign_label": campaign_label,
        "campaign_directory": str(campaign_directory.resolve()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "natural_frequency_estimate_hz": NATURAL_FREQUENCY_ESTIMATE_HZ,
        "end_time": end_time,
        "output_interval": output_interval,
        "signal_dt": signal_dt,
        "write_paraview": write_paraview,
        "input_timeseries_convention": (
            "value = amplitude * cos(theta_rad), "
            "theta_rad = theta_unwrapped_rad mod 2pi"
        ),
        "identification_snapshot_columns": {
            "X": "beam displacement measurement columns prefixed with measurement_",
            "Delta": ["delta_amplitude", "delta_omega_rad_s"],
            "Theta": ["theta_rad"],
        },
        "total_number_of_cases": len(all_cases),
        "cases": all_cases,
    }
    manifest_path = campaign_directory / "manifest.json"
    if case_index is None or not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=4))

    if case_index is not None:
        selection_directory = campaign_directory / "case_selections"
        selection_directory.mkdir(parents=True, exist_ok=True)
        selection_path = selection_directory / f"{case_index:03d}_{cases[0]['label']}.json"
        selection_path.write_text(json.dumps({
            "case_index": case_index,
            "case": cases[0],
        }, indent=4))

    results = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case['label']}", flush=True)
        run_directory = run_case(case, campaign_directory, end_time, output_interval, signal_dt, write_paraview)
        metrics = collect_metrics(run_directory)
        result = {
            **case,
            **metrics,
            "run_directory": str(run_directory.resolve()),
            "input_timeseries": str((run_directory / "input_timeseries.csv").resolve()),
            "identification_snapshots": str((run_directory / "identification_snapshots.csv").resolve()),
        }
        results.append(result)
        write_case_result(campaign_directory, result)
        if case_index is None:
            append_result(campaign_directory / "summary.csv", result)
        print(format_result(result), flush=True)

    if case_index is None:
        (campaign_directory / "summary.json").write_text(json.dumps(results, indent=4))
    print(f"campaign={campaign_directory.resolve()}")


def build_case_library(end_time):
    cases = [{
        "label": "passive_baseline",
        "kind": "passive",
        "controller": "sinusoidal",
        "amplitude": 0.0,
        "frequency": 0.0,
        "phase": 0.0,
        "description": "No actuation baseline with identical output channels.",
    }]

    for amplitude in STEADY_AMPLITUDES:
        for frequency in STEADY_FREQUENCIES_HZ:
            cases.append({
                "label": f"sine_A{amplitude:.3f}_f{frequency:.2f}".replace(".", "p"),
                "kind": "single_sine",
                "controller": "sinusoidal",
                "amplitude": amplitude,
                "frequency": frequency,
                "phase": 0.0,
                "description": "Constant-amplitude periodic forcing.",
            })

    # A few higher-amplitude edge cases, concentrated near resonance. These are
    # useful for nonlinear identification, but kept sparse to avoid a brittle campaign.
    for amplitude, frequency in [(0.10, 3.4), (0.10, 3.8), (0.10, 4.2), (0.12, 3.8)]:
        cases.append({
            "label": f"sine_high_A{amplitude:.3f}_f{frequency:.2f}".replace(".", "p"),
            "kind": "single_sine",
            "controller": "sinusoidal",
            "amplitude": amplitude,
            "frequency": frequency,
            "phase": 0.0,
            "description": "Sparse high-amplitude near-resonance forcing.",
        })

    sweep_specs = [
        ("chirp_up_A0p06_f2p4_to_5p6", 0.06, 2.4, 5.6),
        ("chirp_down_A0p06_f5p6_to_2p4", 0.06, 5.6, 2.4),
        ("chirp_up_A0p08_f2p8_to_4p8", 0.08, 2.8, 4.8),
        ("chirp_down_A0p08_f4p8_to_2p8", 0.08, 4.8, 2.8),
    ]
    for label, amplitude, start_frequency, end_frequency in sweep_specs:
        cases.append({
            "label": label,
            "kind": "frequency_sweep",
            "controller": "csv",
            "amplitude": amplitude,
            "frequency_start": start_frequency,
            "frequency_end": end_frequency,
            "description": "Linear chirp with constant amplitude.",
        })

    amplitude_sweep_specs = [
        ("amp_up_f3p8_A0p02_to_0p10", 3.8, 0.02, 0.10),
        ("amp_down_f3p8_A0p10_to_0p02", 3.8, 0.10, 0.02),
        ("amp_up_f4p2_A0p02_to_0p10", 4.2, 0.02, 0.10),
        ("amp_down_f4p2_A0p10_to_0p02", 4.2, 0.10, 0.02),
    ]
    for label, frequency, start_amplitude, end_amplitude in amplitude_sweep_specs:
        cases.append({
            "label": label,
            "kind": "amplitude_sweep",
            "controller": "csv",
            "frequency": frequency,
            "amplitude_start": start_amplitude,
            "amplitude_end": end_amplitude,
            "description": "Linear amplitude ramp at fixed frequency.",
        })

    cases.append({
        "label": "amp_and_freq_up_A0p02_to_0p10_f2p8_to_4p8",
        "kind": "amplitude_frequency_sweep",
        "controller": "csv",
        "amplitude_start": 0.02,
        "amplitude_end": 0.10,
        "frequency_start": 2.8,
        "frequency_end": 4.8,
        "description": "Combined amplitude and frequency up-sweep.",
    })
    cases.append({
        "label": "amp_and_freq_down_A0p10_to_0p02_f4p8_to_2p8",
        "kind": "amplitude_frequency_sweep",
        "controller": "csv",
        "amplitude_start": 0.10,
        "amplitude_end": 0.02,
        "frequency_start": 4.8,
        "frequency_end": 2.8,
        "description": "Combined amplitude and frequency down-sweep.",
    })

    for case in cases:
        case["end_time"] = end_time
    return cases


def run_case(case, campaign_directory, end_time, output_interval, signal_dt, write_paraview):
    run_directory = campaign_directory / "runs" / case["label"]
    if run_directory.exists() and any(run_directory.iterdir()):
        raise RuntimeError(
            f"Run directory {run_directory} already exists and is not empty. "
            "Use a new KRATOS_FSI_LIBRARY_LABEL for a fresh campaign."
        )

    input_path = campaign_directory / "inputs" / f"{case['label']}.csv"
    write_input_timeseries_csv(input_path, case, end_time, signal_dt)
    signal_path = None
    if case["controller"] == "csv":
        signal_path = input_path

    environment = os.environ.copy()
    environment.update({
        "KRATOS_FSI_RUN_LABEL": case["label"],
        "KRATOS_FSI_RUN_OUTPUT_DIRECTORY": str(run_directory.resolve()),
        "KRATOS_FSI_END_TIME": str(end_time),
        "KRATOS_FSI_OUTPUT_INTERVAL": str(output_interval),
        "KRATOS_FSI_WRITE_PARAVIEW": "1" if write_paraview else "0",
        "KRATOS_FSI_CONTROLLER_TYPE": case["controller"],
        "KRATOS_FSI_ACTUATOR_AMPLITUDE": str(case.get("amplitude", 0.0)),
        "KRATOS_FSI_ACTUATOR_FREQUENCY": str(case.get("frequency", 0.0)),
        "KRATOS_FSI_ACTUATOR_PHASE": str(case.get("phase", 0.0)),
    })
    if PYTHONPATH:
        environment["PYTHONPATH"] = PYTHONPATH
    if DYLD_LIBRARY_PATH:
        environment["DYLD_LIBRARY_PATH"] = DYLD_LIBRARY_PATH
    if LD_LIBRARY_PATH:
        environment["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH
    if signal_path is not None:
        environment.update({
            "KRATOS_FSI_ACTUATOR_CSV_FILE": str(signal_path.resolve()),
            "KRATOS_FSI_ACTUATOR_CSV_TIME_COLUMN": "time",
            "KRATOS_FSI_ACTUATOR_CSV_VALUE_COLUMN": "value",
        })

    completed = subprocess.run(
        [sys.executable, "MainKratos.py"],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path = campaign_directory / "logs" / f"{case['label']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout)

    if completed.returncode != 0:
        raise RuntimeError(f"Run failed for {case['label']}. See {log_path}.")

    metadata = {
        "case": case,
        "input_path": str(input_path.resolve()),
        "signal_path": str(signal_path.resolve()) if signal_path is not None else "",
        "identification_snapshots_path": str((run_directory / "identification_snapshots.csv").resolve()),
        "return_code": completed.returncode,
        "log_path": str(log_path.resolve()),
    }
    shutil.copyfile(input_path, run_directory / "input_timeseries.csv")
    write_identification_snapshots_csv(
        run_directory / "beam_displacement_timeseries.csv",
        input_path,
        run_directory / "identification_snapshots.csv",
    )
    (run_directory / "case_metadata.json").write_text(json.dumps(metadata, indent=4))

    return run_directory


def write_input_timeseries_csv(path, case, end_time, dt):
    path.parent.mkdir(parents=True, exist_ok=True)
    theta_unwrapped = case.get("phase", 0.0)
    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "time",
            "value",
            "amplitude",
            "frequency_hz",
            "omega_rad_s",
            "theta_rad",
            "theta_unwrapped_rad",
        ])
        number_of_steps = int(math.ceil(end_time / dt))
        previous_time = 0.0
        for step in range(number_of_steps + 1):
            time = min(step * dt, end_time)
            ratio = time / end_time if end_time > 0.0 else 0.0
            amplitude = evaluate_amplitude(case, ratio)
            frequency = evaluate_frequency(case, ratio)
            omega = TWO_PI * frequency
            if step > 0:
                theta_unwrapped += omega * (time - previous_time)
            theta = theta_unwrapped % TWO_PI
            value = amplitude * math.cos(theta)
            writer.writerow([
                f"{time:.12g}",
                f"{value:.12g}",
                f"{amplitude:.12g}",
                f"{frequency:.12g}",
                f"{omega:.12g}",
                f"{theta:.12g}",
                f"{theta_unwrapped:.12g}",
            ])
            previous_time = time


def write_identification_snapshots_csv(beam_path, input_path, output_path):
    input_rows = read_input_rows_by_time(input_path)
    input_rows_by_time = {
        round(row["time_float"], 12): row
        for row in input_rows
    }

    with beam_path.open(newline="") as beam_file:
        beam_reader = csv.reader(beam_file)
        metadata = next(beam_reader)
        beam_header = next(beam_reader)
        beam_rows = list(beam_reader)

    measurement_columns = beam_header[1:]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "time",
            *[f"measurement_{name}" for name in measurement_columns],
            "delta_amplitude",
            "delta_omega_rad_s",
            "theta_rad",
            "theta_unwrapped_rad",
            "control_value",
        ])

        for beam_row in beam_rows:
            time = float(beam_row[0])
            input_row = find_input_row(input_rows, input_rows_by_time, time)
            writer.writerow([
                f"{time:.12g}",
                *beam_row[1:],
                input_row["amplitude"],
                input_row["omega_rad_s"],
                input_row["theta_rad"],
                input_row["theta_unwrapped_rad"],
                input_row["value"],
            ])

    metadata_path = output_path.with_suffix(".metadata.csv")
    with metadata_path.open("w", newline="") as metadata_file:
        writer = csv.writer(metadata_file)
        writer.writerow(metadata)


def read_input_rows_by_time(input_path):
    rows = []
    with input_path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            row["time_float"] = float(row["time"])
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No input rows found in {input_path}.")
    return rows


def find_input_row(input_rows, input_rows_by_time, time):
    rounded_time = round(time, 12)
    if rounded_time in input_rows_by_time:
        return input_rows_by_time[rounded_time]

    nearest = min(input_rows, key=lambda row: abs(row["time_float"] - time))
    if abs(nearest["time_float"] - time) > 1e-9:
        raise RuntimeError(
            f"No input sample near measurement time {time:.12g}; "
            f"nearest is {nearest['time_float']:.12g}."
        )
    return nearest


def evaluate_amplitude(case, ratio):
    if "amplitude" in case:
        return case["amplitude"]
    return case["amplitude_start"] + ratio * (case["amplitude_end"] - case["amplitude_start"])


def evaluate_frequency(case, ratio):
    if "frequency" in case:
        return case["frequency"]
    return case["frequency_start"] + ratio * (case["frequency_end"] - case["frequency_start"])


def collect_metrics(run_directory):
    beam_path = run_directory / "beam_displacement_timeseries.csv"
    actuator_path = run_directory / "actuator_timeseries.csv"
    metrics = {
        "last_time": 0.0,
        "max_abs_tip_y": 0.0,
        "rms_tip_y": 0.0,
        "final_tip_y": 0.0,
        "max_abs_control": 0.0,
        "rms_control": 0.0,
        "number_of_samples": 0,
    }

    tip_square_sum = 0.0
    with beam_path.open(newline="") as beam_file:
        reader = csv.reader(beam_file)
        next(reader)
        header = next(reader)
        tip_y_index = header.index("tip_DISPLACEMENT_Y")
        for row in reader:
            tip_y = float(row[tip_y_index])
            metrics["last_time"] = float(row[0])
            metrics["final_tip_y"] = tip_y
            metrics["max_abs_tip_y"] = max(metrics["max_abs_tip_y"], abs(tip_y))
            tip_square_sum += tip_y * tip_y
            metrics["number_of_samples"] += 1

    if metrics["number_of_samples"]:
        metrics["rms_tip_y"] = math.sqrt(tip_square_sum / metrics["number_of_samples"])

    control_square_sum = 0.0
    control_samples = 0
    with actuator_path.open(newline="") as actuator_file:
        reader = csv.DictReader(actuator_file)
        for row in reader:
            if not row["actuator_name"].endswith("_upper"):
                continue
            control = float(row["control_value"])
            metrics["max_abs_control"] = max(metrics["max_abs_control"], abs(control))
            control_square_sum += control * control
            control_samples += 1

    if control_samples:
        metrics["rms_control"] = math.sqrt(control_square_sum / control_samples)

    return metrics


def write_case_result(campaign_directory, result):
    results_directory = campaign_directory / "case_results"
    results_directory.mkdir(parents=True, exist_ok=True)
    result_path = results_directory / f"{result['label']}.json"
    result_path.write_text(json.dumps(result, indent=4))


def append_result(path, result):
    fieldnames = [
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
    exists = path.exists()
    with path.open("a", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(result)


def format_result(result):
    return (
        f"{result['label']}: "
        f"max|tip_y|={result['max_abs_tip_y']:.6g}, "
        f"rms_tip_y={result['rms_tip_y']:.6g}, "
        f"max|u|={result['max_abs_control']:.6g}"
    )


def read_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


def read_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no", "off")


def read_optional_int(name):
    value = os.environ.get(name)
    return None if value is None else int(value)


if __name__ == "__main__":
    main()
