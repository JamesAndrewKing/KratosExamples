#!/usr/bin/env python3
"""Run one fixed-deadline FSI2 capture MPC trajectory and audit it."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

from fsi2_mpc_handoff_controller import validate_artifact_schema, validate_export


DEFAULT_LABEL = "fsi2_equilibrium_capture_mpc_t40"
CASE_LABEL = "equilibrium_capture_mpc"
END_TIME = 40.0
OUTPUT_DT = 0.01
KICK_VALUE = 0.4
KICK_END_TIME = 2.0
ACTIVATION_TIME = 8.0
PRIOR_BEST_RMS = 0.0142754357
WINDOWS = [(6, 8), (8, 12), (12, 16), (16, 20), (20, 30), (30, 40)]


def main():
    arguments = parse_arguments()
    os.chdir(Path(__file__).resolve().parent)
    if arguments.validate_artifact:
        report_artifact(arguments.validate_artifact)
        return
    if arguments.audit:
        audit_experiment(arguments.audit)
        return

    artifact_path = required_path("KRATOS_FSI_HANDOFF_CONTROLLER_FILE")
    artifact = load_artifact(artifact_path)
    campaign = Path("run_outputs") / os.environ.get(
        "KRATOS_FSI_CAPTURE_LABEL", DEFAULT_LABEL
    )
    if campaign.exists():
        raise FileExistsError(f"Refusing to overwrite existing campaign: {campaign}")
    campaign.mkdir(parents=True)

    horizon = float(artifact["mpc"]["prediction_horizon"])
    metadata = {
        "label": campaign.name,
        "case_label": CASE_LABEL,
        "end_time": END_TIME,
        "output_dt": OUTPUT_DT,
        "kick_value": KICK_VALUE,
        "kick_end_time": KICK_END_TIME,
        "activation_time": ACTIVATION_TIME,
        "capture_horizon": horizon,
        "capture_deadline": ACTIVATION_TIME + horizon,
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "write_paraview": read_bool("KRATOS_FSI_CAPTURE_WRITE_PARAVIEW", False),
    }
    write_json(campaign / "experiment_metadata.json", metadata)
    run_case(campaign, artifact_path, metadata)
    audit_experiment(campaign)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-artifact", type=Path)
    parser.add_argument("--audit", type=Path)
    return parser.parse_args()


def required_path(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required.")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_artifact(path):
    data = json.loads(Path(path).read_text())
    validate_artifact_schema(data)
    errors = validate_export(data)
    if max(errors.values()) > 1e-7:
        raise RuntimeError(f"MATLAB/Python artifact mismatch: {errors}")
    if abs(float(data["mpc"]["prediction_horizon"]) - 8.0) > 1e-12:
        raise RuntimeError("Capture experiment requires an eight-second horizon.")
    if not data["handoff"]["force_zero_terminal_amplitude"]:
        raise RuntimeError("Capture artifact does not enforce zero terminal amplitude.")
    return data


def report_artifact(path):
    data = load_artifact(path)
    errors = validate_export(data)
    print(f"artifact={Path(path).resolve()}")
    print("reference_errors=" + ",".join(
        f"{name}:{value:.3e}" for name, value in errors.items()
    ))
    print(
        f"capture_horizon={data['mpc']['prediction_horizon']}s, "
        f"move_block={data['mpc']['move_block_duration']}s, "
        f"terminal_A_zero={data['handoff']['force_zero_terminal_amplitude']}"
    )


def run_case(campaign, artifact_path, metadata):
    run_directory = campaign / "runs" / CASE_LABEL
    log_path = campaign / "logs" / f"{CASE_LABEL}.log"
    run_directory.mkdir(parents=True)
    log_path.parent.mkdir(parents=True)
    artifact_copy = run_directory / "fsi2_mpc_handoff_controller.json"
    shutil.copyfile(artifact_path, artifact_copy)

    environment = os.environ.copy()
    environment.update({
        "KRATOS_FSI_RUN_LABEL": CASE_LABEL,
        "KRATOS_FSI_RUN_OUTPUT_DIRECTORY": str(run_directory.resolve()),
        "KRATOS_FSI_END_TIME": str(END_TIME),
        "KRATOS_FSI_OUTPUT_INTERVAL": str(OUTPUT_DT),
        "KRATOS_FSI_WRITE_PARAVIEW": "1" if metadata["write_paraview"] else "0",
        "KRATOS_FSI_CONTROLLER_TYPE": "equilibrium_capture_mpc",
        "KRATOS_FSI_HANDOFF_CONTROLLER_FILE": str(artifact_copy.resolve()),
        "KRATOS_FSI_MPC_ACTIVATION_TIME": str(ACTIVATION_TIME),
        "KRATOS_FSI_MPC_INITIAL_KICK_VALUE": str(KICK_VALUE),
        "KRATOS_FSI_MPC_INITIAL_KICK_END_TIME": str(KICK_END_TIME),
    })
    completed = subprocess.run(
        [sys.executable, "MainKratos.py"], env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    log_path.write_text(completed.stdout)
    if completed.returncode:
        raise RuntimeError(f"Run failed; see {log_path}.")
    write_json(run_directory / "case_metadata.json", {
        **metadata,
        "controller_type": "equilibrium_capture_mpc",
        "artifact_copy": str(artifact_copy),
    })
    print(f"completed={run_directory.resolve()}")


def audit_experiment(campaign):
    campaign = Path(campaign).resolve()
    metadata = json.loads((campaign / "experiment_metadata.json").read_text())
    deadline = float(metadata["capture_deadline"])
    run_directory = campaign / "runs" / CASE_LABEL
    beam_path = run_directory / "beam_displacement_timeseries.csv"
    actuator_path = run_directory / "actuator_timeseries.csv"
    controller_path = run_directory / "mpc_handoff_timeseries.csv"
    missing = [path.name for path in (beam_path, actuator_path, controller_path)
               if not path.is_file()]
    if missing:
        raise RuntimeError(f"Controlled run is missing files: {missing}")

    time_values, tip = read_tip(beam_path)
    validate_grid(time_values, "controlled run")
    actuator = read_actuator(actuator_path)
    balance_error = actuator_balance_error(actuator)
    controller = read_controller(controller_path)
    release_rows = [row for row in controller if row["event"] == "capture_to_coast"]
    if len(release_rows) != 1:
        raise RuntimeError(
            f"Expected one capture_to_coast event, found {len(release_rows)}."
        )
    release = release_rows[0]
    release_time = float(release["time"])
    if abs(release_time - deadline) > 1e-8:
        raise RuntimeError(
            f"Capture released at {release_time}, expected {deadline}."
        )
    if controller[-1]["mode"] != "coast":
        raise RuntimeError("Controller did not remain in coast mode.")

    best_rms, best_time, _ = minimum_rolling_rms(
        time_values, tip, ACTIVATION_TIME, min(END_TIME, deadline + 4.0), 1.0
    )
    window_report = {}
    for start, stop in WINDOWS:
        name = f"{start:g}_{stop:g}"
        indices = [i for i, value in enumerate(time_values)
                   if start <= value < stop]
        window_report[name] = {
            "rms": rms([tip[i] for i in indices]),
            "peak_to_peak": (
                max(tip[i] for i in indices) - min(tip[i] for i in indices)
            ),
        }

    local_rows = [row for row in controller
                  if math.isfinite(float(row["local_radius"]))]
    minimum_local = min(local_rows, key=lambda row: float(row["local_radius"]))
    post_release_control = [
        abs(float(row["control_value"])) for row in actuator
        if float(row["time"]) >= deadline - 1e-10
    ]
    upper = [row for row in actuator if row["actuator_name"].endswith("_upper")]
    kick_error = max((
        abs(float(row["control_value"]) - KICK_VALUE)
        for row in upper if float(row["time"]) < KICK_END_TIME - 1e-10
    ), default=math.inf)
    pre_activation_control = max((
        abs(float(row["control_value"]))
        for row in upper
        if KICK_END_TIME - 1e-10 <= float(row["time"]) < ACTIVATION_TIME - 1e-10
    ), default=math.inf)
    summary = {
        "integrity_success": (
            balance_error < 1e-10
            and kick_error < 1e-10
            and pre_activation_control < 1e-10
            and max(post_release_control, default=0.0) < 1e-10
            and abs(float(release["A"])) < 1e-10
        ),
        "replicated_prior_transient": best_rms <= PRIOR_BEST_RMS,
        "prior_best_one_second_rms": PRIOR_BEST_RMS,
        "best_one_second_rms": best_rms,
        "best_one_second_end_time": best_time,
        "release_time": release_time,
        "release_local_radius": float(release["local_radius"]),
        "release_amplitude": float(release["A"]),
        "minimum_local_radius": float(minimum_local["local_radius"]),
        "minimum_local_radius_time": float(minimum_local["time"]),
        "maximum_post_release_control": max(post_release_control, default=0.0),
        "maximum_actuator_balance_error": balance_error,
        "maximum_kick_command_error": kick_error,
        "maximum_pre_activation_control": pre_activation_control,
        "windows": window_report,
        "run_directory": str(run_directory),
    }
    write_json(campaign / "summary.json", summary)
    print_summary(summary)
    if not summary["integrity_success"]:
        raise RuntimeError("Capture experiment failed its integrity checks.")

def read_tip(path):
    with Path(path).open(newline="") as input_file:
        reader = csv.reader(input_file)
        next(reader)
        names = next(reader)
        tip_index = names.index("tip_DISPLACEMENT_Y")
        rows = list(reader)
    return ([float(row[0]) for row in rows],
            [float(row[tip_index]) for row in rows])


def read_actuator(path):
    with Path(path).open(newline="") as input_file:
        return list(csv.DictReader(input_file))


def read_controller(path):
    with Path(path).open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    required = {"time", "mode", "event", "local_radius", "A", "control_u"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise RuntimeError(f"Controller log is missing columns: {sorted(missing)}")
    return rows


def validate_grid(time_values, label):
    if len(time_values) != round(END_TIME / OUTPUT_DT) + 1:
        raise RuntimeError(f"{label} has the wrong number of samples.")
    if abs(time_values[0]) > 1e-12 or abs(time_values[-1] - END_TIME) > 1e-10:
        raise RuntimeError(f"{label} has the wrong time interval.")
    if max(abs(time_values[i] - i * OUTPUT_DT)
           for i in range(len(time_values))) > 1e-8:
        raise RuntimeError(f"{label} time grid is irregular.")


def actuator_balance_error(rows):
    totals = {}
    for row in rows:
        time_value = float(row["time"])
        totals[time_value] = totals.get(time_value, 0.0) + float(row["control_value"])
    return max((abs(value) for value in totals.values()), default=0.0)


def minimum_rolling_rms(time_values, values, start, stop, width):
    window = deque()
    squared_sum = 0.0
    best = math.inf
    best_time = math.nan
    best_indices = (0, 0)
    for index, (time_value, value) in enumerate(zip(time_values, values)):
        window.append((index, time_value, value * value))
        squared_sum += value * value
        while window and window[0][1] < time_value - width - 1e-12:
            squared_sum -= window.popleft()[2]
        if start + width <= time_value <= stop + 1e-12 and window:
            current = math.sqrt(squared_sum / len(window))
            if current < best:
                best = current
                best_time = time_value
                best_indices = (window[0][0], index + 1)
    return best, best_time, best_indices


def rms(values):
    return math.sqrt(sum(value * value for value in values) / len(values))


def print_summary(summary):
    print("\nEquilibrium-capture MPC verdict")
    print(f"  integrity passed       : {summary['integrity_success']}")
    print(f"  prior transient matched: {summary['replicated_prior_transient']}")
    print(
        f"  best rolling 1 s RMS   : {summary['best_one_second_rms']:.6g} "
        f"at t={summary['best_one_second_end_time']:.3f} s"
    )
    print(
        f"  prior benchmark        : {summary['prior_best_one_second_rms']:.6g}"
    )
    print(
        f"  local radius at release: {summary['release_local_radius']:.6g}"
    )
    print(
        f"  minimum local radius   : {summary['minimum_local_radius']:.6g} "
        f"at t={summary['minimum_local_radius_time']:.3f} s"
    )
    print(
        f"  max control after release: "
        f"{summary['maximum_post_release_control']:.3e}"
    )


def read_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
