#!/usr/bin/env python3
"""Run and audit the two-case FSI2 amplitude-frequency MPC experiment."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

from fsi2_fourier_envelope_mpc_controller import (
    PARAMETER_COORDINATES,
    validate_artifact_schema,
    validate_control_contract,
    validate_export,
)


DEFAULT_LABEL = "fsi2_amplitude_frequency_mpc_t60"
END_TIME = 60.0
OUTPUT_DT = 0.01
KICK_VALUE = 0.4
KICK_END_TIME = 2.0
ACTIVATION_TIME = 20.0
WINDOWS = [(15, 20), (20, 25), (25, 30), (30, 40), (40, 50), (50, 60)]
CASES = [
    {"index": 0, "label": "passive_baseline", "controller_type": "csv"},
    {
        "index": 1,
        "label": "amplitude_frequency_mpc",
        "controller_type": "fourier_envelope_mpc",
    },
]
EXPECTED_ARTIFACT_SETTINGS = {
    "control_interval": 0.2,
    "internal_step": 0.05,
    "move_block_duration": 0.4,
    "prediction_horizon": 3.2,
    "parameter_lower": [0.0, 2.0 * math.pi * 1.7],
    "parameter_upper": [2.0, 2.0 * math.pi * 2.1],
    "parameter_rate_bound": [0.75, 2.0 * math.pi * 0.1],
    "parameter_weights": [1e-3, 5e-3],
    "rate_weights": [2e-3, 5e-3],
    "terminal_weight": 8.0,
}
OBSOLETE_ENVIRONMENT_VARIABLES = [
    "KRATOS_FSI_MPC_CONTROL_INTERVAL",
    "KRATOS_FSI_MPC_PREDICTION_HORIZON",
    "KRATOS_FSI_MPC_CONTROL_BOUND",
    "KRATOS_FSI_MPC_MAX_CONTROL_INCREMENT",
    "KRATOS_FSI_MPC_MOVE_BLOCKS",
    "KRATOS_FSI_MPC_OPTIMIZER_ITERATIONS",
]
OBSOLETE_PARAMETER_KEYS = {
    "mpc_control_interval",
    "mpc_prediction_horizon",
    "mpc_control_bound",
    "mpc_max_control_increment",
    "mpc_move_blocks",
    "mpc_optimizer_iterations",
}
REQUIRED_MPC_COLUMNS = [
    "time", "eta_1", "eta_2", "theta", "A", "Omega", "frequency_hz",
    "A_dot", "Omega_dot", "frequency_dot_hz_s", "control_u", "objective",
    "solve_time_seconds", "optimizer_iterations", "output_error",
]


def main():
    args = parse_arguments()
    source_directory = Path(__file__).resolve().parent
    os.chdir(source_directory)

    if args.list_cases:
        for case in CASES:
            print(f"{case['index']:03d} {case['label']} [{case['controller_type']}]")
        print(f"number_of_cases={len(CASES)}")
        return

    if args.validate_artifact:
        data = load_and_validate_artifact(args.validate_artifact)
        print_artifact_report(args.validate_artifact, data)
        return

    if args.dry_run:
        settings = read_settings()
        artifact = load_and_validate_artifact(settings["artifact_path"])
        run_dry_experiment(args.dry_run, settings["artifact_path"], artifact)
        audit_experiment(args.dry_run)
        return

    if args.audit:
        audit_experiment(args.audit)
        return

    settings = read_settings()
    artifact = load_and_validate_artifact(settings["artifact_path"])
    print_artifact_report(settings["artifact_path"], artifact)
    campaign_directory = Path("run_outputs") / settings["campaign_label"]
    campaign_directory.mkdir(parents=True, exist_ok=True)

    selected = CASES if args.case_index is None else [get_case(args.case_index)]
    for case in selected:
        print(f"[{case['index']}] {case['label']}", flush=True)
        run_directory = run_case(case, campaign_directory, settings)
        result = {
            **case,
            "status": "completed",
            "end_time": END_TIME,
            "run_directory": str(run_directory.resolve()),
            "beam_displacement_timeseries": str(
                (run_directory / "beam_displacement_timeseries.csv").resolve()),
            "actuator_timeseries": str(
                (run_directory / "actuator_timeseries.csv").resolve()),
            "rom_mpc_timeseries": (
                str((run_directory / "rom_mpc_timeseries.csv").resolve())
                if case["controller_type"] == "fourier_envelope_mpc" else None
            ),
        }
        write_json_atomic(
            campaign_directory / "case_results" / f"{case['label']}.json", result)
        print(f"completed={run_directory.resolve()}", flush=True)

    if args.case_index is None:
        audit_experiment(campaign_directory)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run or audit passive and amplitude-frequency MPC FSI2 cases."
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--case-index", type=int, choices=range(len(CASES)))
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--dry-run", type=Path)
    parser.add_argument("--validate-artifact", type=Path)
    return parser.parse_args()


def read_settings():
    artifact_name = os.environ.get("KRATOS_FSI_ROM_FILE")
    if not artifact_name:
        raise RuntimeError("KRATOS_FSI_ROM_FILE must point to the format-v4 artifact.")
    return {
        "campaign_label": os.environ.get("KRATOS_FSI_AF_MPC_LABEL", DEFAULT_LABEL),
        "artifact_path": Path(artifact_name).resolve(),
        "write_paraview": read_boolean_environment_variable(
            "KRATOS_FSI_AF_MPC_WRITE_PARAVIEW", False),
    }


def read_boolean_environment_variable(name, default_value):
    value = os.environ.get(name)
    if value is None:
        return default_value
    return value.lower() not in ("0", "false", "no", "off")


def get_case(index):
    for case in CASES:
        if case["index"] == index:
            return case
    raise ValueError(f"Unknown case index {index}.")


def load_and_validate_artifact(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Controller artifact not found: {path}")
    data = json.loads(path.read_text())
    validate_artifact_schema(data)
    if data.get("parameter_coordinates") != PARAMETER_COORDINATES:
        raise RuntimeError("Artifact is not amplitude-frequency parameterized.")

    for name, expected in EXPECTED_ARTIFACT_SETTINGS.items():
        actual = data.get(name)
        if isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                raise RuntimeError(f"Artifact setting {name} has invalid shape: {actual!r}")
            differences = [abs(float(actual[i]) - expected[i]) for i in range(len(expected))]
            if max(differences, default=0.0) > 1e-11:
                raise RuntimeError(
                    f"Artifact setting {name}={actual!r}, expected {expected!r}."
                )
        elif abs(float(actual) - expected) > 1e-11:
            raise RuntimeError(f"Artifact setting {name}={actual}, expected {expected}.")

    export_errors = validate_export(data)
    if max(export_errors) > 1e-7:
        raise RuntimeError(f"MATLAB/Python reference mismatch: {export_errors}")
    validate_control_contract(data)
    return data


def print_artifact_report(path, data):
    errors = validate_export(data)
    contract = validate_control_contract(data)
    print(f"artifact={Path(path).resolve()}")
    print(f"format_version={data['format_version']}")
    print(f"parameter_coordinates={data['parameter_coordinates']}")
    print(
        "reference_errors="
        f"dynamics:{errors[0]:.3e},output:{errors[1]:.3e},jacobian:{errors[2]:.3e}"
    )
    print(
        "control_contract="
        f"phase:{contract['phase_error']:.3e},"
        f"parameters:{contract['parameter_error']:.3e},"
        f"u:{contract['control_identity_error']:.3e},"
        f"rate_fraction:{contract['max_projected_rate_fraction']:.6f}"
    )
    print(
        "settings="
        f"dt_control:{data['control_interval']},dt_internal:{data['internal_step']},"
        f"block:{data['move_block_duration']},horizon:{data['prediction_horizon']}"
    )


def run_case(case, campaign_directory, settings):
    run_directory = campaign_directory / "runs" / case["label"]
    if run_directory.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_directory}")
    run_directory.mkdir(parents=True)
    log_path = campaign_directory / "logs" / f"{case['label']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing log: {log_path}")

    write_paraview = (
        settings["write_paraview"]
        and case["controller_type"] == "fourier_envelope_mpc"
    )
    environment = os.environ.copy()
    for name in OBSOLETE_ENVIRONMENT_VARIABLES:
        environment.pop(name, None)
    environment.update({
        "KRATOS_FSI_RUN_LABEL": case["label"],
        "KRATOS_FSI_RUN_OUTPUT_DIRECTORY": str(run_directory.resolve()),
        "KRATOS_FSI_END_TIME": str(END_TIME),
        "KRATOS_FSI_OUTPUT_INTERVAL": str(OUTPUT_DT),
        "KRATOS_FSI_WRITE_PARAVIEW": "1" if write_paraview else "0",
        "KRATOS_FSI_ACTUATOR_AMPLITUDE": "0.0",
        "KRATOS_FSI_ACTUATOR_FREQUENCY": "0.0",
        "KRATOS_FSI_ACTUATOR_PHASE": "0.0",
    })

    if case["controller_type"] == "csv":
        signal_path = campaign_directory / "inputs" / "passive_baseline.csv"
        write_passive_kick_signal(signal_path)
        shutil.copyfile(signal_path, run_directory / "input_timeseries.csv")
        environment.update({
            "KRATOS_FSI_CONTROLLER_TYPE": "csv",
            "KRATOS_FSI_ACTUATOR_CSV_FILE": str(signal_path.resolve()),
            "KRATOS_FSI_ACTUATOR_CSV_TIME_COLUMN": "time",
            "KRATOS_FSI_ACTUATOR_CSV_VALUE_COLUMN": "value",
            "KRATOS_FSI_ACTUATOR_CSV_INTERPOLATION": "zoh",
        })
    else:
        artifact_copy = run_directory / "fsi2_fourier_envelope_controller.json"
        shutil.copyfile(settings["artifact_path"], artifact_copy)
        (run_directory / "controller_artifact.sha256").write_text(
            sha256_file(artifact_copy) + "  " + artifact_copy.name + "\n"
        )
        environment.update({
            "KRATOS_FSI_CONTROLLER_TYPE": "fourier_envelope_mpc",
            "KRATOS_FSI_ROM_FILE": str(settings["artifact_path"]),
            "KRATOS_FSI_MPC_ACTIVATION_TIME": str(ACTIVATION_TIME),
            "KRATOS_FSI_MPC_INITIAL_KICK_VALUE": str(KICK_VALUE),
            "KRATOS_FSI_MPC_INITIAL_KICK_END_TIME": str(KICK_END_TIME),
        })

    completed = subprocess.run(
        [sys.executable, "MainKratos.py"],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"Run failed for {case['label']}. See {log_path}.")

    metadata = {
        "label": case["label"],
        "controller_type": case["controller_type"],
        "end_time": END_TIME,
        "output_dt": OUTPUT_DT,
        "initial_kick_value": KICK_VALUE,
        "initial_kick_end_time": KICK_END_TIME,
        "mpc_activation_time": ACTIVATION_TIME,
        "paraview": write_paraview,
        "artifact_sha256": sha256_file(settings["artifact_path"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json_atomic(run_directory / "case_metadata.json", metadata)
    return run_directory


def write_passive_kick_signal(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        rows = list(csv.DictReader(path.open(newline="")))
        expected = [(0.0, KICK_VALUE), (KICK_END_TIME, 0.0), (END_TIME, 0.0)]
        actual = [(float(row["time"]), float(row["value"])) for row in rows]
        if actual != expected:
            raise RuntimeError(f"Existing passive input differs from expected: {path}")
        return
    with path.open("x", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["time", "value"])
        writer.writerows([
            ["0", f"{KICK_VALUE:.12g}"],
            [f"{KICK_END_TIME:.12g}", "0"],
            [f"{END_TIME:.12g}", "0"],
        ])


def run_dry_experiment(campaign_directory, artifact_path, artifact):
    campaign_directory = Path(campaign_directory)
    if campaign_directory.exists():
        raise FileExistsError(f"Refusing to overwrite dry-run directory: {campaign_directory}")
    omega = artifact["parameter_reference"][1]
    for case in CASES:
        run_directory = campaign_directory / "runs" / case["label"]
        run_directory.mkdir(parents=True)
        write_synthetic_beam(run_directory / "beam_displacement_timeseries.csv", case)
        write_synthetic_actuator(
            run_directory / "actuator_timeseries.csv", case, omega)
        (run_directory / "ProjectParameters.effective.json").write_text(
            json.dumps({"controller_type": case["controller_type"]}, indent=2) + "\n"
        )
        write_json_atomic(run_directory / "case_metadata.json", {
            "label": case["label"],
            "controller_type": case["controller_type"],
            "end_time": END_TIME,
            "dry_run": True,
        })
        if case["controller_type"] == "fourier_envelope_mpc":
            shutil.copyfile(artifact_path, run_directory / artifact_path.name)
            write_synthetic_mpc_log(run_directory / "rom_mpc_timeseries.csv", artifact)


def write_synthetic_beam(path, case):
    with path.open("x", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["time"])
        header = ["time"]
        for name in ["x_0_30", "x_0_40", "x_0_50", "tip"]:
            header.extend([
                f"{name}_DISPLACEMENT_X",
                f"{name}_DISPLACEMENT_Y",
                f"{name}_DISPLACEMENT_Z",
            ])
        writer.writerow(header)
        count = round(END_TIME / OUTPUT_DT)
        for index in range(count + 1):
            time_value = index * OUTPUT_DT
            growth = 1.0 - math.exp(-0.35 * time_value)
            amplitude = 0.08 * growth
            if case["controller_type"] == "fourier_envelope_mpc" and time_value >= ACTIVATION_TIME:
                amplitude *= math.exp(-0.04 * (time_value - ACTIVATION_TIME))
            tip_y = amplitude * math.sin(2.0 * math.pi * 1.9 * time_value)
            values = []
            for factor in [0.3, 0.6, 0.85, 1.0]:
                values.extend(["0", f"{factor * tip_y:.12g}", "0"])
            writer.writerow([f"{time_value:.12g}", *values])


def write_synthetic_actuator(path, case, omega):
    with path.open("x", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "time", "actuator_name", "control_value", "weighted_mean_velocity_x",
            "weighted_mean_velocity_y", "weighted_mean_velocity_z", "number_of_nodes",
        ])
        count = round(END_TIME / OUTPUT_DT)
        for index in range(count + 1):
            time_value = index * OUTPUT_DT
            if time_value < KICK_END_TIME - 1e-10:
                control = KICK_VALUE
            elif case["controller_type"] == "fourier_envelope_mpc" \
                    and time_value >= ACTIVATION_TIME:
                amplitude = min(2.0, 0.05 * (time_value - ACTIVATION_TIME))
                control = amplitude * math.cos(omega * time_value)
            else:
                control = 0.0
            writer.writerow([
                f"{time_value:.12g}", "rabault_pair_upper", f"{control:.12g}",
                "0", "0", "0", "1",
            ])
            writer.writerow([
                f"{time_value:.12g}", "rabault_pair_lower", f"{-control:.12g}",
                "0", "0", "0", "1",
            ])


def write_synthetic_mpc_log(path, artifact):
    omega = artifact["parameter_reference"][1]
    control_dt = artifact["control_interval"]
    with path.open("x", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=REQUIRED_MPC_COLUMNS)
        writer.writeheader()
        count = round((END_TIME - ACTIVATION_TIME) / control_dt)
        for index in range(count + 1):
            time_value = ACTIVATION_TIME + index * control_dt
            amplitude = min(2.0, 0.05 * (time_value - ACTIVATION_TIME))
            amplitude_dot = 0.05 if amplitude < 2.0 - 1e-10 else 0.0
            theta = omega * time_value
            writer.writerow({
                "time": f"{time_value:.12g}",
                "eta_1": "0",
                "eta_2": "0",
                "theta": f"{theta:.12g}",
                "A": f"{amplitude:.12g}",
                "Omega": f"{omega:.12g}",
                "frequency_hz": f"{omega / (2.0 * math.pi):.12g}",
                "A_dot": f"{amplitude_dot:.12g}",
                "Omega_dot": "0",
                "frequency_dot_hz_s": "0",
                "control_u": f"{amplitude * math.cos(theta):.12g}",
                "objective": "0",
                "solve_time_seconds": "0.01",
                "optimizer_iterations": "1",
                "output_error": "0",
            })


def audit_experiment(campaign_directory):
    campaign_directory = Path(campaign_directory)
    artifact_path = campaign_directory / "runs" / "amplitude_frequency_mpc" \
        / "fsi2_fourier_envelope_controller.json"
    artifact = load_and_validate_artifact(artifact_path)
    errors = []
    data = {}
    for case in CASES:
        run_directory = campaign_directory / "runs" / case["label"]
        data[case["label"]] = audit_case(run_directory, case, artifact, errors)

    compare_startups(data, errors)
    if errors:
        print("Experiment audit failed:")
        for error in errors:
            print(f"  ERROR: {error}")
        raise RuntimeError("Amplitude-frequency MPC experiment audit failed.")

    passive = data["passive_baseline"]
    controlled = data["amplitude_frequency_mpc"]
    ratios = {}
    for start, end in WINDOWS:
        name = window_name(start, end)
        denominator = passive["windows"][name]["rms"]
        ratios[name] = (
            controlled["windows"][name]["rms"] / denominator
            if denominator > 1e-15 else None
        )

    summary = {
        "experiment": {
            "label": campaign_directory.name,
            "end_time": END_TIME,
            "initial_kick": {"value": KICK_VALUE, "start": 0.0, "end": KICK_END_TIME},
            "passive_interval": [KICK_END_TIME, ACTIVATION_TIME],
            "controlled_interval": [ACTIVATION_TIME, END_TIME],
            "paraview": {
                label: case_data.get("paraview", False)
                for label, case_data in data.items()
            },
            "regrowth_definition": (
                "first later rolling-1s RMS at least 10% above the minimum and "
                "remaining above that threshold for 1 s"
            ),
        },
        "artifact": {
            "sha256": sha256_file(artifact_path),
            "format_version": artifact["format_version"],
            "parameter_coordinates": artifact["parameter_coordinates"],
            **{name: artifact[name] for name in EXPECTED_ARTIFACT_SETTINGS},
        },
        "cases": data,
        "controlled_passive_rms_ratio": ratios,
        "persistent_suppression": persistent_suppression(ratios),
    }
    write_json_atomic(campaign_directory / "summary.json", summary)
    print("Experiment audit passed: 2/2 cases complete and consistent.")
    print_report(summary)


def audit_case(run_directory, case, artifact, errors):
    required = [
        run_directory / "beam_displacement_timeseries.csv",
        run_directory / "actuator_timeseries.csv",
        run_directory / "ProjectParameters.effective.json",
        run_directory / "case_metadata.json",
    ]
    if case["controller_type"] == "fourier_envelope_mpc":
        required.append(run_directory / "rom_mpc_timeseries.csv")
    for path in required:
        if not path.is_file():
            errors.append(f"{case['label']}: missing {path.name}")
    if any(not path.is_file() for path in required):
        return {"status": "incomplete"}

    times, tip = read_tip_series(run_directory / "beam_displacement_timeseries.csv")
    if abs(times[-1] - END_TIME) > 1e-8:
        errors.append(f"{case['label']}: last beam time is {times[-1]}, expected {END_TIME}")
    if not consistent_step(times, OUTPUT_DT):
        errors.append(f"{case['label']}: beam output step is inconsistent")

    actuator = read_actuator_series(run_directory / "actuator_timeseries.csv")
    max_balance_error = audit_actuator_balance(actuator)
    if max_balance_error > 1e-10:
        errors.append(f"{case['label']}: actuator imbalance {max_balance_error:.3e}")
    effective = json.loads((run_directory / "ProjectParameters.effective.json").read_text())
    metadata = json.loads((run_directory / "case_metadata.json").read_text())
    stale = find_keys(effective, OBSOLETE_PARAMETER_KEYS)
    if stale:
        errors.append(f"{case['label']}: obsolete effective settings {sorted(stale)}")

    windows = {
        window_name(start, end): window_stats(times, tip, start, end)
        for start, end in WINDOWS
    }
    rolling = [item for item in rolling_rms(times, tip, 1.0)
               if item[0] >= ACTIVATION_TIME + 1.0 - 1e-10]
    minimum = min(rolling, key=lambda item: item[1])
    regrowth = find_sustained_regrowth(rolling, minimum, 1.0)
    result = {
        "label": case["label"],
        "run_directory": str(run_directory.resolve()),
        "last_time": times[-1],
        "number_of_samples": len(times),
        "windows": windows,
        "minimum_rolling_1s_tip_rms": minimum[1],
        "minimum_rolling_1s_tip_rms_time": minimum[0],
        "subsequent_regrowth_time": regrowth,
        "time_from_minimum_to_regrowth": (
            regrowth - minimum[0] if regrowth is not None else None
        ),
        "max_actuator_balance_error": max_balance_error,
        "paraview": bool(metadata.get("paraview", False)),
    }

    if result["paraview"]:
        expected_frames = round(END_TIME / OUTPUT_DT) + 1
        result["paraview_frames"] = {}
        for output_name in ("vtk_output_fluid", "vtk_output_structure"):
            frame_count = sum(1 for _ in (run_directory / output_name).rglob("*.vtk"))
            result["paraview_frames"][output_name] = frame_count
            if frame_count != expected_frames:
                errors.append(
                    f"{case['label']}: {output_name} has {frame_count} VTK frames, "
                    f"expected {expected_frames}")

    if case["controller_type"] == "csv":
        if any(abs(value - (KICK_VALUE if time < KICK_END_TIME - 1e-10 else 0.0))
               > 1e-10 for time, value in actuator["upper"]):
            errors.append("passive_baseline: control differs from kick-then-passive schedule")
        if (run_directory / "rom_mpc_timeseries.csv").exists():
            errors.append("passive_baseline: unexpected rom_mpc_timeseries.csv")
    else:
        result["controller"] = audit_mpc_log(
            run_directory / "rom_mpc_timeseries.csv", actuator, artifact, errors)
    return result


def audit_mpc_log(path, actuator, artifact, errors):
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        missing = [name for name in REQUIRED_MPC_COLUMNS if name not in reader.fieldnames]
        rows = list(reader)
    if missing:
        errors.append(f"amplitude_frequency_mpc: missing MPC columns {missing}")
        return {}
    if not rows:
        errors.append("amplitude_frequency_mpc: empty MPC log")
        return {}

    numeric = [{name: float(row[name]) for name in REQUIRED_MPC_COLUMNS
                if name != "optimizer_iterations"} | {
                    "optimizer_iterations": int(row["optimizer_iterations"])
                } for row in rows]
    lower = artifact["parameter_lower"]
    upper = artifact["parameter_upper"]
    rates = artifact["parameter_rate_bound"]
    control_dt = float(artifact["control_interval"])
    if abs(numeric[0]["time"] - ACTIVATION_TIME) > 1e-8 \
            or abs(numeric[-1]["time"] - END_TIME) > 1e-8:
        errors.append("amplitude_frequency_mpc: MPC log does not span [20,60] s")
    if not consistent_step([row["time"] for row in numeric], control_dt):
        errors.append("amplitude_frequency_mpc: MPC update interval is inconsistent")

    upper_by_time = {round(time, 10): value for time, value in actuator["upper"]}
    max_identity_error = 0.0
    max_actuator_error = 0.0
    max_phase_error = 0.0
    for index, row in enumerate(numeric):
        max_identity_error = max(
            max_identity_error,
            abs(row["control_u"] - row["A"] * math.cos(row["theta"])),
        )
        actual = upper_by_time.get(round(row["time"], 10))
        if actual is not None:
            max_actuator_error = max(max_actuator_error, abs(actual - row["control_u"]))
        if not lower[0] - 1e-10 <= row["A"] <= upper[0] + 1e-10:
            errors.append(f"amplitude_frequency_mpc: A bound violation at {row['time']}")
            break
        if not lower[1] - 1e-10 <= row["Omega"] <= upper[1] + 1e-10:
            errors.append(f"amplitude_frequency_mpc: Omega bound violation at {row['time']}")
            break
        if abs(row["A_dot"]) > rates[0] + 1e-10 \
                or abs(row["Omega_dot"]) > rates[1] + 1e-10:
            errors.append(f"amplitude_frequency_mpc: rate violation at {row['time']}")
            break
        if index:
            previous = numeric[index - 1]
            dt = row["time"] - previous["time"]
            expected = (previous["theta"] + previous["Omega"] * dt
                        + 0.5 * previous["Omega_dot"] * dt * dt)
            max_phase_error = max(max_phase_error, abs(row["theta"] - expected))

    # Online CSV fields use 12 significant digits, so identity checks include
    # the corresponding text-rounding error.
    if max_identity_error > 1e-8:
        errors.append(f"amplitude_frequency_mpc: u identity error {max_identity_error:.3e}")
    if max_actuator_error > 1e-10:
        errors.append(f"amplitude_frequency_mpc: controller/actuator error {max_actuator_error:.3e}")
    if max_phase_error > 1e-8:
        errors.append(f"amplitude_frequency_mpc: phase propagation error {max_phase_error:.3e}")

    tolerance = 1e-7
    fractions = {
        "A_lower": fraction_at(numeric, "A", lower[0], tolerance),
        "A_upper": fraction_at(numeric, "A", upper[0], tolerance),
        "frequency_lower": fraction_at(
            numeric, "frequency_hz", lower[1] / (2.0 * math.pi), tolerance),
        "frequency_upper": fraction_at(
            numeric, "frequency_hz", upper[1] / (2.0 * math.pi), tolerance),
        "A_dot_negative": fraction_at(numeric, "A_dot", -rates[0], tolerance),
        "A_dot_positive": fraction_at(numeric, "A_dot", rates[0], tolerance),
        "frequency_dot_negative": fraction_at(
            numeric, "frequency_dot_hz_s", -rates[1] / (2.0 * math.pi), tolerance),
        "frequency_dot_positive": fraction_at(
            numeric, "frequency_dot_hz_s", rates[1] / (2.0 * math.pi), tolerance),
    }
    solve_times = [row["solve_time_seconds"] for row in numeric]
    return {
        "number_of_updates": len(numeric),
        "maximum_A": max(row["A"] for row in numeric),
        "final_A": numeric[-1]["A"],
        "frequency_hz_range": [
            min(row["frequency_hz"] for row in numeric),
            max(row["frequency_hz"] for row in numeric),
        ],
        "maximum_abs_A_dot": max(abs(row["A_dot"]) for row in numeric),
        "maximum_abs_Omega_dot": max(abs(row["Omega_dot"]) for row in numeric),
        "maximum_abs_frequency_dot_hz_s": max(
            abs(row["frequency_dot_hz_s"]) for row in numeric),
        "fraction_of_updates_at_bounds": fractions,
        "median_solve_time_seconds": statistics.median(solve_times),
        "maximum_solve_time_seconds": max(solve_times),
        "maximum_control_identity_error": max_identity_error,
        "maximum_phase_propagation_error": max_phase_error,
    }


def compare_startups(data, errors):
    passive_path = Path(data["passive_baseline"]["run_directory"]) / "actuator_timeseries.csv"
    controlled_path = Path(data["amplitude_frequency_mpc"]["run_directory"]) \
        / "actuator_timeseries.csv"
    passive = dict(read_actuator_series(passive_path)["upper"])
    controlled = dict(read_actuator_series(controlled_path)["upper"])
    common = [time for time in passive if time <= ACTIVATION_TIME + 1e-10 and time in controlled]
    difference = max((abs(passive[time] - controlled[time]) for time in common), default=0.0)
    if not common or difference > 1e-10:
        errors.append(f"startup actuator histories differ by {difference:.3e}")


def read_tip_series(path):
    with path.open(newline="") as input_file:
        reader = csv.reader(input_file)
        next(reader)
        header = next(reader)
        tip_index = header.index("tip_DISPLACEMENT_Y")
        rows = list(reader)
    return ([float(row[0]) for row in rows], [float(row[tip_index]) for row in rows])


def read_actuator_series(path):
    result = {"upper": [], "lower": []}
    with path.open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            key = "upper" if row["actuator_name"].endswith("_upper") else "lower"
            result[key].append((float(row["time"]), float(row["control_value"])))
    return result


def audit_actuator_balance(actuator):
    lower = dict(actuator["lower"])
    return max((abs(value + lower.get(time, math.inf))
                for time, value in actuator["upper"]), default=math.inf)


def window_stats(times, values, start, end):
    selected = [value for time, value in zip(times, values)
                if start - 1e-10 <= time <= end + 1e-10]
    if not selected:
        return {"rms": None, "peak_to_peak": None, "samples": 0}
    return {
        "rms": math.sqrt(sum(value * value for value in selected) / len(selected)),
        "peak_to_peak": max(selected) - min(selected),
        "samples": len(selected),
    }


def rolling_rms(times, values, duration):
    window = deque()
    square_sum = 0.0
    result = []
    for time_value, value in zip(times, values):
        window.append((time_value, value))
        square_sum += value * value
        while window and window[0][0] < time_value - duration - 1e-10:
            _, removed = window.popleft()
            square_sum -= removed * removed
        if window and time_value - window[0][0] >= duration - OUTPUT_DT - 1e-10:
            result.append((time_value, math.sqrt(max(0.0, square_sum) / len(window))))
    return result


def find_sustained_regrowth(rolling, minimum, sustain_duration):
    minimum_index = rolling.index(minimum)
    threshold = 1.1 * minimum[1]
    for index in range(minimum_index + 1, len(rolling)):
        start_time = rolling[index][0]
        sustained = [value for time, value in rolling[index:]
                     if time <= start_time + sustain_duration + 1e-10]
        if sustained and rolling[-1][0] >= start_time + sustain_duration - OUTPUT_DT \
                and min(sustained) >= threshold:
            return start_time
    return None


def persistent_suppression(ratios):
    late = [ratios[window_name(*window)] for window in WINDOWS[-3:]]
    return all(value is not None and value < 1.0 for value in late)


def print_report(summary):
    print("Window report (RMS, peak-to-peak, controlled/passive RMS):")
    for start, end in WINDOWS:
        name = window_name(start, end)
        passive = summary["cases"]["passive_baseline"]["windows"][name]
        controlled = summary["cases"]["amplitude_frequency_mpc"]["windows"][name]
        ratio = summary["controlled_passive_rms_ratio"][name]
        print(
            f"  [{start},{end}] passive=({passive['rms']:.6g}, {passive['peak_to_peak']:.6g}) "
            f"controlled=({controlled['rms']:.6g}, {controlled['peak_to_peak']:.6g}) "
            f"ratio={ratio:.6g}"
        )
    controlled = summary["cases"]["amplitude_frequency_mpc"]
    controller = controlled["controller"]
    print(
        "Controlled rolling RMS: min={:.6g} at t={:.6g}, regrowth_time={}, delay={}".format(
            controlled["minimum_rolling_1s_tip_rms"],
            controlled["minimum_rolling_1s_tip_rms_time"],
            controlled["subsequent_regrowth_time"],
            controlled["time_from_minimum_to_regrowth"],
        )
    )
    print(
        "Controller: max_A={:.6g}, final_A={:.6g}, f_range={}, max_rates=({:.6g}, {:.6g}Hz/s), "
        "solve_median={:.6g}s, solve_max={:.6g}s".format(
            controller["maximum_A"], controller["final_A"],
            controller["frequency_hz_range"], controller["maximum_abs_A_dot"],
            controller["maximum_abs_frequency_dot_hz_s"],
            controller["median_solve_time_seconds"],
            controller["maximum_solve_time_seconds"],
        )
    )
    print(f"persistent_suppression={summary['persistent_suppression']}")


def consistent_step(times, expected):
    return bool(times) and all(abs((times[i] - times[i - 1]) - expected) < 1e-8
                               for i in range(1, len(times)))


def fraction_at(rows, name, bound, tolerance):
    return sum(abs(row[name] - bound) <= tolerance for row in rows) / len(rows)


def find_keys(value, names):
    found = set()
    if isinstance(value, dict):
        found.update(key for key in value if key in names)
        for child in value.values():
            found.update(find_keys(child, names))
    elif isinstance(value, list):
        for child in value:
            found.update(find_keys(child, names))
    return found


def window_name(start, end):
    return f"t_{start:g}_{end:g}"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
