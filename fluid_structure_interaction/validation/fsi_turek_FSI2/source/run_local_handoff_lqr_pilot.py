#!/usr/bin/env python3
"""Run and audit passive versus guarded local-LQR FSI2 startup cases."""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fsi2_local_handoff_lqr_controller import (
    LocalHandoffLaw,
    validate_artifact_schema,
    validate_export,
)


DEFAULT_LABEL = "fsi2_local_handoff_lqr_pilot_t40"
END_TIME = 40.0
OUTPUT_DT = 0.01
ACTIVATION_TIME = 3.0
WINDOWS = [(2, 3), (3, 5), (5, 10), (10, 20), (20, 30), (30, 40)]
CASES = [
    {"index": 0, "label": "passive_startup", "controller_type": "csv"},
    {"index": 1, "label": "local_lqr_startup", "controller_type": "local_handoff_lqr"},
]
REQUIRED_LOG_COLUMNS = [
    "time", "scheduled_update_time", "eta_1", "eta_2", "eta_3", "eta_4",
    "local_radius", "carrier_theta_rad", "carrier_phase_reference_rad",
    "envelope_relative_c", "envelope_relative_s", "envelope_ac", "envelope_as",
    "envelope_average_rate_relative_c", "envelope_average_rate_relative_s",
    "envelope_peak_rate_relative_c", "envelope_peak_rate_relative_s",
    "envelope_amplitude", "control_u", "predicted_next_radius",
    "is_update", "engaged", "guard_tripped", "status",
]


def main():
    args = parse_arguments()
    os.chdir(Path(__file__).resolve().parent)
    if args.list_cases:
        for case in CASES:
            print(f"{case['index']:03d} {case['label']} [{case['controller_type']}]")
        print(f"number_of_cases={len(CASES)}")
        return
    if args.validate_artifact:
        artifact = load_artifact(args.validate_artifact)
        print_artifact_report(args.validate_artifact, artifact)
        return
    if args.audit:
        audit_pilot(args.audit)
        return

    settings = read_settings()
    artifact = load_artifact(settings["artifact_path"])
    print_artifact_report(settings["artifact_path"], artifact)
    campaign = Path("run_outputs") / settings["campaign_label"]
    campaign.mkdir(parents=True, exist_ok=True)
    selected = CASES if args.case_index is None else [CASES[args.case_index]]
    for case in selected:
        print(f"[{case['index']}] {case['label']}", flush=True)
        run_directory = run_case(case, campaign, settings)
        result = {
            **case,
            "status": "completed",
            "run_directory": str(run_directory.resolve()),
            "beam_displacement_timeseries": str(
                (run_directory / "beam_displacement_timeseries.csv").resolve()),
            "actuator_timeseries": str(
                (run_directory / "actuator_timeseries.csv").resolve()),
            "local_lqr_timeseries": (
                str((run_directory / "local_lqr_timeseries.csv").resolve())
                if case["controller_type"] == "local_handoff_lqr" else None),
        }
        write_json_atomic(campaign / "case_results" / f"{case['label']}.json", result)
        print(f"completed={run_directory.resolve()}", flush=True)
    if args.case_index is None:
        audit_pilot(campaign)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run or audit the local-LQR pilot.")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--case-index", type=int, choices=range(len(CASES)))
    parser.add_argument("--validate-artifact", type=Path)
    parser.add_argument("--audit", type=Path)
    return parser.parse_args()


def read_settings():
    artifact = os.environ.get("KRATOS_FSI_LOCAL_CONTROLLER_FILE")
    if not artifact:
        raise RuntimeError("KRATOS_FSI_LOCAL_CONTROLLER_FILE is required.")
    return {
        "artifact_path": Path(artifact).resolve(),
        "campaign_label": os.environ.get(
            "KRATOS_FSI_LOCAL_LQR_LABEL", DEFAULT_LABEL),
        "write_paraview": read_boolean_environment_variable(
            "KRATOS_FSI_LOCAL_LQR_WRITE_PARAVIEW", False),
    }


def read_boolean_environment_variable(name, default):
    value = os.environ.get(name)
    return default if value is None else value.lower() not in ("0", "false", "no", "off")


def load_artifact(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Local-controller artifact not found: {path}")
    data = json.loads(path.read_text())
    validate_artifact_schema(data)
    errors = validate_export(data)
    if max(errors.values()) > 1e-10:
        raise RuntimeError(f"MATLAB/Python local-controller mismatch: {errors}")
    if abs(float(data["recommended_activation_time"]) - ACTIVATION_TIME) > 1e-12:
        raise RuntimeError("Pilot activation time differs from the exported recommendation.")
    return data


def print_artifact_report(path, data):
    errors = validate_export(data)
    print(f"artifact={Path(path).resolve()}")
    print(f"controller_type={data['controller_type']}")
    print(f"nominal_spectral_radius={data['nominal_closed_loop_spectral_radius']:.12g}")
    print(
        f"period={data['period']:.12g}, carrier={data['carrier_frequency_hz']:.12g}Hz, "
        f"envelope_limit={data['envelope_limit']}, rate_limit={data['rate_limit']}, "
        f"guard_radius={data['guard_radius']}")
    print("reference_errors=" + ",".join(
        f"{name}:{value:.3e}" for name, value in errors.items()))


def run_case(case, campaign, settings):
    run_directory = campaign / "runs" / case["label"]
    log_path = campaign / "logs" / f"{case['label']}.log"
    if run_directory.exists() or log_path.exists():
        raise FileExistsError(f"Refusing to overwrite {case['label']} outputs.")
    run_directory.mkdir(parents=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "KRATOS_FSI_RUN_LABEL": case["label"],
        "KRATOS_FSI_RUN_OUTPUT_DIRECTORY": str(run_directory.resolve()),
        "KRATOS_FSI_END_TIME": str(END_TIME),
        "KRATOS_FSI_OUTPUT_INTERVAL": str(OUTPUT_DT),
        "KRATOS_FSI_WRITE_PARAVIEW": "1" if settings["write_paraview"] else "0",
    })
    if case["controller_type"] == "csv":
        signal_path = campaign / "inputs" / "passive_zero.csv"
        write_zero_signal(signal_path)
        shutil.copyfile(signal_path, run_directory / "input_timeseries.csv")
        environment.update({
            "KRATOS_FSI_CONTROLLER_TYPE": "csv",
            "KRATOS_FSI_ACTUATOR_CSV_FILE": str(signal_path.resolve()),
            "KRATOS_FSI_ACTUATOR_CSV_TIME_COLUMN": "time",
            "KRATOS_FSI_ACTUATOR_CSV_VALUE_COLUMN": "value",
            "KRATOS_FSI_ACTUATOR_CSV_INTERPOLATION": "zoh",
        })
    else:
        artifact_copy = run_directory / "fsi2_local_handoff_controller.json"
        shutil.copyfile(settings["artifact_path"], artifact_copy)
        (run_directory / "controller_artifact.sha256").write_text(
            sha256_file(artifact_copy) + "  " + artifact_copy.name + "\n")
        environment.update({
            "KRATOS_FSI_CONTROLLER_TYPE": "local_handoff_lqr",
            "KRATOS_FSI_LOCAL_CONTROLLER_FILE": str(artifact_copy.resolve()),
            "KRATOS_FSI_LOCAL_CONTROLLER_ACTIVATION_TIME": str(ACTIVATION_TIME),
        })
    completed = subprocess.run(
        [sys.executable, "MainKratos.py"], env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"Run failed for {case['label']}. See {log_path}.")
    write_json_atomic(run_directory / "case_metadata.json", {
        "label": case["label"],
        "controller_type": case["controller_type"],
        "end_time": END_TIME,
        "output_dt": OUTPUT_DT,
        "activation_time": ACTIVATION_TIME,
        "startup_protocol": "deterministic_standard_startup_no_actuation",
        "paraview": settings["write_paraview"],
        "artifact_sha256": sha256_file(settings["artifact_path"]),
    })
    return run_directory


def write_zero_signal(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = [(0.0, 0.0), (END_TIME, 0.0)]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["time", "value"])
        writer.writerows(expected)
    temporary.replace(path)
    actual = [(float(row["time"]), float(row["value"]))
              for row in csv.DictReader(path.open(newline=""))]
    if actual != expected:
        raise RuntimeError(f"Published passive input differs: {path}")


def audit_pilot(campaign):
    campaign = Path(campaign)
    artifact_path = campaign / "runs" / "local_lqr_startup" \
        / "fsi2_local_handoff_controller.json"
    artifact = load_artifact(artifact_path)
    law = LocalHandoffLaw(artifact)
    errors = []
    results = {}
    for case in CASES:
        results[case["label"]] = audit_case(
            campaign / "runs" / case["label"], case, law, errors)
    if all("run_directory" in result for result in results.values()):
        compare_startup_actuation(results, errors)
    if errors:
        print("Pilot audit failed:")
        for error in errors:
            print(f"  ERROR: {error}")
        raise RuntimeError("Local-LQR pilot audit failed.")

    passive = results["passive_startup"]
    controlled = results["local_lqr_startup"]
    ratios = {}
    for start, end in WINDOWS:
        name = window_name(start, end)
        ratios[name] = (controlled["windows"][name]["rms"]
                        / passive["windows"][name]["rms"])
    controller = controlled["controller"]
    pilot_success = (
        not controller["guard_tripped"]
        and controller["final_local_radius"] < controller["initial_local_radius"]
        and ratios[window_name(30, 40)] < 0.25
    )
    summary = {
        "experiment": {
            "label": campaign.name,
            "end_time": END_TIME,
            "activation_time": ACTIVATION_TIME,
            "startup_protocol": "deterministic_standard_startup_no_actuation",
            "interpretation": (
                "near-equilibrium startup pilot, not an exact equilibrium restart"),
        },
        "artifact": {
            "sha256": sha256_file(artifact_path),
            "guard_radius": law.guard_radius,
            "envelope_limit": law.envelope_limit,
            "rate_limit": law.rate_limit,
            "period": law.period,
        },
        "cases": results,
        "controlled_passive_rms_ratio": ratios,
        "pilot_success": pilot_success,
    }
    write_json_atomic(campaign / "summary.json", summary)
    print("Pilot audit passed: 2/2 cases complete and internally consistent.")
    for start, end in WINDOWS:
        name = window_name(start, end)
        print(
            f"  [{start},{end}] passive_rms={passive['windows'][name]['rms']:.6g}, "
            f"local_rms={controlled['windows'][name]['rms']:.6g}, "
            f"ratio={ratios[name]:.6g}")
    print(
        "Local controller: initial_radius={:.6g}, final_radius={:.6g}, "
        "max_radius={:.6g}, max|a|={:.6g}, max|adot|={:.6g}, guard_tripped={}".format(
            controller["initial_local_radius"], controller["final_local_radius"],
            controller["maximum_local_radius"], controller["maximum_envelope"],
            controller["maximum_rate"], controller["guard_tripped"]))
    print(f"pilot_success={pilot_success}")


def audit_case(run_directory, case, law, errors):
    required = [
        run_directory / "beam_displacement_timeseries.csv",
        run_directory / "actuator_timeseries.csv",
        run_directory / "ProjectParameters.effective.json",
        run_directory / "case_metadata.json",
    ]
    if case["controller_type"] == "local_handoff_lqr":
        required.append(run_directory / "local_lqr_timeseries.csv")
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        errors.append(f"{case['label']}: missing {missing}")
        return {"status": "incomplete"}
    times, tip = read_tip_series(run_directory / "beam_displacement_timeseries.csv")
    if abs(times[-1] - END_TIME) > 1e-8 or len(times) != round(END_TIME / OUTPUT_DT) + 1:
        errors.append(f"{case['label']}: incomplete beam history")
    actuator = read_actuator_series(run_directory / "actuator_timeseries.csv")
    balance = maximum_balance_error(actuator)
    if balance > 1e-10:
        errors.append(f"{case['label']}: actuator imbalance {balance:.3e}")
    windows = {
        window_name(start, end): window_stats(times, tip, start, end)
        for start, end in WINDOWS
    }
    result = {
        "label": case["label"],
        "run_directory": str(run_directory.resolve()),
        "last_time": times[-1],
        "number_of_samples": len(times),
        "windows": windows,
        "max_actuator_balance_error": balance,
        "maximum_abs_tip_y": max(abs(value) for value in tip),
        "maximum_abs_control": max(abs(value) for _, value in actuator["upper"]),
    }
    if case["controller_type"] == "csv":
        if result["maximum_abs_control"] > 1e-12:
            errors.append("passive_startup: expected exactly zero actuation")
    else:
        result["controller"] = audit_local_log(
            run_directory / "local_lqr_timeseries.csv", actuator, law, errors)
    return result


def audit_local_log(path, actuator, law, errors):
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        missing = [name for name in REQUIRED_LOG_COLUMNS if name not in reader.fieldnames]
        rows = list(reader)
    if missing or not rows:
        errors.append(f"local_lqr_startup: invalid local log, missing={missing}")
        return {}
    numeric_rows = [{
        **{name: float(row[name]) for name in REQUIRED_LOG_COLUMNS
           if name not in ("is_update", "engaged", "guard_tripped", "status")},
        "is_update": int(row["is_update"]),
        "engaged": int(row["engaged"]),
        "guard_tripped": int(row["guard_tripped"]),
        "status": row["status"],
    } for row in rows]
    updates = [row for row in numeric_rows if row["is_update"] == 1]
    if updates:
        scheduled = [row["scheduled_update_time"] for row in updates]
        if abs(scheduled[0] - ACTIVATION_TIME) > 1e-8:
            errors.append("local_lqr_startup: first update is not at activation")
        if any(abs((scheduled[i] - scheduled[i - 1]) - law.period) > 1e-8
               for i in range(1, len(scheduled))):
            errors.append("local_lqr_startup: inconsistent stroboscopic schedule")
    elif not any(row["guard_tripped"] for row in numeric_rows):
        errors.append("local_lqr_startup: no update and no recorded guard trip")
        return {}
    maximum_average_rate = max(math.hypot(
        row["envelope_average_rate_relative_c"],
        row["envelope_average_rate_relative_s"])
        for row in numeric_rows)
    maximum_rate = max(math.hypot(
        row["envelope_peak_rate_relative_c"], row["envelope_peak_rate_relative_s"])
        for row in numeric_rows)
    peak_rate_consistency_error = max(abs(
        math.hypot(row["envelope_peak_rate_relative_c"],
                   row["envelope_peak_rate_relative_s"])
        - law.interpolation_peak_rate_factor * math.hypot(
            row["envelope_average_rate_relative_c"],
            row["envelope_average_rate_relative_s"]))
        for row in numeric_rows)
    if peak_rate_consistency_error > 1e-9:
        errors.append(
            f"local_lqr_startup: inconsistent smoothstep peak rate "
            f"{peak_rate_consistency_error:.3e}")
    maximum_envelope = max(row["envelope_amplitude"] for row in numeric_rows)
    if maximum_rate > law.rate_limit + 1e-10:
        errors.append(f"local_lqr_startup: rate bound violation {maximum_rate}")
    if maximum_envelope > law.envelope_limit + 1e-10:
        errors.append(f"local_lqr_startup: envelope bound violation {maximum_envelope}")
    identity_error = max(abs(
        row["control_u"] - (
            row["envelope_relative_c"] * math.cos(
                row["carrier_theta_rad"] - row["carrier_phase_reference_rad"])
            + row["envelope_relative_s"] * math.sin(
                row["carrier_theta_rad"] - row["carrier_phase_reference_rad"])))
        for row in numeric_rows)
    if identity_error > 1e-9:
        errors.append(f"local_lqr_startup: control identity error {identity_error:.3e}")
    actuator_by_time = {round(time, 8): value for time, value in actuator["upper"]}
    actuator_error = max(abs(
        row["control_u"] - actuator_by_time.get(round(row["time"], 8), math.inf))
        for row in numeric_rows)
    if actuator_error > 1e-9:
        errors.append(f"local_lqr_startup: actuator/log mismatch {actuator_error:.3e}")
    guard_tripped = any(int(row["guard_tripped"]) for row in rows)
    engaged_radii = [row["local_radius"] for row in updates]
    if engaged_radii and max(engaged_radii) > law.guard_radius + 1e-10:
        errors.append("local_lqr_startup: engaged update outside guard")
    finite_radii = [row["local_radius"] for row in numeric_rows
                    if math.isfinite(row["local_radius"])]
    if not finite_radii:
        finite_radii = [math.inf]
    return {
        "number_of_updates": len(updates),
        "initial_local_radius": finite_radii[0],
        "final_local_radius": finite_radii[-1],
        "minimum_local_radius": min(finite_radii),
        "maximum_local_radius": max(finite_radii),
        "final_initial_radius_ratio": (
            finite_radii[-1] / finite_radii[0] if finite_radii[0] else None),
        "maximum_envelope": maximum_envelope,
        "maximum_average_rate": maximum_average_rate,
        "maximum_rate": maximum_rate,
        "maximum_peak_rate_consistency_error": peak_rate_consistency_error,
        "maximum_control_identity_error": identity_error,
        "maximum_actuator_log_error": actuator_error,
        "guard_tripped": guard_tripped,
        "final_status": rows[-1]["status"],
    }


def compare_startup_actuation(results, errors):
    passive = read_actuator_series(
        Path(results["passive_startup"]["run_directory"]) / "actuator_timeseries.csv")
    local = read_actuator_series(
        Path(results["local_lqr_startup"]["run_directory"]) / "actuator_timeseries.csv")
    p = dict(passive["upper"])
    q = dict(local["upper"])
    common = [time for time in p if time <= ACTIVATION_TIME + 1e-10 and time in q]
    difference = max((abs(p[time] - q[time]) for time in common), default=math.inf)
    if difference > 1e-12:
        errors.append(f"startup actuator histories differ by {difference:.3e}")


def read_tip_series(path):
    with path.open(newline="") as input_file:
        reader = csv.reader(input_file)
        next(reader)
        header = next(reader)
        index = header.index("tip_DISPLACEMENT_Y")
        rows = list(reader)
    return [float(row[0]) for row in rows], [float(row[index]) for row in rows]


def read_actuator_series(path):
    result = {"upper": [], "lower": []}
    with path.open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            key = "upper" if row["actuator_name"].endswith("_upper") else "lower"
            result[key].append((float(row["time"]), float(row["control_value"])))
    return result


def maximum_balance_error(actuator):
    lower = dict(actuator["lower"])
    return max((abs(value + lower.get(time, math.inf))
                for time, value in actuator["upper"]), default=math.inf)


def window_stats(times, values, start, end):
    selected = [value for time, value in zip(times, values)
                if start - 1e-10 <= time <= end + 1e-10]
    return {
        "rms": math.sqrt(sum(value * value for value in selected) / len(selected)),
        "peak_to_peak": max(selected) - min(selected),
        "samples": len(selected),
    }


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
