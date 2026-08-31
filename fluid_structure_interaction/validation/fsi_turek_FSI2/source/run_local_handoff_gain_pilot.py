#!/usr/bin/env python3
"""Run and audit the 10 s gain-three guarded local-LQR pilot."""

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


DEFAULT_LABEL = "fsi2_local_handoff_lqr_gain3_t10"
CASE_LABEL = "local_lqr_gain3"
END_TIME = 10.0
OUTPUT_DT = 0.01
ACTIVATION_TIME = 3.0
GAIN_MULTIPLIER = 3.0
EXPECTED_GUARD_RADIUS = 0.01
WEAK_BURST_ENVELOPE_LIMIT = 0.1
WINDOWS = [(2.0, 3.0), (3.0, 5.0), (5.0, 7.5), (7.5, 10.0)]
REQUIRED_LOG_COLUMNS = [
    "time", "scheduled_update_time", "eta_1", "eta_2", "eta_3", "eta_4",
    "local_radius", "carrier_theta_rad", "carrier_phase_reference_rad",
    "envelope_relative_c", "envelope_relative_s", "envelope_ac", "envelope_as",
    "envelope_average_rate_relative_c", "envelope_average_rate_relative_s",
    "envelope_peak_rate_relative_c", "envelope_peak_rate_relative_s",
    "envelope_amplitude", "control_u", "predicted_next_radius",
    "feedback_gain_multiplier", "is_update", "engaged", "guard_tripped", "status",
]


def main():
    args = parse_arguments()
    os.chdir(Path(__file__).resolve().parent)
    if args.validate_artifact:
        print_artifact_report(args.validate_artifact, load_artifact(args.validate_artifact))
        return
    if args.validate_reference:
        report = validate_passive_reference(args.validate_reference)
        print_reference_report(report)
        return
    if args.audit:
        audit_pilot(args.audit)
        return

    settings = read_settings()
    artifact = load_artifact(settings["artifact_path"])
    reference = validate_passive_reference(settings["passive_reference"])
    print_artifact_report(settings["artifact_path"], artifact)
    print_reference_report(reference)

    campaign = Path("run_outputs") / settings["campaign_label"]
    if campaign.exists():
        raise FileExistsError(f"Refusing to overwrite campaign: {campaign}")
    run_directory = campaign / "runs" / CASE_LABEL
    log_path = campaign / "logs" / f"{CASE_LABEL}.log"
    run_directory.mkdir(parents=True)
    log_path.parent.mkdir(parents=True)

    artifact_copy = run_directory / "fsi2_local_handoff_controller.json"
    shutil.copyfile(settings["artifact_path"], artifact_copy)
    (run_directory / "controller_artifact.sha256").write_text(
        sha256_file(artifact_copy) + "  " + artifact_copy.name + "\n")

    environment = os.environ.copy()
    environment.update({
        "KRATOS_FSI_RUN_LABEL": CASE_LABEL,
        "KRATOS_FSI_RUN_OUTPUT_DIRECTORY": str(run_directory.resolve()),
        "KRATOS_FSI_END_TIME": str(END_TIME),
        "KRATOS_FSI_OUTPUT_INTERVAL": str(OUTPUT_DT),
        "KRATOS_FSI_WRITE_PARAVIEW": "0",
        "KRATOS_FSI_CONTROLLER_TYPE": "local_handoff_lqr",
        "KRATOS_FSI_LOCAL_CONTROLLER_FILE": str(artifact_copy.resolve()),
        "KRATOS_FSI_LOCAL_CONTROLLER_ACTIVATION_TIME": str(ACTIVATION_TIME),
        "KRATOS_FSI_LOCAL_CONTROLLER_GAIN_MULTIPLIER": str(GAIN_MULTIPLIER),
    })
    completed = subprocess.run(
        [sys.executable, "MainKratos.py"], env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"Run failed. See {log_path}.")

    metadata = {
        "label": CASE_LABEL,
        "controller_type": "local_handoff_lqr",
        "end_time": END_TIME,
        "output_dt": OUTPUT_DT,
        "activation_time": ACTIVATION_TIME,
        "feedback_gain_multiplier": GAIN_MULTIPLIER,
        "guard_radius": float(artifact["guard_radius"]),
        "startup_protocol": "deterministic_standard_startup_no_actuation",
        "passive_reference_run": str(settings["passive_reference"]),
        "paraview": False,
        "artifact_sha256": sha256_file(settings["artifact_path"]),
    }
    write_json_atomic(run_directory / "case_metadata.json", metadata)
    write_json_atomic(campaign / "manifest.json", {
        "label": settings["campaign_label"],
        "case": metadata,
    })
    audit_pilot(campaign)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-artifact", type=Path)
    parser.add_argument("--validate-reference", type=Path)
    parser.add_argument("--audit", type=Path)
    return parser.parse_args()


def read_settings():
    artifact = os.environ.get("KRATOS_FSI_LOCAL_CONTROLLER_FILE")
    reference = os.environ.get("KRATOS_FSI_LOCAL_GAIN_PASSIVE_REFERENCE_RUN")
    if not artifact:
        raise RuntimeError("KRATOS_FSI_LOCAL_CONTROLLER_FILE is required.")
    if not reference:
        raise RuntimeError("KRATOS_FSI_LOCAL_GAIN_PASSIVE_REFERENCE_RUN is required.")
    return {
        "artifact_path": Path(artifact).resolve(),
        "passive_reference": Path(reference).resolve(),
        "campaign_label": os.environ.get(
            "KRATOS_FSI_LOCAL_GAIN_LABEL", DEFAULT_LABEL),
    }


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
        raise RuntimeError("Activation time differs from the exported recommendation.")
    if abs(float(data["guard_radius"]) - EXPECTED_GUARD_RADIUS) > 1e-12:
        raise RuntimeError(
            f"Pilot requires guard radius {EXPECTED_GUARD_RADIUS}; "
            f"artifact contains {data['guard_radius']}.")
    return data


def print_artifact_report(path, data):
    errors = validate_export(data)
    print(f"artifact={Path(path).resolve()}")
    print(
        f"gain={GAIN_MULTIPLIER:g}, activation={ACTIVATION_TIME:g}s, "
        f"guard={data['guard_radius']}, envelope_limit={data['envelope_limit']}, "
        f"rate_limit={data['rate_limit']}")
    print("reference_errors=" + ",".join(
        f"{name}:{value:.3e}" for name, value in errors.items()))


def validate_passive_reference(path):
    run = Path(path).resolve()
    required = [
        run / "beam_displacement_timeseries.csv",
        run / "actuator_timeseries.csv",
        run / "case_metadata.json",
    ]
    missing = [item.name for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Passive reference is missing {missing}: {run}")
    times, tip = read_tip_series(required[0])
    if times[0] > 1e-10 or times[-1] + 1e-10 < END_TIME:
        raise RuntimeError("Passive reference does not span the 10 s pilot.")
    relevant_times = [time for time in times if time <= END_TIME + 1e-10]
    if len(relevant_times) != round(END_TIME / OUTPUT_DT) + 1:
        raise RuntimeError("Passive reference has the wrong number of 0-10 s samples.")
    spacing_error = max(abs(
        relevant_times[i] - relevant_times[i - 1] - OUTPUT_DT)
        for i in range(1, len(relevant_times)))
    if spacing_error > 1e-9:
        raise RuntimeError(f"Passive reference output spacing error: {spacing_error:.3e}")
    actuator = read_actuator_series(required[1])
    balance_error = maximum_balance_error(actuator, END_TIME)
    maximum_control = max(
        (abs(value) for time, value in actuator["upper"]
         if time <= END_TIME + 1e-10), default=math.inf)
    if balance_error > 1e-10 or maximum_control > 1e-12:
        raise RuntimeError(
            "Passive reference must have balanced, exactly zero actuation through 10 s.")
    metadata = json.loads(required[2].read_text())
    reference_label = metadata.get("label")
    if reference_label is None and isinstance(metadata.get("case"), dict):
        reference_label = metadata["case"].get("label")
    if reference_label not in ("passive_startup", "early_passive_reference"):
        raise RuntimeError(
            "Passive reference metadata must describe passive_startup or "
            "early_passive_reference.")
    return {
        "run_directory": str(run),
        "last_time": times[-1],
        "number_of_samples_through_10s": len(relevant_times),
        "spacing_error": spacing_error,
        "balance_error": balance_error,
        "maximum_abs_control": maximum_control,
        "times": times,
        "tip": tip,
        "actuator": actuator,
    }


def print_reference_report(report):
    print(f"passive_reference={report['run_directory']}")
    print(
        f"reference_last_time={report['last_time']:g}, "
        f"samples_0_10={report['number_of_samples_through_10s']}, "
        f"max|u|={report['maximum_abs_control']:.3e}")


def audit_pilot(campaign_path):
    campaign = Path(campaign_path).resolve()
    run = campaign / "runs" / CASE_LABEL
    required = [
        run / "beam_displacement_timeseries.csv",
        run / "actuator_timeseries.csv",
        run / "local_lqr_timeseries.csv",
        run / "ProjectParameters.effective.json",
        run / "case_metadata.json",
        run / "fsi2_local_handoff_controller.json",
    ]
    missing = [item.name for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Controlled run is missing {missing}: {run}")

    metadata = json.loads((run / "case_metadata.json").read_text())
    for name, expected in (
            ("end_time", END_TIME), ("output_dt", OUTPUT_DT),
            ("activation_time", ACTIVATION_TIME),
            ("feedback_gain_multiplier", GAIN_MULTIPLIER),
            ("guard_radius", EXPECTED_GUARD_RADIUS)):
        if abs(float(metadata.get(name, math.inf)) - expected) > 1e-12:
            raise RuntimeError(f"Metadata mismatch for {name}.")
    artifact = load_artifact(run / "fsi2_local_handoff_controller.json")
    law = LocalHandoffLaw(artifact, GAIN_MULTIPLIER)
    reference_path = Path(metadata["passive_reference_run"])
    if not reference_path.is_dir():
        override = os.environ.get("KRATOS_FSI_LOCAL_GAIN_PASSIVE_REFERENCE_RUN")
        if override:
            reference_path = Path(override)
    reference = validate_passive_reference(reference_path)

    times, tip = read_tip_series(run / "beam_displacement_timeseries.csv")
    if len(times) != round(END_TIME / OUTPUT_DT) + 1 \
            or abs(times[0]) > 1e-10 or abs(times[-1] - END_TIME) > 1e-8:
        raise RuntimeError("Controlled beam history is incomplete.")
    spacing_error = max(abs(
        times[i] - times[i - 1] - OUTPUT_DT) for i in range(1, len(times)))
    if spacing_error > 1e-9:
        raise RuntimeError(f"Controlled output spacing error: {spacing_error:.3e}")

    actuator = read_actuator_series(run / "actuator_timeseries.csv")
    balance_error = maximum_balance_error(actuator, END_TIME)
    if balance_error > 1e-10:
        raise RuntimeError(f"Controlled actuator imbalance: {balance_error:.3e}")
    upper = {round(time, 9): value for time, value in actuator["upper"]}
    startup_control = max(
        (abs(value) for time, value in actuator["upper"]
         if time <= ACTIVATION_TIME + 1e-10), default=math.inf)
    if startup_control > 1e-12:
        raise RuntimeError("Controlled actuation is nonzero before activation.")

    log_report = audit_controller_log(
        run / "local_lqr_timeseries.csv", upper, law)
    passive_times = reference["times"]
    passive_tip = reference["tip"]
    windows = {}
    for start, end in WINDOWS:
        name = window_name(start, end)
        controlled_stats = window_stats(times, tip, start, end)
        passive_stats = window_stats(passive_times, passive_tip, start, end)
        windows[name] = {
            "controlled": controlled_stats,
            "passive": passive_stats,
            "controlled_passive_rms_ratio": (
                controlled_stats["rms"] / passive_stats["rms"]),
        }
    passive_by_time = {
        round(time, 9): value for time, value in zip(passive_times, passive_tip)
        if time <= ACTIVATION_TIME + 1e-10
    }
    startup_tip_mismatch = max(abs(
        value - passive_by_time.get(round(time, 9), math.inf))
        for time, value in zip(times, tip)
        if time <= ACTIVATION_TIME + 1e-10)
    if startup_tip_mismatch > 1e-8:
        raise RuntimeError(
            f"Controlled/passive startup mismatch: {startup_tip_mismatch:.3e}")

    departed = log_report["guard_tripped"]
    within_weak_burst_input_range = (
        log_report["maximum_envelope"] <= WEAK_BURST_ENVELOPE_LIMIT + 1e-10)
    pilot_success = (
        not departed
        and within_weak_burst_input_range
        and log_report["final_local_radius"] < log_report["initial_local_radius"]
    )
    if departed:
        recommended_next_step = "pivot_to_phase_synchronous_probe_feedback"
    elif not within_weak_burst_input_range:
        recommended_next_step = "review_input_domain_before_further_validation"
    else:
        recommended_next_step = "retain_local_lqr_architecture_for_further_validation"
    summary = {
        "experiment": {
            "label": campaign.name,
            "end_time": END_TIME,
            "activation_time": ACTIVATION_TIME,
            "feedback_gain_multiplier": GAIN_MULTIPLIER,
            "guard_radius": EXPECTED_GUARD_RADIUS,
            "paraview": False,
        },
        "artifact": {
            "sha256": sha256_file(run / "fsi2_local_handoff_controller.json"),
            "envelope_limit": law.envelope_limit,
            "rate_limit": law.rate_limit,
        },
        "passive_reference": reference["run_directory"],
        "controlled_run": str(run),
        "maximum_actuator_balance_error": balance_error,
        "maximum_tip_mismatch_before_activation": startup_tip_mismatch,
        "windows": windows,
        "controller": log_report,
        "departed_validated_neighborhood": departed,
        "within_weak_burst_input_range": within_weak_burst_input_range,
        "pilot_success": pilot_success,
        "recommended_next_step": recommended_next_step,
    }
    write_json_atomic(campaign / "summary.json", summary)
    print("Gain-three pilot integrity audit passed.")
    for start, end in WINDOWS:
        item = windows[window_name(start, end)]
        print(
            f"  [{start:g},{end:g}] passive_rms={item['passive']['rms']:.6g}, "
            f"controlled_rms={item['controlled']['rms']:.6g}, "
            f"ratio={item['controlled_passive_rms_ratio']:.6g}")
    print(
        "Controller: initial_radius={:.6g}, final_radius={:.6g}, "
        "max_radius={:.6g}, max|a|={:.6g}, max|adot|={:.6g}, "
        "guard_tripped={}".format(
            log_report["initial_local_radius"], log_report["final_local_radius"],
            log_report["maximum_local_radius"], log_report["maximum_envelope"],
            log_report["maximum_rate"], log_report["guard_tripped"]))
    print(f"within_A0p1_identification_range={within_weak_burst_input_range}")
    print(f"pilot_success={pilot_success}")
    print(f"recommended_next_step={summary['recommended_next_step']}")


def audit_controller_log(path, actuator_by_time, law):
    with Path(path).open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        missing = [name for name in REQUIRED_LOG_COLUMNS if name not in reader.fieldnames]
        rows = list(reader)
    if missing or not rows:
        raise RuntimeError(f"Invalid local-controller log; missing={missing}")
    numeric_names = [name for name in REQUIRED_LOG_COLUMNS
                     if name not in ("is_update", "engaged", "guard_tripped", "status")]
    numeric_rows = [{
        **{name: float(row[name]) for name in numeric_names},
        "is_update": int(row["is_update"]),
        "engaged": int(row["engaged"]),
        "guard_tripped": int(row["guard_tripped"]),
        "status": row["status"],
    } for row in rows]
    if any(abs(row["feedback_gain_multiplier"] - GAIN_MULTIPLIER) > 1e-12
           for row in numeric_rows):
        raise RuntimeError("Controller log does not consistently use gain multiplier 3.")
    updates = [row for row in numeric_rows if row["is_update"]]
    if not updates:
        raise RuntimeError("Local controller recorded no update.")
    scheduled = [row["scheduled_update_time"] for row in updates]
    if abs(scheduled[0] - ACTIVATION_TIME) > 1e-8 or any(abs(
            scheduled[i] - scheduled[i - 1] - law.period) > 1e-8
            for i in range(1, len(scheduled))):
        raise RuntimeError("Local-controller update schedule is inconsistent.")
    maximum_rate = max(math.hypot(
        row["envelope_peak_rate_relative_c"],
        row["envelope_peak_rate_relative_s"]) for row in numeric_rows)
    maximum_envelope = max(row["envelope_amplitude"] for row in numeric_rows)
    if maximum_rate > law.rate_limit + 1e-10:
        raise RuntimeError(f"Rate bound violation: {maximum_rate}")
    if maximum_envelope > law.envelope_limit + 1e-10:
        raise RuntimeError(f"Envelope bound violation: {maximum_envelope}")
    identity_error = max(abs(
        row["control_u"] - (
            row["envelope_relative_c"] * math.cos(
                row["carrier_theta_rad"] - row["carrier_phase_reference_rad"])
            + row["envelope_relative_s"] * math.sin(
                row["carrier_theta_rad"] - row["carrier_phase_reference_rad"])))
        for row in numeric_rows)
    actuator_error = max(abs(
        row["control_u"]
        - actuator_by_time.get(round(row["time"], 9), math.inf))
        for row in numeric_rows)
    if identity_error > 1e-9 or actuator_error > 1e-9:
        raise RuntimeError(
            f"Controller identity/actuator mismatch: {identity_error:.3e}, "
            f"{actuator_error:.3e}")
    finite_radii = [row["local_radius"] for row in numeric_rows
                    if math.isfinite(row["local_radius"])]
    if not finite_radii:
        raise RuntimeError("Controller log contains no finite local radius.")
    guard_rows = [row for row in numeric_rows if row["guard_tripped"]]
    return {
        "number_of_updates": len(updates),
        "initial_local_radius": finite_radii[0],
        "final_local_radius": finite_radii[-1],
        "minimum_local_radius": min(finite_radii),
        "maximum_local_radius": max(finite_radii),
        "maximum_envelope": maximum_envelope,
        "maximum_rate": maximum_rate,
        "maximum_control_identity_error": identity_error,
        "maximum_actuator_log_error": actuator_error,
        "guard_tripped": bool(guard_rows),
        "guard_trip_time": guard_rows[0]["time"] if guard_rows else None,
        "final_status": rows[-1]["status"],
    }


def read_tip_series(path):
    with Path(path).open(newline="") as input_file:
        reader = csv.reader(input_file)
        next(reader)
        header = next(reader)
        index = header.index("tip_DISPLACEMENT_Y")
        rows = list(reader)
    return [float(row[0]) for row in rows], [float(row[index]) for row in rows]


def read_actuator_series(path):
    result = {"upper": [], "lower": []}
    with Path(path).open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            key = "upper" if row["actuator_name"].endswith("_upper") else "lower"
            result[key].append((float(row["time"]), float(row["control_value"])))
    return result


def maximum_balance_error(actuator, end_time):
    lower = {round(time, 9): value for time, value in actuator["lower"]}
    return max((abs(value + lower.get(round(time, 9), math.inf))
                for time, value in actuator["upper"]
                if time <= end_time + 1e-10), default=math.inf)


def window_stats(times, values, start, end):
    selected = [value for time, value in zip(times, values)
                if start - 1e-10 <= time <= end + 1e-10]
    if not selected:
        raise RuntimeError(f"No samples in window [{start},{end}].")
    mean = sum(selected) / len(selected)
    return {
        "rms": math.sqrt(sum(value * value for value in selected) / len(selected)),
        "mean": mean,
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
