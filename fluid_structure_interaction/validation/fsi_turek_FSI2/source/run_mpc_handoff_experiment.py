#!/usr/bin/env python3
"""Run and audit one MPC-to-local-LQR FSI2 trajectory."""

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

from fsi2_mpc_handoff_controller import validate_artifact_schema, validate_export


DEFAULT_LABEL = "fsi2_mpc_local_handoff_t40"
DEFAULT_PASSIVE_REFERENCE = Path(
    "run_outputs/fsi_control_mpc_identification_t40_u20/"
    "runs/kick_then_passive_no_intervention"
)
CASE_LABEL = "mpc_local_handoff"
END_TIME = 40.0
OUTPUT_DT = 0.01
KICK_VALUE = 0.4
KICK_END_TIME = 2.0
ACTIVATION_TIME = 8.0
WINDOWS = [(6, 8), (8, 12), (12, 16), (16, 20), (20, 30), (30, 40)]
FINAL_WINDOW = "30_40"
REQUIRED_CONTROLLER_COLUMNS = {
    "time", "mode", "event", "local_radius", "A", "envelope_amplitude",
    "handoff_streak",
}


def main():
    args = parse_arguments()
    os.chdir(Path(__file__).resolve().parent)
    if args.validate_artifact:
        artifact = load_artifact(args.validate_artifact)
        report_artifact(args.validate_artifact, artifact)
        return
    if args.validate_reference:
        report_reference(validate_passive_reference(args.validate_reference))
        return
    if args.audit:
        audit_experiment(args.audit)
        return

    settings = read_settings()
    artifact = load_artifact(settings["artifact_path"])
    reference = validate_passive_reference(settings["passive_reference"])
    report_artifact(settings["artifact_path"], artifact)
    report_reference(reference)

    campaign = Path("run_outputs") / settings["campaign_label"]
    if campaign.exists():
        raise FileExistsError(f"Refusing to overwrite existing campaign: {campaign}")
    campaign.mkdir(parents=True)
    write_json(campaign / "experiment_metadata.json", {
        "label": settings["campaign_label"],
        "case_label": CASE_LABEL,
        "end_time": END_TIME,
        "output_dt": OUTPUT_DT,
        "kick_value": KICK_VALUE,
        "kick_end_time": KICK_END_TIME,
        "activation_time": ACTIVATION_TIME,
        "passive_reference_run": str(settings["passive_reference"]),
        "artifact_path": str(settings["artifact_path"]),
        "artifact_sha256": sha256(settings["artifact_path"]),
        "write_paraview": settings["write_paraview"],
    })
    run_controlled_case(campaign, settings)
    audit_experiment(campaign)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-artifact", type=Path)
    parser.add_argument("--validate-reference", type=Path)
    parser.add_argument("--audit", type=Path)
    return parser.parse_args()


def read_settings():
    artifact = required_path("KRATOS_FSI_HANDOFF_CONTROLLER_FILE")
    reference = Path(os.environ.get(
        "KRATOS_FSI_PASSIVE_REFERENCE_RUN", str(DEFAULT_PASSIVE_REFERENCE)
    )).resolve()
    return {
        "artifact_path": artifact,
        "passive_reference": reference,
        "campaign_label": os.environ.get(
            "KRATOS_FSI_MPC_HANDOFF_LABEL", DEFAULT_LABEL),
        "write_paraview": read_bool(
            "KRATOS_FSI_MPC_HANDOFF_WRITE_PARAVIEW", False),
    }


def required_path(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required.")
    return Path(value).resolve()


def load_artifact(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text())
    if data.get("controller_type") != "mpc_local_handoff":
        raise RuntimeError(
            "Expected fsi2_mpc_handoff_controller.json; received a standalone "
            f"{data.get('controller_type', 'unknown')} artifact."
        )
    validate_artifact_schema(data)
    errors = validate_export(data)
    if max(errors.values()) > 1e-7:
        raise RuntimeError(f"MATLAB/Python artifact mismatch: {errors}")
    return data


def report_artifact(path, data):
    errors = validate_export(data)
    print(f"artifact={Path(path).resolve()}")
    print("reference_errors=" + ",".join(
        f"{name}:{value:.3e}" for name, value in errors.items()))
    print(
        f"handoff: entry={data['handoff']['entry_radius']}, "
        f"exit={data['handoff']['exit_radius']}, "
        f"Amax={data['handoff']['maximum_entry_amplitude']}, "
        f"dwell={data['handoff']['required_consecutive_updates']} updates"
    )


def run_controlled_case(campaign, settings):
    run_directory = campaign / "runs" / CASE_LABEL
    log_path = campaign / "logs" / f"{CASE_LABEL}.log"
    run_directory.mkdir(parents=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_copy = run_directory / "fsi2_mpc_handoff_controller.json"
    shutil.copyfile(settings["artifact_path"], artifact_copy)
    (run_directory / "controller_artifact.sha256").write_text(
        f"{sha256(artifact_copy)}  {artifact_copy.name}\n"
    )
    environment = os.environ.copy()
    environment.update({
        "KRATOS_FSI_RUN_LABEL": CASE_LABEL,
        "KRATOS_FSI_RUN_OUTPUT_DIRECTORY": str(run_directory.resolve()),
        "KRATOS_FSI_END_TIME": str(END_TIME),
        "KRATOS_FSI_OUTPUT_INTERVAL": str(OUTPUT_DT),
        "KRATOS_FSI_WRITE_PARAVIEW": (
            "1" if settings["write_paraview"] else "0"),
        "KRATOS_FSI_CONTROLLER_TYPE": "mpc_local_handoff",
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
    if completed.returncode != 0:
        raise RuntimeError(f"Run failed for {CASE_LABEL}; see {log_path}.")

    metadata = {
        "label": CASE_LABEL,
        "controller_type": "mpc_local_handoff",
        "end_time": END_TIME,
        "output_dt": OUTPUT_DT,
        "kick_value": KICK_VALUE,
        "kick_end_time": KICK_END_TIME,
        "activation_time": ACTIVATION_TIME,
        "passive_reference_run": str(settings["passive_reference"]),
        "artifact_sha256": sha256(artifact_copy),
        "paraview": settings["write_paraview"],
    }
    write_json(run_directory / "case_metadata.json", metadata)
    write_json(campaign / "case_results" / f"{CASE_LABEL}.json", {
        **metadata,
        "status": "completed",
        "run_directory": str(run_directory.resolve()),
    })
    print(f"completed={run_directory.resolve()}")


def validate_passive_reference(run_directory):
    run_directory = Path(run_directory).resolve()
    errors = []
    beam_path = run_directory / "beam_displacement_timeseries.csv"
    actuator_path = run_directory / "actuator_timeseries.csv"
    metadata_path = run_directory / "case_metadata.json"
    missing = [path.name for path in (beam_path, actuator_path, metadata_path)
               if not path.is_file()]
    if missing:
        raise RuntimeError(f"Passive reference is missing files: {missing}")

    times, tip = read_tip(beam_path)
    validate_beam_grid(times, "passive reference", errors)
    actuator = read_actuator(actuator_path)
    validate_actuator_balance(actuator, "passive reference", errors)
    validate_reference_actuation(actuator, errors)
    metadata = json.loads(metadata_path.read_text())
    case = metadata.get("case", {})
    if case.get("label") != "kick_then_passive_no_intervention":
        errors.append("reference metadata has the wrong case label")
    if case.get("kind") != "kick_then_passive":
        errors.append("reference metadata has the wrong case kind")
    settings = metadata.get("campaign_settings", {})
    if abs(float(settings.get("end_time", math.nan)) - END_TIME) > 1e-12:
        errors.append("reference metadata has the wrong end_time")
    if abs(float(settings.get("output_dt", math.nan)) - OUTPUT_DT) > 1e-12:
        errors.append("reference metadata has the wrong output_dt")
    segments = case.get("segments", [])
    if len(segments) != 2 \
            or segments[0].get("control_kind") != "initial_kick" \
            or segments[1].get("control_kind") != "passive_development":
        errors.append("reference metadata does not describe kick-then-passive startup")
    if errors:
        raise RuntimeError(
            "Invalid passive reference:\n  " + "\n  ".join(errors))
    return {
        "run_directory": str(run_directory),
        "beam_path": str(beam_path),
        "actuator_path": str(actuator_path),
        "metadata_path": str(metadata_path),
        "times": times,
        "tip": tip,
        "actuator": actuator,
        "number_of_beam_samples": len(times),
        "last_time": times[-1],
    }


def report_reference(reference):
    print(f"passive_reference={reference['run_directory']}")
    print(
        f"passive_reference_samples={reference['number_of_beam_samples']}, "
        f"last_time={reference['last_time']:.12g}, output_dt={OUTPUT_DT}"
    )


def audit_experiment(campaign):
    campaign = Path(campaign).resolve()
    experiment_path = campaign / "experiment_metadata.json"
    if not experiment_path.is_file():
        raise RuntimeError(f"Missing experiment metadata: {experiment_path}")
    experiment = json.loads(experiment_path.read_text())
    reference = validate_passive_reference(experiment["passive_reference_run"])
    run_directory = campaign / "runs" / CASE_LABEL
    errors = []
    required = [
        run_directory / "beam_displacement_timeseries.csv",
        run_directory / "actuator_timeseries.csv",
        run_directory / "case_metadata.json",
        run_directory / "mpc_handoff_timeseries.csv",
        run_directory / "fsi2_mpc_handoff_controller.json",
    ]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Controlled run is missing files: {missing}")

    times, tip = read_tip(required[0])
    validate_beam_grid(times, "controlled run", errors)
    actuator = read_actuator(required[1])
    validate_actuator_balance(actuator, "controlled run", errors)
    metadata = json.loads(required[2].read_text())
    validate_controlled_metadata(metadata, reference, errors)
    controller = audit_controller(required[3], errors)

    passive_windows = calculate_windows(reference["times"], reference["tip"])
    controlled_windows = calculate_windows(times, tip)
    ratios = {
        name: controlled_windows[name]["rms"] / passive_windows[name]["rms"]
        for name in passive_windows
    }
    actuator_mismatch = compare_startup_actuators(
        reference["actuator"], actuator, errors)
    tip_mismatch = maximum_tip_mismatch(
        reference["times"], reference["tip"], times, tip, ACTIVATION_TIME)

    no_fallback_after_final_handoff = (
        bool(controller["handoff_times"])
        and not any(time_value > controller["handoff_times"][-1] + 1e-10
                    for time_value in controller["fallback_times"])
    )
    scientific_checks = {
        "at_least_one_handoff": controller["handoff_count"] >= 1,
        "no_nonfinite_failure": controller["nonfinite_fail_safe_count"] == 0,
        "final_mode_is_local": controller["final_mode"] == "local",
        "no_fallback_after_final_handoff": no_fallback_after_final_handoff,
        "late_rms_ratio_below_0p25": ratios[FINAL_WINDOW] < 0.25,
    }
    scientific_success = all(scientific_checks.values())
    integrity_success = not errors
    summary = {
        "integrity_success": integrity_success,
        "integrity_errors": errors,
        "scientific_success": scientific_success,
        "scientific_checks": scientific_checks,
        "experiment": experiment,
        "passive_reference": {
            key: value for key, value in reference.items()
            if key not in ("times", "tip", "actuator")
        },
        "controlled_run_directory": str(run_directory),
        "passive_windows": passive_windows,
        "controlled_windows": controlled_windows,
        "controlled_over_passive_rms": ratios,
        "maximum_startup_actuator_mismatch": actuator_mismatch,
        "maximum_startup_tip_displacement_mismatch": tip_mismatch,
        "controller": controller,
    }
    write_json(campaign / "summary.json", summary)
    print_audit_report(summary)
    if errors:
        raise RuntimeError("MPC handoff experiment failed its integrity audit.")


def audit_controller(path, errors):
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        missing = sorted(REQUIRED_CONTROLLER_COLUMNS - set(reader.fieldnames or []))
        rows = list(reader)
    if missing or not rows:
        errors.append(f"controller log is invalid; missing columns={missing}")
        return empty_controller_report()
    times = [float(row["time"]) for row in rows]
    if any(times[index] < times[index - 1] - 1e-12
           for index in range(1, len(times))):
        errors.append("controller log times move backward")
    if times[0] < ACTIVATION_TIME - 1e-8 or times[-1] > END_TIME + 1e-8:
        errors.append("controller log time range is inconsistent")

    events = [row["event"] for row in rows]
    modes = [row["mode"] for row in rows]
    radii = [float(row["local_radius"]) for row in rows]
    amplitudes = [float(row["A"]) for row in rows]
    handoff_times = [time_value for time_value, event in zip(times, events)
                     if event == "mpc_to_local"]
    fallback_times = [time_value for time_value, event in zip(times, events)
                      if event == "local_to_mpc"]
    nonfinite_times = [time_value for time_value, event in zip(times, events)
                       if event == "nonfinite_fail_safe"]
    residences = local_residences(times, events)
    mpc_radii = [radius for mode, event, radius in zip(modes, events, radii)
                 if (mode == "mpc" or event == "mpc_to_local")
                 and math.isfinite(radius)]
    mpc_amplitudes = [amplitude for mode, amplitude in zip(modes, amplitudes)
                      if mode == "mpc" and math.isfinite(amplitude)]
    return {
        "number_of_log_rows": len(rows),
        "minimum_radius_in_mpc_mode": min(mpc_radii, default=math.nan),
        "handoff_count": len(handoff_times),
        "handoff_times": handoff_times,
        "fallback_count": len(fallback_times),
        "fallback_times": fallback_times,
        "local_residences": residences,
        "local_residence_durations": [item["duration"] for item in residences],
        "maximum_local_residence_time": max(
            (item["duration"] for item in residences), default=0.0),
        "final_mode": modes[-1],
        "maximum_mpc_amplitude": max(mpc_amplitudes, default=math.nan),
        "nonfinite_fail_safe_count": len(nonfinite_times),
        "nonfinite_fail_safe_times": nonfinite_times,
        "maximum_handoff_streak": max(int(row["handoff_streak"]) for row in rows),
    }


def local_residences(times, events):
    residences = []
    start = None
    for time_value, event in zip(times, events):
        if event == "mpc_to_local":
            if start is not None:
                residences.append({
                    "start_time": start,
                    "end_time": time_value,
                    "duration": time_value - start,
                    "ended_by": "new_handoff",
                })
            start = time_value
        elif event in ("local_to_mpc", "nonfinite_fail_safe") and start is not None:
            residences.append({
                "start_time": start,
                "end_time": time_value,
                "duration": time_value - start,
                "ended_by": event,
            })
            start = None
    if start is not None:
        residences.append({
            "start_time": start,
            "end_time": END_TIME,
            "duration": END_TIME - start,
            "ended_by": "end_of_run",
        })
    return residences


def empty_controller_report():
    return {
        "minimum_radius_in_mpc_mode": math.nan,
        "handoff_count": 0,
        "handoff_times": [],
        "fallback_count": 0,
        "fallback_times": [],
        "local_residences": [],
        "local_residence_durations": [],
        "maximum_local_residence_time": 0.0,
        "final_mode": "unknown",
        "maximum_mpc_amplitude": math.nan,
        "nonfinite_fail_safe_count": 0,
        "nonfinite_fail_safe_times": [],
        "maximum_handoff_streak": 0,
    }


def validate_beam_grid(times, label, errors):
    expected_samples = round(END_TIME / OUTPUT_DT) + 1
    if len(times) != expected_samples:
        errors.append(f"{label}: expected {expected_samples} beam samples, got {len(times)}")
    if not times or abs(times[0]) > 1e-10 or abs(times[-1] - END_TIME) > 1e-8:
        errors.append(f"{label}: expected beam time range 0..{END_TIME:g}")
    if any(abs((times[index] - times[index - 1]) - OUTPUT_DT) > 1e-8
           for index in range(1, len(times))):
        errors.append(f"{label}: inconsistent output spacing")


def validate_reference_actuation(actuator, errors):
    upper = actuator["upper"]
    expected_samples = round(END_TIME / 0.002)
    if len(upper) != expected_samples:
        errors.append(
            f"passive reference: expected {expected_samples} upper actuator samples, "
            f"got {len(upper)}")
    maximum_schedule_error = max((abs(
        value - (KICK_VALUE if time_value < KICK_END_TIME - 1e-10 else 0.0)
    ) for time_value, value in upper), default=math.inf)
    if maximum_schedule_error > 1e-12:
        errors.append(
            f"passive reference: kick schedule error {maximum_schedule_error:.3e}")


def validate_controlled_metadata(metadata, reference, errors):
    if metadata.get("label") != CASE_LABEL:
        errors.append("controlled metadata has the wrong label")
    expected = {
        "end_time": END_TIME,
        "output_dt": OUTPUT_DT,
        "kick_value": KICK_VALUE,
        "kick_end_time": KICK_END_TIME,
        "activation_time": ACTIVATION_TIME,
    }
    for name, value in expected.items():
        if abs(float(metadata.get(name, math.nan)) - value) > 1e-12:
            errors.append(f"controlled metadata has the wrong {name}")
    if Path(metadata.get("passive_reference_run", "")).resolve() \
            != Path(reference["run_directory"]):
        errors.append("controlled metadata points to the wrong passive reference")


def read_tip(path):
    with path.open(newline="") as input_file:
        reader = csv.reader(input_file)
        next(reader)
        header = next(reader)
        tip_index = header.index("tip_DISPLACEMENT_Y")
        rows = list(reader)
    return (
        [float(row[0]) for row in rows],
        [float(row[tip_index]) for row in rows],
    )


def read_actuator(path):
    result = {"upper": [], "lower": []}
    with path.open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            name = row["actuator_name"]
            if name.endswith("_upper"):
                key = "upper"
            elif name.endswith("_lower"):
                key = "lower"
            else:
                raise RuntimeError(f"Unknown actuator name: {name}")
            result[key].append((float(row["time"]), float(row["control_value"])))
    return result


def validate_actuator_balance(actuator, label, errors):
    upper = {round(time_value, 10): value
             for time_value, value in actuator["upper"]}
    lower = {round(time_value, 10): value
             for time_value, value in actuator["lower"]}
    if upper.keys() != lower.keys():
        errors.append(f"{label}: upper/lower actuator time grids differ")
        return
    imbalance = max((abs(upper[time_value] + lower[time_value])
                     for time_value in upper), default=math.inf)
    if imbalance > 1e-10:
        errors.append(f"{label}: actuator imbalance {imbalance:.3e}")


def compare_startup_actuators(reference, controlled, errors):
    maximum_error = 0.0
    for name in ("upper", "lower"):
        passive = {round(time_value, 10): value
                   for time_value, value in reference[name]
                   if time_value < ACTIVATION_TIME - 1e-10}
        active = {round(time_value, 10): value
                  for time_value, value in controlled[name]
                  if time_value < ACTIVATION_TIME - 1e-10}
        if passive.keys() != active.keys():
            errors.append(f"pre-control {name} actuator time grids differ")
            maximum_error = math.inf
            continue
        maximum_error = max(maximum_error, max(
            (abs(passive[time_value] - active[time_value])
             for time_value in passive), default=0.0))
    if maximum_error > 1e-12:
        errors.append(f"pre-control actuator mismatch {maximum_error:.3e}")
    return maximum_error


def maximum_tip_mismatch(reference_times, reference_tip, controlled_times,
                         controlled_tip, end_time):
    passive = {round(time_value, 10): value
               for time_value, value in zip(reference_times, reference_tip)
               if time_value <= end_time + 1e-10}
    active = {round(time_value, 10): value
              for time_value, value in zip(controlled_times, controlled_tip)
              if time_value <= end_time + 1e-10}
    if passive.keys() != active.keys():
        return math.inf
    return max((abs(passive[time_value] - active[time_value])
                for time_value in passive), default=math.inf)


def calculate_windows(times, values):
    return {
        f"{start:g}_{end:g}": window_stats(times, values, start, end)
        for start, end in WINDOWS
    }


def window_stats(times, values, start, end):
    selected = [value for time_value, value in zip(times, values)
                if start - 1e-10 <= time_value <= end + 1e-10]
    if not selected:
        raise RuntimeError(f"No samples in window {start:g}..{end:g}")
    mean = sum(selected) / len(selected)
    return {
        "rms": math.sqrt(sum(value * value for value in selected) / len(selected)),
        "mean": mean,
        "peak_to_peak": max(selected) - min(selected),
        "number_of_samples": len(selected),
    }


def print_audit_report(summary):
    print(f"Integrity success: {summary['integrity_success']}")
    for name in summary["passive_windows"]:
        passive = summary["passive_windows"][name]
        controlled = summary["controlled_windows"][name]
        ratio = summary["controlled_over_passive_rms"][name]
        print(
            f"  {name}s: passive(rms={passive['rms']:.6g}, "
            f"mean={passive['mean']:.6g}, p2p={passive['peak_to_peak']:.6g}); "
            f"controlled(rms={controlled['rms']:.6g}, "
            f"mean={controlled['mean']:.6g}, "
            f"p2p={controlled['peak_to_peak']:.6g}); ratio={ratio:.6g}"
        )
    print(
        "Startup comparison: actuator_mismatch={:.3e}, tip_mismatch={:.3e}".format(
            summary["maximum_startup_actuator_mismatch"],
            summary["maximum_startup_tip_displacement_mismatch"],
        ))
    controller = summary["controller"]
    print(
        f"Controller: minimum_mpc_radius={controller['minimum_radius_in_mpc_mode']:.6g}, "
        f"handoffs={controller['handoff_times']}, "
        f"fallbacks={controller['fallback_times']}, "
        f"local_residences={controller['local_residence_durations']}, "
        f"max_local_residence={controller['maximum_local_residence_time']:.6g}, "
        f"final_mode={controller['final_mode']}, "
        f"max_mpc_A={controller['maximum_mpc_amplitude']:.6g}, "
        f"nonfinite_events={controller['nonfinite_fail_safe_times']}"
    )
    print(f"Scientific checks: {summary['scientific_checks']}")
    print(f"Scientific success: {summary['scientific_success']}")


def read_bool(name, default):
    value = os.environ.get(name)
    return default if value is None else value.lower() not in ("0", "false", "no", "off")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


if __name__ == "__main__":
    main()
