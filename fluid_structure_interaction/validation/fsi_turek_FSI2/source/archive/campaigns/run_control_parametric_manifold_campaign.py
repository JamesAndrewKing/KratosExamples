"""Run scalar-control parametric-manifold data campaigns for FSI2.

The data is meant for reduced models of the extended controlled system

    x_dot = f(x, u)
    u_dot = epsilon

with frozen-control manifolds x = W(eta, u), eta_dot = R(eta, u).

This is not a harmonic/phase-parametric forcing campaign. The scalar command
control_u is applied directly to the mass-balanced Rabault actuator pair:

    upper control = u
    lower control = -u
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import run_control_identification_campaign as base


DEFAULT_CAMPAIGN_LABEL = "fsi_control_parametric_manifold_t40_u04"
DEFAULT_SMOKE_LABEL = "fsi_control_parametric_manifold_t40_u04_smoke"
DEFAULT_U_MAX = 0.4
DEFAULT_END_TIME = 40.0
DEFAULT_SMOKE_END_TIME = 5.0
DEFAULT_OUTPUT_INTERVAL = 0.01
DEFAULT_SIGNAL_DT = 0.002
DEFAULT_WRITE_PARAVIEW = False
DEFAULT_CONSTANT_COUNT = 13
DEFAULT_INCLUDE_FAST_RAMPS = True
DEFAULT_INCLUDE_ZOH = True
DEFAULT_MAX_REASONABLE_TIP_DISPLACEMENT = 0.5

RAMP_DIRECTIONS = [
    (-0.4, 0.4),
    (0.4, -0.4),
    (0.0, 0.4),
    (0.0, -0.4),
    (0.4, 0.0),
    (-0.4, 0.0),
]
TRAIN_RANDOM_SEEDS = [1, 2, 3, 4]
VALIDATION_RANDOM_SEEDS = [101, 102]
CHALLENGE_RANDOM_SEEDS = [201, 202]
ZOH_TRAIN_SEEDS = [51, 52]
ZOH_VALIDATION_SEEDS = [151]
INTERVENTION_TARGETS = [-0.4, -0.3, -0.2, 0.2, 0.3, 0.4]
INTERVENTION_RAMP_DURATIONS = [2.0, 5.0]

PYTHONPATH = os.environ.get("KRATOS_FSI_PYTHONPATH")
DYLD_LIBRARY_PATH = os.environ.get("KRATOS_FSI_DYLD_LIBRARY_PATH")
LD_LIBRARY_PATH = os.environ.get("KRATOS_FSI_LD_LIBRARY_PATH")

SUMMARY_FIELDNAMES = [
    "label",
    "role",
    "kind",
    "run_directory",
    "identification_snapshots",
    "end_time",
    "output_dt",
    "input_dt",
    "u_max",
    "u_value",
    "u_start",
    "u_end",
    "ramp_duration",
    "random_seed",
    "t_intervention",
    "u_target",
    "intervention_type",
    "max_abs_u",
    "max_abs_u_dot",
    "min_u",
    "max_u",
    "last_time",
    "number_of_samples",
    "max_abs_tip_y",
    "rms_tip_y",
    "final_tip_y",
    "max_abs_control",
    "rms_control",
]

REQUIRED_IDENTIFICATION_COLUMNS = [
    "time",
    "control_u",
    "control_u_dot",
    "control_segment_id",
    "time_since_switch",
    "is_switch_sample",
    "control_kind",
]


def main():
    args = parse_arguments()
    source_directory = Path(__file__).resolve().parent
    os.chdir(source_directory)

    if args.audit is not None:
        campaign_directory = Path(args.audit)
        manifest = read_manifest(campaign_directory)
        results = read_case_results(campaign_directory)
        write_summary_files(campaign_directory, results, manifest)
        audit_campaign(campaign_directory, manifest, results)
        print_audit_report(results, manifest)
        return

    smoke = args.smoke or read_bool("KRATOS_FSI_PM_SMOKE", False)
    settings = read_settings(smoke)
    cases = build_smoke_cases(settings) if smoke else build_case_library(settings)

    if args.list_cases:
        for index, case in enumerate(cases):
            print(f"{index:03d} {case['label']} [{case['role']}/{case['kind']}]")
        print(f"number_of_cases={len(cases)}")
        return

    if args.dry_run is not None:
        settings["dry_run"] = True
        campaign_directory = Path(args.dry_run)
        run_dry_campaign(campaign_directory, settings, cases)
        manifest = read_manifest(campaign_directory)
        results = read_case_results(campaign_directory)
        write_summary_files(campaign_directory, results, manifest)
        audit_campaign(campaign_directory, manifest, results)
        print_audit_report(results, manifest)
        return

    campaign_directory = Path("run_outputs") / settings["campaign_label"]
    campaign_directory.mkdir(parents=True, exist_ok=True)
    write_manifest_once(campaign_directory, settings, cases)

    selected_cases = select_cases(cases)
    for index, case in selected_cases:
        print(f"[{index}] {case['label']}", flush=True)
        write_case_selection(campaign_directory, index, case)
        run_directory = run_case(case, campaign_directory, settings)
        result = collect_case_result(case, campaign_directory, run_directory, settings)
        write_case_result(campaign_directory, result)
        print(format_result(result), flush=True)

    if len(selected_cases) == len(cases):
        manifest = make_manifest(settings, cases)
        results = read_case_results(campaign_directory)
        write_summary_files(campaign_directory, results, manifest)
        audit_campaign(campaign_directory, manifest, results)
        print_audit_report(results, manifest)

    print(f"campaign={campaign_directory.resolve()}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run or audit the FSI2 scalar-control parametric-manifold campaign."
    )
    parser.add_argument("--dry-run", type=Path, help="Generate synthetic files and audit them.")
    parser.add_argument("--audit", type=Path, help="Audit an existing campaign directory.")
    parser.add_argument("--list-cases", action="store_true", help="List case indices and labels.")
    parser.add_argument("--smoke", action="store_true", help="Use the four short smoke-test cases.")
    return parser.parse_args()


def read_settings(smoke):
    label_default = DEFAULT_SMOKE_LABEL if smoke else DEFAULT_CAMPAIGN_LABEL
    end_time_default = DEFAULT_SMOKE_END_TIME if smoke else DEFAULT_END_TIME
    return {
        "campaign_label": os.environ.get("KRATOS_FSI_PM_LABEL", label_default),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "u_max": read_float("KRATOS_FSI_PM_U_MAX", DEFAULT_U_MAX),
        "end_time": read_float("KRATOS_FSI_PM_END_TIME", end_time_default),
        "output_dt": read_float("KRATOS_FSI_PM_OUTPUT_INTERVAL", DEFAULT_OUTPUT_INTERVAL),
        "input_dt": read_float("KRATOS_FSI_PM_SIGNAL_DT", DEFAULT_SIGNAL_DT),
        "write_paraview": read_bool("KRATOS_FSI_PM_WRITE_PARAVIEW", DEFAULT_WRITE_PARAVIEW),
        "constant_count": read_int("KRATOS_FSI_PM_CONSTANT_COUNT", DEFAULT_CONSTANT_COUNT),
        "include_fast_ramps": read_bool("KRATOS_FSI_PM_INCLUDE_FAST_RAMPS", DEFAULT_INCLUDE_FAST_RAMPS),
        "include_zoh": read_bool("KRATOS_FSI_PM_INCLUDE_ZOH", DEFAULT_INCLUDE_ZOH),
        "max_reasonable_tip_displacement": read_float(
            "KRATOS_FSI_PM_MAX_REASONABLE_TIP_DISPLACEMENT",
            DEFAULT_MAX_REASONABLE_TIP_DISPLACEMENT,
        ),
        "smoke": smoke,
    }


def build_case_library(settings):
    cases = [make_passive_case(settings)]
    cases.extend(make_constant_cases(settings))
    cases.extend(make_smooth_ramp_cases(settings))
    cases.extend(make_smooth_random_cases(settings))
    cases.extend(make_intervention_cases(settings))
    if settings["include_zoh"]:
        cases.extend(make_zoh_cases(settings))
    return cases


def build_smoke_cases(settings):
    u_max = settings["u_max"]
    end_time = settings["end_time"]
    return [
        make_constant_case(settings, u_max, label="smoke_constant_u_p0p400", role="smoke"),
        make_constant_case(settings, -u_max, label="smoke_constant_u_m0p400", role="smoke"),
        make_case(
            label="smoke_smooth_ramp_m0p400_to_p0p400",
            role="smoke",
            kind="smooth_ramp",
            metadata={
                "u_start": -u_max,
                "u_end": u_max,
                "ramp_duration": min(3.0, end_time),
            },
            segments=[
                ramp_segment(0, 0.0, min(3.0, end_time), -u_max, u_max, "smooth_ramp_transition"),
                hold_segment(1, min(3.0, end_time), end_time, u_max, "smooth_ramp_hold"),
            ],
        ),
        make_smooth_random_case(
            settings,
            role="smoke",
            seed=999,
            label="smoke_smooth_random_seed999",
            min_segment_duration=0.8,
            max_segment_duration=1.2,
            min_transition_duration=0.2,
            max_transition_duration=0.4,
        ),
    ]


def make_passive_case(settings):
    end_time = settings["end_time"]
    return make_case(
        label="passive_baseline",
        role="passive",
        kind="passive",
        metadata={"u_value": 0.0},
        segments=[hold_segment(0, 0.0, end_time, 0.0, "passive")],
    )


def make_constant_cases(settings):
    count = settings["constant_count"]
    if count < 9:
        raise ValueError("KRATOS_FSI_PM_CONSTANT_COUNT must be at least 9.")
    u_max = settings["u_max"]
    return [
        make_constant_case(settings, value)
        for value in linspace(-u_max, u_max, count)
    ]


def make_constant_case(settings, value, label=None, role="train"):
    value = round(float(value), 12)
    end_time = settings["end_time"]
    return make_case(
        label=label or f"constant_u_{format_label_value(value)}",
        role=role,
        kind="constant_u",
        metadata={"u_value": value},
        segments=[hold_segment(0, 0.0, end_time, value, "constant_u")],
    )


def make_smooth_ramp_cases(settings):
    durations = [10.0, 20.0]
    if settings["include_fast_ramps"]:
        durations.insert(0, 5.0)
    cases = []
    for duration in durations:
        role = "validation" if duration == 20.0 else "train"
        for u_start_factor, u_end_factor in RAMP_DIRECTIONS:
            u_start = u_start_factor / DEFAULT_U_MAX * settings["u_max"]
            u_end = u_end_factor / DEFAULT_U_MAX * settings["u_max"]
            label = (
                f"smooth_ramp_{format_label_value(u_start)}_to_"
                f"{format_label_value(u_end)}_T{duration:g}"
            )
            cases.append(make_ramp_case(settings, label, role, u_start, u_end, duration))
    return cases


def make_ramp_case(settings, label, role, u_start, u_end, ramp_duration):
    end_time = settings["end_time"]
    hold_before = min(5.0, end_time)
    ramp_start = hold_before
    ramp_end = min(end_time, ramp_start + ramp_duration)
    segments = [
        hold_segment(0, 0.0, ramp_start, u_start, "smooth_ramp_hold"),
        ramp_segment(1, ramp_start, ramp_end, u_start, u_end, "smooth_ramp_transition"),
    ]
    if ramp_end < end_time:
        segments.append(hold_segment(2, ramp_end, end_time, u_end, "smooth_ramp_hold"))
    return make_case(
        label=label,
        role=role,
        kind="smooth_ramp",
        metadata={
            "u_start": round(u_start, 12),
            "u_end": round(u_end, 12),
            "ramp_duration": ramp_duration,
            "max_abs_u_dot": max_abs_ramp_derivative(u_start, u_end, ramp_duration),
        },
        segments=segments,
    )


def make_smooth_random_cases(settings):
    cases = []
    for seed in TRAIN_RANDOM_SEEDS:
        cases.append(make_smooth_random_case(settings, "train", seed, f"train_smooth_random_seed{seed:02d}"))
    for seed in VALIDATION_RANDOM_SEEDS:
        cases.append(make_smooth_random_case(settings, "validation", seed, f"validation_smooth_random_seed{seed:03d}"))
    for seed in CHALLENGE_RANDOM_SEEDS:
        cases.append(make_smooth_random_case(settings, "challenge", seed, f"release_challenge_smooth_random_seed{seed:03d}"))
    return cases


def make_smooth_random_case(
    settings,
    role,
    seed,
    label,
    min_segment_duration=1.5,
    max_segment_duration=3.0,
    min_transition_duration=0.3,
    max_transition_duration=0.8,
):
    rng = random.Random(seed)
    end_time = settings["end_time"]
    u_max = settings["u_max"]
    current_time = 0.0
    current_u = 0.0
    segment_id = 0
    segments = []
    while current_time < end_time - 1e-12:
        total_segment_duration = rng.uniform(min_segment_duration, max_segment_duration)
        transition_duration = min(
            rng.uniform(min_transition_duration, max_transition_duration),
            0.8 * total_segment_duration,
        )
        hold_duration = max(0.0, total_segment_duration - transition_duration)
        hold_end = min(end_time, quantize_time(current_time + hold_duration, settings["output_dt"]))
        if hold_end > current_time + 1e-12:
            segments.append(hold_segment(segment_id, current_time, hold_end, current_u, "smooth_random_hold"))
            segment_id += 1
            current_time = hold_end
        if current_time >= end_time - 1e-12:
            break
        next_u = rng.uniform(-u_max, u_max)
        transition_end = min(end_time, quantize_time(current_time + transition_duration, settings["output_dt"]))
        if transition_end <= current_time + 1e-12:
            transition_end = min(end_time, quantize_time(current_time + settings["output_dt"], settings["output_dt"]))
        segments.append(ramp_segment(segment_id, current_time, transition_end, current_u, next_u, "smooth_random_transition"))
        segment_id += 1
        current_u = next_u
        current_time = transition_end

    return make_case(
        label=label,
        role=role,
        kind="smooth_random",
        metadata={
            "random_seed": seed,
            "min_segment_duration": min_segment_duration,
            "max_segment_duration": max_segment_duration,
            "min_transition_duration": min_transition_duration,
            "max_transition_duration": max_transition_duration,
            "max_abs_u_dot": max_abs_case_u_dot(segments),
        },
        segments=segments,
    )


def make_intervention_cases(settings):
    cases = []
    for target in INTERVENTION_TARGETS:
        role = "challenge" if abs(target) >= 0.4 - 1e-12 else "validation"
        label = f"intervention_constant_u_{format_label_value(target)}"
        cases.append(make_intervention_case(settings, label, role, target, "constant", 0.0))
    for duration in INTERVENTION_RAMP_DURATIONS:
        for target in INTERVENTION_TARGETS:
            role = "challenge" if duration == 5.0 or abs(target) >= 0.4 - 1e-12 else "validation"
            label = f"intervention_ramp_u_{format_label_value(target)}_T{duration:g}"
            cases.append(make_intervention_case(settings, label, role, target, "ramp", duration))
    return cases


def make_intervention_case(settings, label, role, target, intervention_type, ramp_duration):
    end_time = settings["end_time"]
    t_intervention = min(20.0, end_time)
    segments = [hold_segment(0, 0.0, t_intervention, 0.0, "intervention_passive")]
    if intervention_type == "constant":
        segments.append(hold_segment(1, t_intervention, end_time, target, "intervention_constant"))
    else:
        ramp_end = min(end_time, t_intervention + ramp_duration)
        segments.append(ramp_segment(1, t_intervention, ramp_end, 0.0, target, "intervention_ramp_transition"))
        if ramp_end < end_time:
            segments.append(hold_segment(2, ramp_end, end_time, target, "intervention_ramp_hold"))
    return make_case(
        label=label,
        role=role,
        kind="limit_cycle_intervention",
        metadata={
            "t_intervention": t_intervention,
            "u_target": round(target, 12),
            "ramp_duration": ramp_duration if intervention_type == "ramp" else "",
            "intervention_type": intervention_type,
            "max_abs_u_dot": max_abs_case_u_dot(segments),
        },
        segments=segments,
    )


def make_zoh_cases(settings):
    cases = []
    for seed in ZOH_TRAIN_SEEDS:
        cases.append(make_zoh_case(settings, "train", seed, f"train_zoh_switch_seed{seed:02d}"))
    for seed in ZOH_VALIDATION_SEEDS:
        cases.append(make_zoh_case(settings, "validation", seed, f"validation_zoh_switch_seed{seed:03d}"))
    return cases


def make_zoh_case(settings, role, seed, label):
    rng = random.Random(seed)
    end_time = settings["end_time"]
    u_max = settings["u_max"]
    levels = [-u_max, -0.5 * u_max, 0.0, 0.5 * u_max, u_max]
    current_time = 0.0
    previous_u = 0.0
    segment_id = 0
    segments = []
    while current_time < end_time - 1e-12:
        candidates = [value for value in levels if abs(value - previous_u) > 1e-12]
        value = rng.choice(candidates)
        duration = rng.uniform(1.5, 3.0)
        next_time = min(end_time, quantize_time(current_time + duration, settings["output_dt"]))
        if next_time <= current_time:
            next_time = min(end_time, quantize_time(current_time + settings["output_dt"], settings["output_dt"]))
        segments.append(hold_segment(segment_id, current_time, next_time, value, "zoh_switching"))
        previous_u = value
        current_time = next_time
        segment_id += 1
    return make_case(
        label=label,
        role=role,
        kind="zoh_switching",
        metadata={
            "random_seed": seed,
            "u_max": u_max,
            "min_segment_duration": 1.5,
            "max_segment_duration": 3.0,
        },
        segments=segments,
        csv_interpolation="zoh",
    )


def make_case(label, role, kind, metadata, segments, csv_interpolation="linear"):
    return {
        "label": label,
        "role": role,
        "kind": kind,
        "controller": "csv",
        "csv_interpolation": csv_interpolation,
        "metadata": metadata,
        "segments": normalize_segments(segments),
    }


def normalize_segments(segments):
    normalized = []
    for index, segment in enumerate(segments):
        if segment["end_time"] <= segment["start_time"] + 1e-12:
            continue
        segment = dict(segment)
        segment["segment_id"] = index
        normalized.append(segment)
    return normalized


def hold_segment(segment_id, start_time, end_time, value, control_kind):
    return {
        "segment_id": segment_id,
        "start_time": round(float(start_time), 12),
        "end_time": round(float(end_time), 12),
        "profile": "hold",
        "u_start": round(float(value), 12),
        "u_end": round(float(value), 12),
        "control_kind": control_kind,
    }


def ramp_segment(segment_id, start_time, end_time, u_start, u_end, control_kind):
    return {
        "segment_id": segment_id,
        "start_time": round(float(start_time), 12),
        "end_time": round(float(end_time), 12),
        "profile": "cosine_ramp",
        "u_start": round(float(u_start), 12),
        "u_end": round(float(u_end), 12),
        "control_kind": control_kind,
    }


def run_case(case, campaign_directory, settings):
    run_directory = campaign_directory / "runs" / case["label"]
    base.ensure_empty_directory(run_directory)
    input_path = campaign_directory / "inputs" / f"{case['label']}.csv"
    log_path = campaign_directory / "logs" / f"{case['label']}.log"
    base.ensure_file_does_not_exist(log_path)
    base.ensure_file_does_not_exist(campaign_directory / "case_results" / f"{case['label']}.json")
    write_input_timeseries_csv(input_path, case, settings)

    environment = os.environ.copy()
    environment.update({
        "KRATOS_FSI_RUN_LABEL": case["label"],
        "KRATOS_FSI_RUN_OUTPUT_DIRECTORY": str(run_directory.resolve()),
        "KRATOS_FSI_END_TIME": str(settings["end_time"]),
        "KRATOS_FSI_OUTPUT_INTERVAL": str(settings["output_dt"]),
        "KRATOS_FSI_WRITE_PARAVIEW": "1" if settings["write_paraview"] else "0",
        "KRATOS_FSI_CONTROLLER_TYPE": "csv",
        "KRATOS_FSI_ACTUATOR_CSV_FILE": str(input_path.resolve()),
        "KRATOS_FSI_ACTUATOR_CSV_TIME_COLUMN": "time",
        "KRATOS_FSI_ACTUATOR_CSV_VALUE_COLUMN": "value",
        "KRATOS_FSI_ACTUATOR_CSV_INTERPOLATION": case["csv_interpolation"],
        "KRATOS_FSI_ACTUATOR_AMPLITUDE": "0.0",
        "KRATOS_FSI_ACTUATOR_FREQUENCY": "0.0",
        "KRATOS_FSI_ACTUATOR_PHASE": "0.0",
    })
    if PYTHONPATH:
        environment["PYTHONPATH"] = PYTHONPATH
    if DYLD_LIBRARY_PATH:
        environment["DYLD_LIBRARY_PATH"] = DYLD_LIBRARY_PATH
    if LD_LIBRARY_PATH:
        environment["LD_LIBRARY_PATH"] = LD_LIBRARY_PATH

    completed = subprocess.run(
        [sys.executable, "MainKratos.py"],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base.write_text_no_overwrite(log_path, completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(f"Run failed for {case['label']}. See {log_path}.")

    shutil.copyfile(input_path, run_directory / "input_timeseries.csv")
    write_identification_snapshots_csv(
        run_directory / "beam_displacement_timeseries.csv",
        run_directory / "input_timeseries.csv",
        run_directory / "identification_snapshots.csv",
    )
    write_case_metadata(case, settings, campaign_directory, input_path, run_directory, log_path, completed.returncode)
    return run_directory


def run_dry_campaign(campaign_directory, settings, cases):
    base.ensure_empty_directory(campaign_directory)
    write_manifest_once(campaign_directory, settings, cases)
    for index, case in enumerate(cases):
        print(f"[dry {index}] {case['label']}", flush=True)
        run_directory = campaign_directory / "runs" / case["label"]
        base.ensure_empty_directory(run_directory)
        input_path = campaign_directory / "inputs" / f"{case['label']}.csv"
        write_input_timeseries_csv(input_path, case, settings)
        shutil.copyfile(input_path, run_directory / "input_timeseries.csv")
        base.write_synthetic_beam_displacements(run_directory / "beam_displacement_timeseries.csv", {
            "end_time": settings["end_time"],
            "output_interval": settings["output_dt"],
        })
        base.write_synthetic_actuator_timeseries(run_directory / "actuator_timeseries.csv", input_path)
        write_identification_snapshots_csv(
            run_directory / "beam_displacement_timeseries.csv",
            run_directory / "input_timeseries.csv",
            run_directory / "identification_snapshots.csv",
        )
        log_path = campaign_directory / "logs" / f"{case['label']}.log"
        base.write_text_no_overwrite(log_path, "dry run: Kratos was not launched\n")
        write_case_metadata(case, settings, campaign_directory, input_path, run_directory, log_path, 0)
        result = collect_case_result(case, campaign_directory, run_directory, settings)
        write_case_result(campaign_directory, result)


def write_input_timeseries_csv(path, case, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    base.ensure_file_does_not_exist(path)
    schedule = SegmentSchedule(case["segments"])
    switch_times = {
        round(segment["start_time"], 12)
        for segment in case["segments"]
        if segment["segment_id"] > 0
    }
    with path.open("x", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "time",
            "value",
            "u_dot",
            "segment_id",
            "time_since_switch",
            "is_switch_sample",
            "control_kind",
        ])
        for time in base.iter_sample_times(settings["end_time"], settings["input_dt"]):
            segment = schedule.lookup(time)
            value, u_dot = evaluate_segment(segment, time)
            writer.writerow([
                base.format_float(time),
                base.format_float(value),
                base.format_float(u_dot),
                segment["segment_id"],
                base.format_float(max(0.0, time - segment["start_time"])),
                int(round(time, 12) in switch_times),
                segment["control_kind"],
            ])


def write_identification_snapshots_csv(beam_path, input_path, output_path):
    input_rows = base.read_input_rows_by_time(input_path)
    input_rows_by_time = {round(row["time_float"], 12): row for row in input_rows}
    with beam_path.open(newline="") as beam_file:
        beam_reader = csv.reader(beam_file)
        metadata = next(beam_reader)
        beam_header = next(beam_reader)
        beam_rows = list(beam_reader)

    measurement_columns = beam_header[1:]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.ensure_file_does_not_exist(output_path)
    with output_path.open("x", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "time",
            *[f"measurement_{name}" for name in measurement_columns],
            "control_u",
            "control_u_dot",
            "control_segment_id",
            "time_since_switch",
            "is_switch_sample",
            "control_kind",
        ])
        for beam_row in beam_rows:
            time = float(beam_row[0])
            input_row = base.find_input_row(input_rows, input_rows_by_time, time)
            writer.writerow([
                base.format_float(time),
                *beam_row[1:],
                input_row["value"],
                input_row["u_dot"],
                input_row["segment_id"],
                input_row["time_since_switch"],
                input_row["is_switch_sample"],
                input_row["control_kind"],
            ])

    metadata_path = output_path.with_suffix(".metadata.csv")
    base.ensure_file_does_not_exist(metadata_path)
    with metadata_path.open("x", newline="") as metadata_file:
        writer = csv.writer(metadata_file)
        writer.writerow(metadata)


def evaluate_segment(segment, time):
    if segment["profile"] == "hold":
        return segment["u_start"], 0.0
    if segment["profile"] != "cosine_ramp":
        raise ValueError(f"Unsupported control segment profile {segment['profile']!r}.")

    duration = segment["end_time"] - segment["start_time"]
    if duration <= 0.0:
        return segment["u_end"], 0.0
    tau = min(1.0, max(0.0, (time - segment["start_time"]) / duration))
    shape = 0.5 * (1.0 - math.cos(math.pi * tau))
    shape_dot = 0.5 * math.pi * math.sin(math.pi * tau) / duration
    delta = segment["u_end"] - segment["u_start"]
    return segment["u_start"] + delta * shape, delta * shape_dot


class SegmentSchedule:
    def __init__(self, segments):
        self.segments = segments
        self.starts = [segment["start_time"] for segment in segments]

    def lookup(self, time):
        index = bisect_right(self.starts, round(time, 12)) - 1
        if index < 0:
            return self.segments[0]
        return self.segments[index]


def collect_case_result(case, campaign_directory, run_directory, settings):
    metrics = collect_metrics(run_directory)
    input_metrics = collect_input_metrics(run_directory / "input_timeseries.csv")
    metadata = case["metadata"]
    return {
        "label": case["label"],
        "role": case["role"],
        "kind": case["kind"],
        "end_time": settings["end_time"],
        "output_dt": settings["output_dt"],
        "input_dt": settings["input_dt"],
        "u_max": settings["u_max"],
        "u_value": metadata.get("u_value", ""),
        "u_start": metadata.get("u_start", ""),
        "u_end": metadata.get("u_end", ""),
        "ramp_duration": metadata.get("ramp_duration", ""),
        "random_seed": metadata.get("random_seed", ""),
        "t_intervention": metadata.get("t_intervention", ""),
        "u_target": metadata.get("u_target", ""),
        "intervention_type": metadata.get("intervention_type", ""),
        **input_metrics,
        **metrics,
        "run_directory": str(run_directory.resolve()),
        "input_timeseries": str((run_directory / "input_timeseries.csv").resolve()),
        "identification_snapshots": str((run_directory / "identification_snapshots.csv").resolve()),
        "case_result_path": str((campaign_directory / "case_results" / f"{case['label']}.json").resolve()),
    }


def collect_metrics(run_directory):
    beam_path = run_directory / "beam_displacement_timeseries.csv"
    actuator_path = run_directory / "actuator_timeseries.csv"
    metrics = {
        "last_time": 0.0,
        "number_of_samples": 0,
        "max_abs_tip_y": 0.0,
        "rms_tip_y": 0.0,
        "final_tip_y": 0.0,
        "max_abs_control": 0.0,
        "rms_control": 0.0,
    }
    tip_square_sum = 0.0
    with beam_path.open(newline="") as beam_file:
        reader = csv.reader(beam_file)
        next(reader)
        header = next(reader)
        tip_y_index = header.index("tip_DISPLACEMENT_Y")
        for row in reader:
            time = float(row[0])
            tip_y = float(row[tip_y_index])
            metrics["last_time"] = time
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


def collect_input_metrics(input_path):
    max_abs_u = 0.0
    max_abs_u_dot = 0.0
    min_u = float("inf")
    max_u = -float("inf")
    with input_path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            value = float(row["value"])
            u_dot = float(row["u_dot"])
            max_abs_u = max(max_abs_u, abs(value))
            max_abs_u_dot = max(max_abs_u_dot, abs(u_dot))
            min_u = min(min_u, value)
            max_u = max(max_u, value)
    return {
        "max_abs_u": max_abs_u,
        "max_abs_u_dot": max_abs_u_dot,
        "min_u": min_u,
        "max_u": max_u,
    }


def write_case_metadata(case, settings, campaign_directory, input_path, run_directory, log_path, return_code):
    metadata = {
        "case": case,
        "campaign_settings": {
            "u_max": settings["u_max"],
            "end_time": settings["end_time"],
            "output_dt": settings["output_dt"],
            "input_dt": settings["input_dt"],
            "control_convention": "upper control = u, lower control = -u",
            "modeling_goal": "x_dot = f(x,u), u_dot = epsilon with frozen-u manifolds",
        },
        "input_path": str(input_path.resolve()),
        "signal_path": str(input_path.resolve()),
        "identification_snapshots_path": str((run_directory / "identification_snapshots.csv").resolve()),
        "return_code": return_code,
        "log_path": str(log_path.resolve()),
    }
    base.write_text_no_overwrite(run_directory / "case_metadata.json", json.dumps(metadata, indent=4))


def write_case_result(campaign_directory, result):
    result_path = campaign_directory / "case_results" / f"{result['label']}.json"
    base.write_text_no_overwrite(result_path, json.dumps(result, indent=4))


def write_summary_files(campaign_directory, results, manifest):
    ordered = sort_results(results, manifest)
    (campaign_directory / "summary.json").write_text(json.dumps(ordered, indent=4))
    with (campaign_directory / "summary.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=SUMMARY_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    print(f"wrote {campaign_directory / 'summary.csv'}")
    print(f"wrote {campaign_directory / 'summary.json'}")


def audit_campaign(campaign_directory, manifest, results):
    errors = []
    expected_cases = manifest["cases"]
    expected_labels = [case["label"] for case in expected_cases]
    result_by_label = {result["label"]: result for result in results}
    missing_results = [label for label in expected_labels if label not in result_by_label]
    if missing_results:
        errors.append(f"missing case results: {missing_results}")

    run_dirs = [result.get("run_directory", "") for result in results]
    duplicates = [item for item, count in Counter(run_dirs).items() if item and count > 1]
    if duplicates:
        errors.append(f"duplicate run directories: {duplicates}")

    expected_rows = int(round(float(manifest["end_time"]) / float(manifest["output_dt"]))) + 1
    input_dt = float(manifest["input_dt"])
    u_max = float(manifest["u_max"])
    for case in expected_cases:
        result = result_by_label.get(case["label"])
        if result is None:
            continue
        run_directory = Path(result["run_directory"])
        if not run_directory.exists():
            run_directory = campaign_directory / "runs" / case["label"]
        errors.extend(audit_case(
            campaign_directory,
            run_directory,
            case,
            result,
            expected_rows,
            input_dt,
            u_max,
            float(manifest["max_reasonable_tip_displacement"]),
        ))

    if errors:
        print("Campaign audit failed:")
        for error in errors:
            print(f"  ERROR: {error}")
        raise RuntimeError("Parametric-manifold campaign audit failed.")
    print("Campaign audit passed:")
    print(f"  expected cases completed: {len(expected_labels)}")
    print("  files, rows, schemas, controls, metadata, and actuator balance are OK")


def audit_case(campaign_directory, run_directory, case, result, expected_rows, input_dt, u_max, max_reasonable_tip_displacement):
    errors = []
    required_files = [
        run_directory / "input_timeseries.csv",
        run_directory / "identification_snapshots.csv",
        run_directory / "beam_displacement_timeseries.csv",
        run_directory / "actuator_timeseries.csv",
        run_directory / "case_metadata.json",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        return [f"{case['label']}: missing files {missing}"]

    input_rows = base.read_input_rows_by_time(run_directory / "input_timeseries.csv")
    identification_rows = base.read_csv_dicts(run_directory / "identification_snapshots.csv")
    if len(identification_rows) != expected_rows:
        errors.append(f"{case['label']}: identification rows {len(identification_rows)} != {expected_rows}")
    if abs(float(identification_rows[0]["time"])) > 1e-12:
        errors.append(f"{case['label']}: first identification time is not 0")
    if abs(float(identification_rows[-1]["time"]) - float(result["end_time"])) > 5e-9:
        errors.append(f"{case['label']}: final identification time is not end_time")

    header = identification_rows[0].keys()
    missing_columns = [column for column in REQUIRED_IDENTIFICATION_COLUMNS if column not in header]
    if missing_columns:
        errors.append(f"{case['label']}: missing identification columns {missing_columns}")

    input_by_time = {round(row["time_float"], 12): row for row in input_rows}
    for row in identification_rows:
        time = round(float(row["time"]), 12)
        input_row = input_by_time.get(time)
        if input_row is None:
            errors.append(f"{case['label']}: missing input sample at t={time:g}")
            break
        comparisons = [
            ("control_u", "value"),
            ("control_u_dot", "u_dot"),
            ("control_segment_id", "segment_id"),
            ("time_since_switch", "time_since_switch"),
            ("is_switch_sample", "is_switch_sample"),
            ("control_kind", "control_kind"),
        ]
        for output_column, input_column in comparisons:
            if output_column in ("control_kind", "control_segment_id", "is_switch_sample"):
                if str(row[output_column]) != str(input_row[input_column]):
                    errors.append(f"{case['label']}: {output_column} mismatch at t={time:g}")
                    return errors
            else:
                if abs(float(row[output_column]) - float(input_row[input_column])) > 1e-10:
                    errors.append(f"{case['label']}: {output_column} mismatch at t={time:g}")
                    return errors

    input_metrics = collect_input_metrics(run_directory / "input_timeseries.csv")
    if input_metrics["max_abs_u"] > u_max + 1e-10:
        errors.append(f"{case['label']}: |control_u| exceeds u_max")
    if abs(float(result["max_abs_u"]) - input_metrics["max_abs_u"]) > 1e-10:
        errors.append(f"{case['label']}: summary max_abs_u mismatch")
    if abs(float(result["max_abs_u_dot"]) - input_metrics["max_abs_u_dot"]) > 1e-10:
        errors.append(f"{case['label']}: summary max_abs_u_dot mismatch")
    if float(result["max_abs_tip_y"]) > max_reasonable_tip_displacement:
        errors.append(
            f"{case['label']}: max_abs_tip_y={result['max_abs_tip_y']} exceeds "
            f"max_reasonable_tip_displacement={max_reasonable_tip_displacement}"
        )

    errors.extend(audit_actuator_balance(run_directory, case, input_rows, input_dt))
    errors.extend(audit_case_metadata(run_directory, case, result))
    return errors


def audit_actuator_balance(run_directory, case, input_rows, input_dt):
    errors = []
    input_schedule = InputSchedule(input_rows)
    actuator_by_time = defaultdict(dict)
    with (run_directory / "actuator_timeseries.csv").open(newline="") as actuator_file:
        reader = csv.DictReader(actuator_file)
        for row in reader:
            time = round(float(row["time"]), 12)
            if row["actuator_name"].endswith("_upper"):
                actuator_by_time[time]["upper"] = float(row["control_value"])
            elif row["actuator_name"].endswith("_lower"):
                actuator_by_time[time]["lower"] = float(row["control_value"])
    for time, pair in actuator_by_time.items():
        if "upper" not in pair or "lower" not in pair:
            errors.append(f"{case['label']}: incomplete actuator pair at t={time:g}")
            break
        if abs(pair["upper"] + pair["lower"]) > 1e-10:
            errors.append(f"{case['label']}: actuator mass balance failed at t={time:g}")
            break
        expected = input_schedule.lookup(time)
        if abs(pair["upper"] - float(expected["value"])) > 1e-8:
            if case["csv_interpolation"] != "zoh" or not base.actuator_value_is_consistent_near_switch(
                input_schedule,
                time,
                pair["upper"],
                input_dt,
            ):
                errors.append(f"{case['label']}: actuator/input mismatch at t={time:g}")
                break
    return errors


def audit_case_metadata(run_directory, case, result):
    errors = []
    metadata = json.loads((run_directory / "case_metadata.json").read_text())
    stored_case = metadata.get("case", {})
    if stored_case.get("label") != case["label"]:
        errors.append(f"{case['label']}: case_metadata label mismatch")
    if result.get("kind") != case["kind"] or result.get("role") != case["role"]:
        errors.append(f"{case['label']}: summary role/kind mismatch")
    return errors


class InputSchedule:
    def __init__(self, input_rows):
        self.rows = input_rows
        self.times = [row["time_float"] for row in input_rows]

    def lookup(self, time):
        index = bisect_right(self.times, round(time, 12)) - 1
        if index < 0:
            return self.rows[0]
        return self.rows[index]


def print_audit_report(results, manifest):
    ordered = sort_results(results, manifest)
    print("Audit report:")
    print(f"  number_of_cases={len(ordered)}")
    for result in ordered:
        print(
            f"  {result['label']}: "
            f"kind={result['kind']}, role={result['role']}, "
            f"u=[{float(result['min_u']):.6g}, {float(result['max_u']):.6g}], "
            f"max|u|={float(result['max_abs_u']):.6g}, "
            f"max|u_dot|={float(result['max_abs_u_dot']):.6g}, "
            f"max|tip_y|={float(result['max_abs_tip_y']):.6g}"
        )


def make_manifest(settings, cases):
    return {
        "campaign_label": settings["campaign_label"],
        "created_at": settings["created_at"],
        "modeling_goal": "x_dot = f(x,u), u_dot = epsilon; frozen-u manifolds x=W(eta,u)",
        "control_convention": "scalar control_u applied as upper=u, lower=-u",
        "not_harmonic_forcing": True,
        "u_max": settings["u_max"],
        "end_time": settings["end_time"],
        "output_dt": settings["output_dt"],
        "input_dt": settings["input_dt"],
        "write_paraview": settings["write_paraview"],
        "constant_count": settings["constant_count"],
        "include_fast_ramps": settings["include_fast_ramps"],
        "include_zoh": settings["include_zoh"],
        "max_reasonable_tip_displacement": settings["max_reasonable_tip_displacement"],
        "smoke": settings["smoke"],
        "identification_snapshot_columns": REQUIRED_IDENTIFICATION_COLUMNS,
        "total_number_of_cases": len(cases),
        "cases": cases,
    }


def write_manifest_once(campaign_directory, settings, cases):
    campaign_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = campaign_directory / "manifest.json"
    manifest = make_manifest(settings, cases)
    try:
        with manifest_path.open("x") as output_file:
            json.dump(manifest, output_file, indent=4)
    except FileExistsError:
        existing = json.loads(manifest_path.read_text())
        existing_labels = [case["label"] for case in existing.get("cases", [])]
        new_labels = [case["label"] for case in cases]
        if existing_labels != new_labels:
            raise RuntimeError(f"{manifest_path} exists for a different campaign.")


def read_manifest(campaign_directory):
    manifest_path = campaign_directory / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def read_case_results(campaign_directory):
    results = []
    for result_path in sorted((campaign_directory / "case_results").glob("*.json")):
        result = json.loads(result_path.read_text())
        localize_result_paths(campaign_directory, result, result_path)
        results.append(result)
    if not results:
        raise RuntimeError(f"No case result JSON files found in {campaign_directory / 'case_results'}.")
    return results


def localize_result_paths(campaign_directory, result, result_path):
    label = result.get("label")
    if not label:
        return
    local_run_directory = (campaign_directory / "runs" / label).resolve()
    recorded_run_directory = Path(result.get("run_directory", ""))
    if recorded_run_directory.exists() or not local_run_directory.exists():
        return
    result["run_directory"] = str(local_run_directory)
    result["input_timeseries"] = str(local_run_directory / "input_timeseries.csv")
    result["identification_snapshots"] = str(local_run_directory / "identification_snapshots.csv")
    result["case_result_path"] = str(result_path.resolve())


def write_case_selection(campaign_directory, index, case):
    path = campaign_directory / "case_selections" / f"{index:03d}_{case['label']}.json"
    base.write_text_no_overwrite(path, json.dumps({"case_index": index, "case": case}, indent=4))


def select_cases(cases):
    labels = os.environ.get("KRATOS_FSI_PM_CASE_LABELS")
    index = os.environ.get("KRATOS_FSI_PM_CASE_INDEX")
    limit = os.environ.get("KRATOS_FSI_PM_LIMIT")
    if labels:
        wanted = {label.strip() for label in labels.split(",") if label.strip()}
        return [(i, case) for i, case in enumerate(cases) if case["label"] in wanted]
    if index is not None:
        index = int(index)
        if index < 0 or index >= len(cases):
            raise IndexError(f"KRATOS_FSI_PM_CASE_INDEX={index} outside 0..{len(cases)-1}.")
        return [(index, cases[index])]
    selected = list(enumerate(cases))
    if limit is not None:
        selected = selected[:int(limit)]
    return selected


def sort_results(results, manifest):
    order = {case["label"]: index for index, case in enumerate(manifest["cases"])}
    return sorted(results, key=lambda result: order.get(result["label"], len(order)))


def linspace(start, stop, count):
    if count == 1:
        return [0.5 * (start + stop)]
    return [start + i * (stop - start) / (count - 1) for i in range(count)]


def max_abs_ramp_derivative(u_start, u_end, duration):
    if duration <= 0.0:
        return 0.0
    return abs(u_end - u_start) * 0.5 * math.pi / duration


def max_abs_case_u_dot(segments):
    return max(
        (max_abs_ramp_derivative(segment["u_start"], segment["u_end"], segment["end_time"] - segment["start_time"])
         for segment in segments),
        default=0.0,
    )


def quantize_time(time, dt):
    return round(round(time / dt) * dt, 12)


def format_label_value(value):
    prefix = "m" if value < -1e-12 else "p"
    text = f"{abs(value):.4f}".replace(".", "p")
    return f"{prefix}{text}"


def format_result(result):
    return (
        f"{result['label']}: "
        f"max|u|={float(result['max_abs_u']):.6g}, "
        f"max|u_dot|={float(result['max_abs_u_dot']):.6g}, "
        f"max|tip_y|={float(result['max_abs_tip_y']):.6g}"
    )


def read_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


def read_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def read_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no", "off")


if __name__ == "__main__":
    main()
