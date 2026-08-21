"""Run an FSI2 scalar-control identification campaign for ROM/MPC design.

This campaign targets a reduced model of the extended controlled system

    x_dot = f(x, u)
    u_dot = epsilon

with frozen-control manifolds x = W(eta, u), eta_dot = R(eta, u).

The scalar control convention is the same as the previous scalar-control
campaigns:

    upper actuator control = u
    lower actuator control = -u
"""

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import run_control_identification_campaign as base
import run_control_parametric_manifold_campaign as pm


DEFAULT_CAMPAIGN_LABEL = "fsi_control_mpc_identification_t40_u20"
DEFAULT_U_MAX = 2.0
DEFAULT_U_KICK = 0.4
DEFAULT_END_TIME = 40.0
DEFAULT_T_KICK_END = 2.0
DEFAULT_T_INTERVENTION = 20.0
DEFAULT_RAMP_DURATION = 2.0
DEFAULT_OUTPUT_INTERVAL = 0.01
DEFAULT_SIGNAL_DT = 0.002
DEFAULT_WRITE_PARAVIEW = False
DEFAULT_INCLUDE_FAST_RAMPS = False
DEFAULT_INCLUDE_ZOH = True
DEFAULT_MAX_REASONABLE_TIP_DISPLACEMENT = 0.5

CONSTANT_U_VALUES = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.5, 0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0]
INTERVENTION_TARGETS = [-1.0, 1.0, -1.25, 1.25, -1.5, 1.5, -1.75, 1.75, -2.0, 2.0]
RAMP_DIRECTIONS = [(-2.0, 2.0), (2.0, -2.0), (0.0, 2.0), (0.0, -2.0), (2.0, 0.0), (-2.0, 0.0)]
TRAIN_RANDOM_SEEDS = [1, 2, 3, 4, 5]
VALIDATION_RANDOM_SEEDS = [101, 102]
CHALLENGE_RANDOM_SEEDS = [201, 202]
ZOH_TRAIN_SEEDS = [51]
ZOH_VALIDATION_SEEDS = [151]
RANDOM_TARGET_DISTRIBUTION_DESCRIPTION = (
    "70% from |u| in [1,2], 20% from |u| in [0.5,1], 10% from {-0.5,0,0.5}"
)

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
    "u_kick",
    "t_kick_end",
    "t_intervention",
    "u_target",
    "ramp_duration",
    "intervention_type",
    "random_seed",
    "target_distribution_description",
    "max_abs_u",
    "max_abs_u_dot",
    "min_u",
    "max_u",
    "last_time",
    "number_of_samples",
    "max_abs_tip_y",
    "rms_tip_y",
    "final_tip_y",
    "rms_tip_y_before_intervention",
    "rms_tip_y_after_intervention",
    "rms_tip_y_change_after_before",
    "rms_tip_y_relative_change_after_before",
    "max_abs_tip_y_before_intervention",
    "max_abs_tip_y_after_intervention",
    "max_abs_tip_y_change_after_before",
    "max_abs_tip_y_relative_change_after_before",
    "tip_y_peak_to_peak_before_intervention",
    "tip_y_peak_to_peak_after_intervention",
    "tip_y_peak_to_peak_change_after_before",
    "tip_y_peak_to_peak_relative_change_after_before",
    "tip_authority_score",
    "max_abs_control",
    "rms_control",
    "input_timeseries",
    "case_result_path",
]

REQUIRED_IDENTIFICATION_COLUMNS = [
    "time",
    "measurement_x_0_30_DISPLACEMENT_X",
    "measurement_x_0_30_DISPLACEMENT_Y",
    "measurement_x_0_30_DISPLACEMENT_Z",
    "measurement_x_0_40_DISPLACEMENT_X",
    "measurement_x_0_40_DISPLACEMENT_Y",
    "measurement_x_0_40_DISPLACEMENT_Z",
    "measurement_x_0_50_DISPLACEMENT_X",
    "measurement_x_0_50_DISPLACEMENT_Y",
    "measurement_x_0_50_DISPLACEMENT_Z",
    "measurement_tip_DISPLACEMENT_X",
    "measurement_tip_DISPLACEMENT_Y",
    "measurement_tip_DISPLACEMENT_Z",
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
        print_mpc_report(results, manifest)
        return

    settings = read_settings()
    cases = build_case_library(settings)

    if args.list_cases:
        for index, case in enumerate(cases):
            print(f"{index:03d} {case['label']} [{case['role']}/{case['kind']}]")
        print(f"number_of_cases={len(cases)}")
        return

    if args.dry_run is not None:
        campaign_directory = Path(args.dry_run)
        run_dry_campaign(campaign_directory, settings, cases)
        manifest = read_manifest(campaign_directory)
        results = read_case_results(campaign_directory)
        write_summary_files(campaign_directory, results, manifest)
        audit_campaign(campaign_directory, manifest, results)
        print_mpc_report(results, manifest)
        return

    campaign_directory = Path("run_outputs") / settings["campaign_label"]
    campaign_directory.mkdir(parents=True, exist_ok=True)
    write_manifest_once(campaign_directory, settings, cases)

    selected_cases = select_cases(cases)
    for index, case in selected_cases:
        print(f"[{index}] {case['label']}", flush=True)
        pm.write_case_selection(campaign_directory, index, case)
        run_directory = pm.run_case(case, campaign_directory, settings)
        result = collect_case_result(case, campaign_directory, run_directory, settings)
        write_case_result(campaign_directory, result)
        print(format_result(result), flush=True)

    if len(selected_cases) == len(cases):
        manifest = make_manifest(settings, cases)
        results = read_case_results(campaign_directory)
        write_summary_files(campaign_directory, results, manifest)
        audit_campaign(campaign_directory, manifest, results)
        print_mpc_report(results, manifest)

    print(f"campaign={campaign_directory.resolve()}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run or audit the FSI2 scalar-control MPC-identification campaign."
    )
    parser.add_argument("--dry-run", type=Path, help="Generate synthetic files and audit them.")
    parser.add_argument("--audit", type=Path, help="Audit an existing campaign directory.")
    parser.add_argument("--list-cases", action="store_true", help="List case indices and labels.")
    return parser.parse_args()


def read_settings():
    return {
        "campaign_label": os.environ.get("KRATOS_FSI_MPC_LABEL", DEFAULT_CAMPAIGN_LABEL),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "u_max": read_float("KRATOS_FSI_MPC_U_MAX", DEFAULT_U_MAX),
        "u_kick": read_float("KRATOS_FSI_MPC_U_KICK", DEFAULT_U_KICK),
        "end_time": read_float("KRATOS_FSI_MPC_END_TIME", DEFAULT_END_TIME),
        "t_kick_end": read_float("KRATOS_FSI_MPC_T_KICK_END", DEFAULT_T_KICK_END),
        "t_intervention": read_float("KRATOS_FSI_MPC_T_INTERVENTION", DEFAULT_T_INTERVENTION),
        "ramp_duration": read_float("KRATOS_FSI_MPC_RAMP_DURATION", DEFAULT_RAMP_DURATION),
        "output_dt": read_float("KRATOS_FSI_MPC_OUTPUT_INTERVAL", DEFAULT_OUTPUT_INTERVAL),
        "input_dt": read_float("KRATOS_FSI_MPC_SIGNAL_DT", DEFAULT_SIGNAL_DT),
        "write_paraview": read_bool("KRATOS_FSI_MPC_WRITE_PARAVIEW", DEFAULT_WRITE_PARAVIEW),
        "include_fast_ramps": read_bool("KRATOS_FSI_MPC_INCLUDE_FAST_RAMPS", DEFAULT_INCLUDE_FAST_RAMPS),
        "include_zoh": read_bool("KRATOS_FSI_MPC_INCLUDE_ZOH", DEFAULT_INCLUDE_ZOH),
        "max_reasonable_tip_displacement": read_float(
            "KRATOS_FSI_MPC_MAX_REASONABLE_TIP_DISPLACEMENT",
            DEFAULT_MAX_REASONABLE_TIP_DISPLACEMENT,
        ),
    }


def build_case_library(settings):
    cases = [
        make_passive_baseline(settings),
    ]
    cases.extend(make_constant_cases(settings))
    cases.append(make_kick_then_passive_case(settings))
    for target in INTERVENTION_TARGETS:
        cases.append(make_kick_then_intervention_case(settings, target, settings["ramp_duration"]))
    cases.extend(make_smooth_random_cases(settings))
    cases.extend(make_smooth_ramp_cases(settings))
    if settings["include_zoh"]:
        cases.extend(make_zoh_cases(settings))
    return cases


def make_passive_baseline(settings):
    end_time = settings["end_time"]
    return pm.make_case(
        label="passive_baseline",
        role="reference",
        kind="passive",
        metadata={
            "u_value": 0.0,
            "u_start": 0.0,
            "u_end": 0.0,
            "u_kick": 0.0,
            "t_kick_end": 0.0,
            "t_intervention": "",
            "u_target": 0.0,
            "ramp_duration": "",
            "intervention_type": "none",
            "random_seed": "",
            "target_distribution_description": "",
        },
        segments=[pm.hold_segment(0, 0.0, end_time, 0.0, "passive")],
    )


def make_constant_cases(settings):
    return [make_constant_case(settings, value) for value in CONSTANT_U_VALUES]


def make_constant_case(settings, value):
    end_time = settings["end_time"]
    value = round(float(value), 12)
    return pm.make_case(
        label=f"constant_u_{format_label_value(value)}",
        role="train",
        kind="constant_u",
        metadata={
            "u_value": value,
            "u_start": value,
            "u_end": value,
            "u_kick": "",
            "t_kick_end": "",
            "t_intervention": "",
            "u_target": "",
            "ramp_duration": "",
            "intervention_type": "",
            "random_seed": "",
            "target_distribution_description": "",
            "max_abs_u_dot": 0.0,
        },
        segments=[pm.hold_segment(0, 0.0, end_time, value, "constant_u")],
    )


def make_kick_then_passive_case(settings):
    end_time = settings["end_time"]
    t_kick_end = min(settings["t_kick_end"], end_time)
    return pm.make_case(
        label="kick_then_passive_no_intervention",
        role="reference",
        kind="kick_then_passive",
        metadata={
            "u_value": "",
            "u_start": settings["u_kick"],
            "u_end": 0.0,
            "u_kick": settings["u_kick"],
            "t_kick_end": t_kick_end,
            "t_intervention": "",
            "u_target": 0.0,
            "ramp_duration": "",
            "intervention_type": "none",
            "random_seed": "",
            "target_distribution_description": "",
        },
        segments=[
            pm.hold_segment(0, 0.0, t_kick_end, settings["u_kick"], "initial_kick"),
            pm.hold_segment(1, t_kick_end, end_time, 0.0, "passive_development"),
        ],
    )


def make_kick_then_intervention_case(settings, target, ramp_duration):
    end_time = settings["end_time"]
    t_kick_end = min(settings["t_kick_end"], end_time)
    t_intervention = min(settings["t_intervention"], end_time)
    ramp_end = min(end_time, t_intervention + ramp_duration)
    label = f"kick_then_intervention_{format_label_value(target)}_T{ramp_duration:g}"
    segments = [
        pm.hold_segment(0, 0.0, t_kick_end, settings["u_kick"], "initial_kick"),
        pm.hold_segment(1, t_kick_end, t_intervention, 0.0, "passive_development"),
        pm.ramp_segment(2, t_intervention, ramp_end, 0.0, target, "intervention_ramp_transition"),
    ]
    if ramp_end < end_time:
        segments.append(pm.hold_segment(3, ramp_end, end_time, target, "intervention_hold"))
    return pm.make_case(
        label=label,
        role="challenge" if abs(target) >= 1.75 - 1e-12 else "validation",
        kind="kick_then_intervention",
        metadata={
            "u_value": "",
            "u_start": settings["u_kick"],
            "u_end": round(float(target), 12),
            "u_kick": settings["u_kick"],
            "t_kick_end": t_kick_end,
            "t_intervention": t_intervention,
            "u_target": round(float(target), 12),
            "ramp_duration": ramp_duration,
            "intervention_type": "smooth_ramp_hold",
            "random_seed": "",
            "target_distribution_description": "",
            "max_abs_u_dot": pm.max_abs_case_u_dot(segments),
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
        hold_end = min(end_time, pm.quantize_time(current_time + hold_duration, settings["output_dt"]))
        if hold_end > current_time + 1e-12:
            segments.append(pm.hold_segment(segment_id, current_time, hold_end, current_u, "smooth_random_hold"))
            segment_id += 1
            current_time = hold_end
        if current_time >= end_time - 1e-12:
            break
        next_u = sample_mpc_random_target(rng, settings["u_max"])
        transition_end = min(end_time, pm.quantize_time(current_time + transition_duration, settings["output_dt"]))
        if transition_end <= current_time + 1e-12:
            transition_end = min(end_time, pm.quantize_time(current_time + settings["output_dt"], settings["output_dt"]))
        segments.append(pm.ramp_segment(segment_id, current_time, transition_end, current_u, next_u, "smooth_random_transition"))
        segment_id += 1
        current_u = next_u
        current_time = transition_end

    return pm.make_case(
        label=label,
        role=role,
        kind="smooth_random",
        metadata={
            "random_seed": seed,
            "target_distribution_description": RANDOM_TARGET_DISTRIBUTION_DESCRIPTION,
            "min_segment_duration": min_segment_duration,
            "max_segment_duration": max_segment_duration,
            "min_transition_duration": min_transition_duration,
            "max_transition_duration": max_transition_duration,
            "u_kick": "",
            "u_start": "",
            "u_end": "",
            "t_kick_end": "",
            "t_intervention": "",
            "u_target": "",
            "ramp_duration": "",
            "intervention_type": "",
            "max_abs_u_dot": pm.max_abs_case_u_dot(segments),
        },
        segments=segments,
    )


def sample_mpc_random_target(rng, u_max):
    draw = rng.random()
    sign = -1.0 if rng.random() < 0.5 else 1.0
    if draw < 0.70:
        return sign * rng.uniform(0.5 * u_max, u_max)
    if draw < 0.90:
        return sign * rng.uniform(0.25 * u_max, 0.5 * u_max)
    return rng.choice([-0.25 * u_max, 0.0, 0.25 * u_max])


def make_smooth_ramp_cases(settings):
    durations = [10.0, 20.0]
    if settings["include_fast_ramps"]:
        durations.insert(0, 5.0)
    cases = []
    for duration in durations:
        role = "validation" if duration == 20.0 else "train"
        if duration == 5.0:
            role = "challenge"
        for u_start, u_end in RAMP_DIRECTIONS:
            label = f"smooth_ramp_{format_label_value(u_start)}_to_{format_label_value(u_end)}_T{duration:g}"
            cases.append(make_smooth_ramp_case(settings, label, role, u_start, u_end, duration))
    return cases


def make_smooth_ramp_case(settings, label, role, u_start, u_end, ramp_duration):
    end_time = settings["end_time"]
    hold_before = min(5.0, end_time)
    ramp_start = hold_before
    ramp_end = min(end_time, ramp_start + ramp_duration)
    segments = [
        pm.hold_segment(0, 0.0, ramp_start, u_start, "smooth_ramp_hold"),
        pm.ramp_segment(1, ramp_start, ramp_end, u_start, u_end, "smooth_ramp_transition"),
    ]
    if ramp_end < end_time:
        segments.append(pm.hold_segment(2, ramp_end, end_time, u_end, "smooth_ramp_hold"))
    return pm.make_case(
        label=label,
        role=role,
        kind="smooth_ramp",
        metadata={
            "u_start": round(float(u_start), 12),
            "u_end": round(float(u_end), 12),
            "u_value": "",
            "ramp_duration": ramp_duration,
            "u_kick": "",
            "t_kick_end": "",
            "t_intervention": "",
            "u_target": "",
            "intervention_type": "",
            "random_seed": "",
            "target_distribution_description": "",
            "max_abs_u_dot": pm.max_abs_case_u_dot(segments),
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
    levels = [-u_max, -0.75 * u_max, -0.5 * u_max, -0.25 * u_max, 0.0, 0.25 * u_max, 0.5 * u_max, 0.75 * u_max, u_max]
    current_time = 0.0
    previous_u = 0.0
    segment_id = 0
    segments = []
    while current_time < end_time - 1e-12:
        candidates = [value for value in levels if abs(value - previous_u) > 1e-12]
        value = rng.choice(candidates)
        duration = rng.uniform(1.5, 3.0)
        next_time = min(end_time, pm.quantize_time(current_time + duration, settings["output_dt"]))
        if next_time <= current_time:
            next_time = min(end_time, pm.quantize_time(current_time + settings["output_dt"], settings["output_dt"]))
        segments.append(pm.hold_segment(segment_id, current_time, next_time, value, "zoh_switching"))
        previous_u = value
        current_time = next_time
        segment_id += 1
    return pm.make_case(
        label=label,
        role=role,
        kind="zoh_switching",
        metadata={
            "random_seed": seed,
            "target_distribution_description": "secondary ZOH coverage over nine levels in [-2,2]",
            "u_value": "",
            "u_start": "",
            "u_end": "",
            "u_kick": "",
            "t_kick_end": "",
            "t_intervention": "",
            "u_target": "",
            "ramp_duration": "",
            "intervention_type": "",
            "max_abs_u_dot": 0.0,
        },
        segments=segments,
        csv_interpolation="zoh",
    )


def run_dry_campaign(campaign_directory, settings, cases):
    base.ensure_empty_directory(campaign_directory)
    write_manifest_once(campaign_directory, settings, cases)
    for index, case in enumerate(cases):
        print(f"[dry {index}] {case['label']}", flush=True)
        run_directory = campaign_directory / "runs" / case["label"]
        base.ensure_empty_directory(run_directory)
        input_path = campaign_directory / "inputs" / f"{case['label']}.csv"
        pm.write_input_timeseries_csv(input_path, case, settings)
        (run_directory / "input_timeseries.csv").write_text(input_path.read_text())
        base.write_synthetic_beam_displacements(run_directory / "beam_displacement_timeseries.csv", {
            "end_time": settings["end_time"],
            "output_interval": settings["output_dt"],
        })
        base.write_synthetic_actuator_timeseries(run_directory / "actuator_timeseries.csv", input_path)
        pm.write_identification_snapshots_csv(
            run_directory / "beam_displacement_timeseries.csv",
            run_directory / "input_timeseries.csv",
            run_directory / "identification_snapshots.csv",
        )
        log_path = campaign_directory / "logs" / f"{case['label']}.log"
        base.write_text_no_overwrite(log_path, "dry run: Kratos was not launched\n")
        pm.write_case_metadata(case, settings, campaign_directory, input_path, run_directory, log_path, 0)
        result = collect_case_result(case, campaign_directory, run_directory, settings)
        write_case_result(campaign_directory, result)


def collect_case_result(case, campaign_directory, run_directory, settings):
    metadata = case["metadata"]
    metrics = collect_tip_metrics(run_directory, settings)
    input_metrics = pm.collect_input_metrics(run_directory / "input_timeseries.csv")
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
        "u_kick": metadata.get("u_kick", ""),
        "t_kick_end": metadata.get("t_kick_end", ""),
        "t_intervention": metadata.get("t_intervention", ""),
        "u_target": metadata.get("u_target", ""),
        "ramp_duration": metadata.get("ramp_duration", ""),
        "intervention_type": metadata.get("intervention_type", ""),
        "random_seed": metadata.get("random_seed", ""),
        "target_distribution_description": metadata.get("target_distribution_description", ""),
        **input_metrics,
        **metrics,
        "run_directory": str(run_directory.resolve()),
        "input_timeseries": str((run_directory / "input_timeseries.csv").resolve()),
        "identification_snapshots": str((run_directory / "identification_snapshots.csv").resolve()),
        "case_result_path": str((campaign_directory / "case_results" / f"{case['label']}.json").resolve()),
    }


def collect_tip_metrics(run_directory, settings):
    beam_path = run_directory / "beam_displacement_timeseries.csv"
    actuator_path = run_directory / "actuator_timeseries.csv"
    metrics = {
        "last_time": 0.0,
        "number_of_samples": 0,
        "max_abs_tip_y": 0.0,
        "rms_tip_y": 0.0,
        "final_tip_y": 0.0,
        "rms_tip_y_before_intervention": 0.0,
        "rms_tip_y_after_intervention": 0.0,
        "rms_tip_y_change_after_before": 0.0,
        "rms_tip_y_relative_change_after_before": "",
        "max_abs_tip_y_before_intervention": 0.0,
        "max_abs_tip_y_after_intervention": 0.0,
        "max_abs_tip_y_change_after_before": 0.0,
        "max_abs_tip_y_relative_change_after_before": "",
        "tip_y_peak_to_peak_before_intervention": 0.0,
        "tip_y_peak_to_peak_after_intervention": 0.0,
        "tip_y_peak_to_peak_change_after_before": 0.0,
        "tip_y_peak_to_peak_relative_change_after_before": "",
        "tip_authority_score": "",
        "max_abs_control": 0.0,
        "rms_control": 0.0,
    }
    total_square_sum = 0.0
    before_square_sum = 0.0
    after_square_sum = 0.0
    before_count = 0
    after_count = 0
    before_values = []
    after_values = []
    with beam_path.open(newline="") as beam_file:
        reader = csv.reader(beam_file)
        next(reader)
        header = next(reader)
        tip_y_index = header.index("tip_DISPLACEMENT_Y")
        for row in reader:
            time = float(row[0])
            tip_y = float(row[tip_y_index])
            square = tip_y * tip_y
            metrics["last_time"] = time
            metrics["final_tip_y"] = tip_y
            metrics["max_abs_tip_y"] = max(metrics["max_abs_tip_y"], abs(tip_y))
            total_square_sum += square
            metrics["number_of_samples"] += 1
            if 15.0 - 1e-12 <= time <= settings["t_intervention"] + 1e-12:
                before_square_sum += square
                before_count += 1
                before_values.append(tip_y)
            if 30.0 - 1e-12 <= time <= settings["end_time"] + 1e-12:
                after_square_sum += square
                after_count += 1
                after_values.append(tip_y)

    if metrics["number_of_samples"]:
        metrics["rms_tip_y"] = math.sqrt(total_square_sum / metrics["number_of_samples"])
    if before_count:
        metrics["rms_tip_y_before_intervention"] = math.sqrt(before_square_sum / before_count)
    if after_count:
        metrics["rms_tip_y_after_intervention"] = math.sqrt(after_square_sum / after_count)
    before = metrics["rms_tip_y_before_intervention"]
    after = metrics["rms_tip_y_after_intervention"]
    metrics["rms_tip_y_change_after_before"] = after - before
    if before > 1e-14:
        metrics["rms_tip_y_relative_change_after_before"] = (after - before) / before

    if before_values:
        metrics["max_abs_tip_y_before_intervention"] = max(abs(value) for value in before_values)
        metrics["tip_y_peak_to_peak_before_intervention"] = max(before_values) - min(before_values)
    if after_values:
        metrics["max_abs_tip_y_after_intervention"] = max(abs(value) for value in after_values)
        metrics["tip_y_peak_to_peak_after_intervention"] = max(after_values) - min(after_values)

    max_abs_before = metrics["max_abs_tip_y_before_intervention"]
    max_abs_after = metrics["max_abs_tip_y_after_intervention"]
    metrics["max_abs_tip_y_change_after_before"] = max_abs_after - max_abs_before
    if max_abs_before > 1e-14:
        metrics["max_abs_tip_y_relative_change_after_before"] = (max_abs_after - max_abs_before) / max_abs_before

    p2p_before = metrics["tip_y_peak_to_peak_before_intervention"]
    p2p_after = metrics["tip_y_peak_to_peak_after_intervention"]
    metrics["tip_y_peak_to_peak_change_after_before"] = p2p_after - p2p_before
    if p2p_before > 1e-14:
        p2p_relative_change = (p2p_after - p2p_before) / p2p_before
        metrics["tip_y_peak_to_peak_relative_change_after_before"] = p2p_relative_change
        metrics["tip_authority_score"] = -p2p_relative_change

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
    max_tip = float(manifest["max_reasonable_tip_displacement"])
    for case in expected_cases:
        result = result_by_label.get(case["label"])
        if result is None:
            continue
        run_directory = Path(result["run_directory"])
        if not run_directory.exists():
            run_directory = campaign_directory / "runs" / case["label"]
        errors.extend(pm.audit_case(campaign_directory, run_directory, case, result, expected_rows, input_dt, u_max, max_tip))
        errors.extend(audit_authority_case(run_directory, case, result))

    if errors:
        print("Campaign audit failed:")
        for error in errors:
            print(f"  ERROR: {error}")
        raise RuntimeError("MPC-identification campaign audit failed.")
    print("Campaign audit passed:")
    print(f"  expected cases completed: {len(expected_labels)}")
    print("  files, rows, schemas, controls, diagnostics, metadata, and actuator balance are OK")


def audit_authority_case(run_directory, case, result):
    errors = []
    rows = base.read_csv_dicts(run_directory / "identification_snapshots.csv")
    header = rows[0].keys()
    missing_columns = [column for column in REQUIRED_IDENTIFICATION_COLUMNS if column not in header]
    if missing_columns:
        errors.append(f"{case['label']}: missing identification columns {missing_columns}")

    before = compute_window_stats(rows, 15.0, float(result["t_intervention"] or 20.0))
    after = compute_window_stats(rows, 30.0, float(result["end_time"]))
    if abs(before["rms"] - float(result["rms_tip_y_before_intervention"])) > 1e-10:
        errors.append(f"{case['label']}: before-intervention RMS mismatch")
    if abs(after["rms"] - float(result["rms_tip_y_after_intervention"])) > 1e-10:
        errors.append(f"{case['label']}: after-intervention RMS mismatch")
    if abs(before["max_abs"] - float(result["max_abs_tip_y_before_intervention"])) > 1e-10:
        errors.append(f"{case['label']}: before-intervention max-abs mismatch")
    if abs(after["max_abs"] - float(result["max_abs_tip_y_after_intervention"])) > 1e-10:
        errors.append(f"{case['label']}: after-intervention max-abs mismatch")
    if abs(before["peak_to_peak"] - float(result["tip_y_peak_to_peak_before_intervention"])) > 1e-10:
        errors.append(f"{case['label']}: before-intervention peak-to-peak mismatch")
    if abs(after["peak_to_peak"] - float(result["tip_y_peak_to_peak_after_intervention"])) > 1e-10:
        errors.append(f"{case['label']}: after-intervention peak-to-peak mismatch")

    expected_changes = {
        "rms_tip_y_change_after_before": after["rms"] - before["rms"],
        "max_abs_tip_y_change_after_before": after["max_abs"] - before["max_abs"],
        "tip_y_peak_to_peak_change_after_before": after["peak_to_peak"] - before["peak_to_peak"],
    }
    for column, expected in expected_changes.items():
        if abs(expected - float(result[column])) > 1e-10:
            errors.append(f"{case['label']}: {column} mismatch")
            break

    relative_expected = {
        "rms_tip_y_relative_change_after_before": relative_change(after["rms"], before["rms"]),
        "max_abs_tip_y_relative_change_after_before": relative_change(after["max_abs"], before["max_abs"]),
        "tip_y_peak_to_peak_relative_change_after_before": relative_change(after["peak_to_peak"], before["peak_to_peak"]),
    }
    for column, expected in relative_expected.items():
        recorded = result[column]
        if expected == "":
            if recorded != "":
                errors.append(f"{case['label']}: {column} should be empty")
                break
        elif abs(expected - float(recorded)) > 1e-10:
            errors.append(f"{case['label']}: {column} mismatch")
            break
    expected_score = ""
    if relative_expected["tip_y_peak_to_peak_relative_change_after_before"] != "":
        expected_score = -relative_expected["tip_y_peak_to_peak_relative_change_after_before"]
    recorded_score = result["tip_authority_score"]
    if expected_score == "":
        if recorded_score != "":
            errors.append(f"{case['label']}: tip_authority_score should be empty")
    elif abs(expected_score - float(recorded_score)) > 1e-10:
        errors.append(f"{case['label']}: tip_authority_score mismatch")
    return errors


def compute_window_stats(rows, start_time, end_time):
    values = []
    square_sum = 0.0
    for row in rows:
        time = float(row["time"])
        if start_time - 1e-12 <= time <= end_time + 1e-12:
            tip_y = float(row["measurement_tip_DISPLACEMENT_Y"])
            values.append(tip_y)
            square_sum += tip_y * tip_y
    if not values:
        return {"rms": 0.0, "max_abs": 0.0, "peak_to_peak": 0.0}
    return {
        "rms": math.sqrt(square_sum / len(values)),
        "max_abs": max(abs(value) for value in values),
        "peak_to_peak": max(values) - min(values),
    }


def relative_change(after, before):
    if abs(before) <= 1e-14:
        return ""
    return (after - before) / before


def make_manifest(settings, cases):
    return {
        "campaign_label": settings["campaign_label"],
        "created_at": settings["created_at"],
        "purpose": "scalar-control reduced-order MPC identification",
        "base_campaigns": [
            "fsi_control_parametric_manifold_t40_u04",
            "fsi_control_authority_discovery_t40_u20",
        ],
        "modeling_goal": "x_dot = f(x,u), u_dot = epsilon; frozen-u manifolds x=W(eta,u)",
        "control_convention": "scalar control_u applied as upper=u, lower=-u",
        "u_max": settings["u_max"],
        "u_kick": settings["u_kick"],
        "t_kick_end": settings["t_kick_end"],
        "t_intervention": settings["t_intervention"],
        "ramp_duration": settings["ramp_duration"],
        "constant_u_values": CONSTANT_U_VALUES,
        "intervention_targets": INTERVENTION_TARGETS,
        "ramp_directions": RAMP_DIRECTIONS,
        "ramp_durations": [5.0, 10.0, 20.0] if settings["include_fast_ramps"] else [10.0, 20.0],
        "train_random_seeds": TRAIN_RANDOM_SEEDS,
        "validation_random_seeds": VALIDATION_RANDOM_SEEDS,
        "challenge_random_seeds": CHALLENGE_RANDOM_SEEDS,
        "random_target_distribution_description": RANDOM_TARGET_DISTRIBUTION_DESCRIPTION,
        "include_fast_ramps": settings["include_fast_ramps"],
        "include_zoh": settings["include_zoh"],
        "zoh_train_seeds": ZOH_TRAIN_SEEDS if settings["include_zoh"] else [],
        "zoh_validation_seeds": ZOH_VALIDATION_SEEDS if settings["include_zoh"] else [],
        "end_time": settings["end_time"],
        "output_dt": settings["output_dt"],
        "input_dt": settings["input_dt"],
        "write_paraview": settings["write_paraview"],
        "max_reasonable_tip_displacement": settings["max_reasonable_tip_displacement"],
        "identification_snapshot_columns": REQUIRED_IDENTIFICATION_COLUMNS,
        "main_diagnostic": (
            "tip_authority_score = -tip_y_peak_to_peak_relative_change_after_before "
            "between [15,20] and [30,40]; positive means peak-to-peak suppression"
        ),
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
    return pm.read_case_results(campaign_directory)


def select_cases(cases):
    labels = os.environ.get("KRATOS_FSI_MPC_CASE_LABELS")
    index = os.environ.get("KRATOS_FSI_MPC_CASE_INDEX")
    limit = os.environ.get("KRATOS_FSI_MPC_LIMIT")
    if labels:
        wanted = {label.strip() for label in labels.split(",") if label.strip()}
        return [(i, case) for i, case in enumerate(cases) if case["label"] in wanted]
    if index is not None:
        index = int(index)
        if index < 0 or index >= len(cases):
            raise IndexError(f"KRATOS_FSI_MPC_CASE_INDEX={index} outside 0..{len(cases)-1}.")
        return [(index, cases[index])]
    selected = list(enumerate(cases))
    if limit is not None:
        selected = selected[:int(limit)]
    return selected


def sort_results(results, manifest):
    order = {case["label"]: index for index, case in enumerate(manifest["cases"])}
    return sorted(results, key=lambda result: order.get(result["label"], len(order)))


def print_mpc_report(results, manifest):
    ordered = sort_results(results, manifest)
    print("MPC identification report:")
    print(f"  number_of_cases={len(ordered)}")
    for result in ordered:
        score = result["tip_authority_score"]
        score_text = "" if score == "" else f", suppression_score={float(score):.6g}"
        print(
            f"  {result['label']}: "
            f"u_target={result['u_target']}, "
            f"max|u|={float(result['max_abs_u']):.6g}, "
            f"max|u_dot|={float(result['max_abs_u_dot']):.6g}, "
            f"max|tip_y|={float(result['max_abs_tip_y']):.6g}, "
            f"rms_before={float(result['rms_tip_y_before_intervention']):.6g}, "
            f"rms_after={float(result['rms_tip_y_after_intervention']):.6g}, "
            f"p2p_before={float(result['tip_y_peak_to_peak_before_intervention']):.6g}, "
            f"p2p_after={float(result['tip_y_peak_to_peak_after_intervention']):.6g}"
            f"{score_text}"
        )


def format_result(result):
    score = result["tip_authority_score"]
    score_text = "" if score == "" else f", authority_score={float(score):.6g}"
    return (
        f"{result['label']}: "
        f"max|u|={float(result['max_abs_u']):.6g}, "
        f"max|u_dot|={float(result['max_abs_u_dot']):.6g}, "
        f"max|tip_y|={float(result['max_abs_tip_y']):.6g}"
        f"{score_text}"
    )


def format_label_value(value):
    prefix = "m" if value < -1e-12 else "p"
    text = f"{abs(value):.3f}".replace(".", "p")
    return f"{prefix}{text}"


def read_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


def read_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no", "off")


if __name__ == "__main__":
    main()
