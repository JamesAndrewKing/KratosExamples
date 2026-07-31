"""Run a ZOH control-identification campaign for the Turek FSI2 example.

This campaign treats the Rabault actuator pair as a directly controlled input:

    upper control = u
    lower control = -u

The generated identification data is for eta_dot = R(eta, u). It intentionally
does not write amplitude, frequency, or phase columns.
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
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path


NATURAL_FREQUENCY_ESTIMATE_HZ = 3.8
NATURAL_PERIOD = 1.0 / NATURAL_FREQUENCY_ESTIMATE_HZ
DWELL_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 3.0]
LEVEL_FACTORS = [-1.0, -0.5, 0.0, 0.5, 1.0]

DEFAULT_CAMPAIGN_LABEL = "fsi_control_identification_t40"
DEFAULT_CONTROL_MAX = 0.12
DEFAULT_END_TIME = 40.0
DEFAULT_OUTPUT_INTERVAL = 0.01
DEFAULT_SIGNAL_DT = 0.002
DEFAULT_WRITE_PARAVIEW = False
DEFAULT_WARMUP_DURATION = 10.0
DEFAULT_RELEASE_SWITCH_DURATION = 15.0
DEFAULT_RANDOM_SEED = 314159
DEFAULT_MOVING_WINDOW_DURATION = 1.0
DEFAULT_NEAR_UNDEFORMED_TOLERANCE = 0.005

TRAIN_SEEDS = [1, 2, 3, 4]
VALIDATION_SEEDS = [101, 102]
CHALLENGE_SEEDS = [201, 202, 203, 204]
ONSET_FRACTIONS_OF_T0 = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.0625, 0.1875]

PYTHONPATH = os.environ.get("KRATOS_FSI_PYTHONPATH")
DYLD_LIBRARY_PATH = os.environ.get("KRATOS_FSI_DYLD_LIBRARY_PATH")
LD_LIBRARY_PATH = os.environ.get("KRATOS_FSI_LD_LIBRARY_PATH")

SUMMARY_FIELDNAMES = [
    "label",
    "role",
    "kind",
    "seed",
    "end_time",
    "warmup_end_time",
    "switching_start_time",
    "switching_end_time",
    "release_start_time",
    "last_time",
    "max_abs_tip_y",
    "rms_tip_y",
    "post_warmup_rms_tip_y",
    "min_moving_window_tip_rms",
    "near_undeformed_fraction",
    "final_tip_y",
    "max_abs_control",
    "rms_control",
    "number_of_samples",
    "post_warmup_number_of_samples",
    "run_directory",
    "input_timeseries",
    "identification_snapshots",
]

REQUIRED_INPUT_COLUMNS = ["time", "value", "segment_id", "time_since_switch"]
REQUIRED_IDENTIFICATION_COLUMNS = [
    "time",
    "control_u",
    "control_segment_id",
    "time_since_switch",
    "is_switch_sample",
]
FORBIDDEN_IDENTIFICATION_COLUMNS = {
    "delta_amplitude",
    "delta_omega_rad_s",
    "theta_rad",
    "theta_unwrapped_rad",
    "amplitude",
    "frequency_hz",
    "omega_rad_s",
}


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
        print_authority_report(results, manifest)
        return

    settings = read_campaign_settings()
    all_cases = build_case_library(settings)

    if args.dry_run is not None:
        settings["dry_run"] = True
        campaign_directory = Path(args.dry_run)
        run_dry_campaign(campaign_directory, settings, all_cases)
        manifest = read_manifest(campaign_directory)
        results = read_case_results(campaign_directory)
        write_summary_files(campaign_directory, results, manifest)
        audit_campaign(campaign_directory, manifest, results)
        print_authority_report(results, manifest)
        return

    campaign_directory = Path("run_outputs") / settings["campaign_label"]
    campaign_directory.mkdir(parents=True, exist_ok=True)
    write_manifest_once(campaign_directory, settings, all_cases)

    case_index = read_optional_int("KRATOS_FSI_CONTROL_CASE_INDEX")
    cases = select_cases(all_cases, case_index)

    if case_index is not None:
        write_case_selection(campaign_directory, case_index, cases[0])

    results = []
    for i, case in enumerate(cases, start=1):
        print(f"[{i}/{len(cases)}] {case['label']}", flush=True)
        run_directory = run_case(case, campaign_directory, settings)
        result = collect_case_result(case, campaign_directory, run_directory, settings)
        results.append(result)
        write_case_result(campaign_directory, result)
        print(format_result(result), flush=True)

    if case_index is None:
        write_summary_files(campaign_directory, results, make_manifest(settings, all_cases))
        audit_campaign(campaign_directory, make_manifest(settings, all_cases), results)
        print_authority_report(results, make_manifest(settings, all_cases))

    print(f"campaign={campaign_directory.resolve()}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run or audit the FSI2 ZOH control-identification campaign."
    )
    parser.add_argument(
        "--dry-run",
        type=Path,
        help="Generate synthetic files and run the audit without launching Kratos.",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        help="Audit an existing campaign directory and regenerate its summary files.",
    )
    return parser.parse_args()


def read_campaign_settings():
    output_interval = read_float("KRATOS_FSI_CONTROL_OUTPUT_INTERVAL", DEFAULT_OUTPUT_INTERVAL)
    signal_dt = read_float("KRATOS_FSI_CONTROL_SIGNAL_DT", DEFAULT_SIGNAL_DT)
    dwell_choices = [
        quantize_time(multiplier * NATURAL_PERIOD, output_interval)
        for multiplier in DWELL_MULTIPLIERS
    ]
    return {
        "campaign_label": os.environ.get("KRATOS_FSI_CONTROL_LABEL", DEFAULT_CAMPAIGN_LABEL),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "random_seed": read_int("KRATOS_FSI_CONTROL_RANDOM_SEED", DEFAULT_RANDOM_SEED),
        "control_max": read_float("KRATOS_FSI_CONTROL_MAX", DEFAULT_CONTROL_MAX),
        "end_time": read_float("KRATOS_FSI_CONTROL_END_TIME", DEFAULT_END_TIME),
        "output_interval": output_interval,
        "signal_dt": signal_dt,
        "write_paraview": read_bool("KRATOS_FSI_CONTROL_WRITE_PARAVIEW", DEFAULT_WRITE_PARAVIEW),
        "warmup_duration": read_float("KRATOS_FSI_CONTROL_WARMUP_DURATION", DEFAULT_WARMUP_DURATION),
        "release_switch_duration": read_float(
            "KRATOS_FSI_CONTROL_RELEASE_SWITCH_DURATION",
            DEFAULT_RELEASE_SWITCH_DURATION,
        ),
        "dwell_time_choices": dwell_choices,
        "moving_window_duration": read_float(
            "KRATOS_FSI_CONTROL_MOVING_WINDOW_DURATION",
            DEFAULT_MOVING_WINDOW_DURATION,
        ),
        "near_undeformed_tolerance": read_float(
            "KRATOS_FSI_NEAR_UNDEFORMED_TOLERANCE",
            DEFAULT_NEAR_UNDEFORMED_TOLERANCE,
        ),
    }


def build_case_library(settings):
    control_levels = get_control_levels(settings["control_max"])
    forced_transition_chunks = build_training_transition_chunks(settings["random_seed"])
    cases = [
        make_passive_case(settings, control_levels),
    ]

    switching_specs = []
    for seed in TRAIN_SEEDS:
        switching_specs.append(("train", seed, False))
    for seed in VALIDATION_SEEDS:
        switching_specs.append(("validation", seed, False))
    for seed in CHALLENGE_SEEDS:
        switching_specs.append(("challenge", seed, True))

    for index, (role, seed, is_release_case) in enumerate(switching_specs):
        forced_transitions = None
        if role == "train":
            train_index = TRAIN_SEEDS.index(seed)
            forced_transitions = forced_transition_chunks[train_index]
        cases.append(make_switching_case(
            settings=settings,
            control_levels=control_levels,
            role=role,
            seed=seed,
            onset_fraction=ONSET_FRACTIONS_OF_T0[index],
            is_release_case=is_release_case,
            forced_transitions=forced_transitions,
        ))

    return cases


def make_passive_case(settings, control_levels):
    end_time = settings["end_time"]
    return {
        "label": "passive_baseline",
        "kind": "passive",
        "role": "passive",
        "seed": None,
        "controller": "csv",
        "csv_interpolation": "zoh",
        "end_time": end_time,
        "warmup_end_time": 0.0,
        "switching_start_time": 0.0,
        "switching_end_time": 0.0,
        "release_start_time": None,
        "control_levels": control_levels,
        "segments": [make_segment(0, 0.0, end_time, 0.0, 0.0)],
        "description": "No actuation baseline written through the same ZOH control channel.",
    }


def make_switching_case(
    settings,
    control_levels,
    role,
    seed,
    onset_fraction,
    is_release_case,
    forced_transitions,
):
    end_time = settings["end_time"]
    output_interval = settings["output_interval"]
    warmup_end_time = quantize_time(
        settings["warmup_duration"] + onset_fraction * NATURAL_PERIOD,
        output_interval,
    )
    switch_end_time = end_time
    release_start_time = None
    if is_release_case:
        switch_end_time = quantize_time(
            min(end_time, warmup_end_time + settings["release_switch_duration"]),
            output_interval,
        )
        release_start_time = switch_end_time

    level_factors = LEVEL_FACTORS
    if is_release_case:
        # Keep challenge actuation energetic during the forced interval, then release.
        level_factors = [-1.0, -0.5, 0.5, 1.0]

    segments = [make_segment(0, 0.0, warmup_end_time, 0.0, 0.0)]
    segments.extend(generate_switching_segments(
        seed=seed,
        start_time=warmup_end_time,
        end_time=switch_end_time,
        level_factors=level_factors,
        control_max=settings["control_max"],
        dwell_choices=settings["dwell_time_choices"],
        output_interval=output_interval,
        forced_transitions=forced_transitions,
        first_segment_id=1,
        previous_level_factor=0.0,
    ))
    if is_release_case and switch_end_time < end_time:
        previous_segment = segments[-1]
        next_segment_id = previous_segment["segment_id"] + 1
        if previous_segment["level_factor"] == 0.0:
            previous_segment["end_time"] = end_time
        else:
            segments.append(make_segment(next_segment_id, switch_end_time, end_time, 0.0, 0.0))

    label_prefix = {
        "train": "train_switch",
        "validation": "validation_switch",
        "challenge": "release_challenge",
    }[role]
    label = f"{label_prefix}_seed{seed:03d}" if seed >= 100 else f"{label_prefix}_seed{seed:02d}"

    return {
        "label": label,
        "kind": "zoh_switching_release" if is_release_case else "zoh_switching",
        "role": role,
        "seed": seed,
        "controller": "csv",
        "csv_interpolation": "zoh",
        "end_time": end_time,
        "warmup_end_time": warmup_end_time,
        "switching_start_time": warmup_end_time,
        "switching_end_time": switch_end_time,
        "release_start_time": release_start_time,
        "onset_offset_fraction_of_T0": onset_fraction,
        "onset_offset_seconds": quantize_time(onset_fraction * NATURAL_PERIOD, output_interval),
        "control_levels": control_levels,
        "segments": segments,
        "description": "Zero-order-held signed scalar control input for the Rabault actuator pair.",
    }


def generate_switching_segments(
    seed,
    start_time,
    end_time,
    level_factors,
    control_max,
    dwell_choices,
    output_interval,
    forced_transitions=None,
    first_segment_id=1,
    previous_level_factor=0.0,
):
    if start_time >= end_time:
        return []

    rng = random.Random(seed)
    segments = []
    occupancy = defaultdict(float)
    current_time = start_time
    segment_id = first_segment_id
    previous_factor = previous_level_factor

    forced_sequence = flatten_forced_transitions(forced_transitions or [], previous_factor)
    for factor in forced_sequence:
        if current_time >= end_time - 1e-12:
            break
        if factor == previous_factor:
            continue
        segment_id, current_time, previous_factor = append_random_duration_segment(
            segments,
            segment_id,
            current_time,
            end_time,
            factor,
            control_max,
            dwell_choices,
            output_interval,
            occupancy,
            rng,
        )

    while current_time < end_time - 1e-12:
        candidates = [factor for factor in level_factors if factor != previous_factor]
        minimum_occupancy = min(occupancy[factor] for factor in candidates)
        best_candidates = [
            factor
            for factor in candidates
            if occupancy[factor] <= minimum_occupancy + 1e-12
        ]
        factor = rng.choice(best_candidates)
        segment_id, current_time, previous_factor = append_random_duration_segment(
            segments,
            segment_id,
            current_time,
            end_time,
            factor,
            control_max,
            dwell_choices,
            output_interval,
            occupancy,
            rng,
        )

    return segments


def append_random_duration_segment(
    segments,
    segment_id,
    current_time,
    end_time,
    factor,
    control_max,
    dwell_choices,
    output_interval,
    occupancy,
    rng,
):
    duration = rng.choice(dwell_choices)
    next_time = quantize_time(current_time + duration, output_interval)
    if next_time <= current_time:
        next_time = quantize_time(current_time + output_interval, output_interval)
    next_time = min(next_time, end_time)
    segments.append(make_segment(
        segment_id,
        current_time,
        next_time,
        factor,
        factor * control_max,
    ))
    occupancy[factor] += next_time - current_time
    return segment_id + 1, next_time, factor


def make_segment(segment_id, start_time, end_time, level_factor, value):
    return {
        "segment_id": segment_id,
        "start_time": round(float(start_time), 12),
        "end_time": round(float(end_time), 12),
        "level_factor": round(float(level_factor), 12),
        "value": round(float(value), 12),
    }


def build_training_transition_chunks(random_seed):
    rng = random.Random(random_seed)
    transitions = [
        (source, target)
        for source in LEVEL_FACTORS
        for target in LEVEL_FACTORS
        if target != source
    ]
    rng.shuffle(transitions)
    return [
        transitions[index::len(TRAIN_SEEDS)]
        for index in range(len(TRAIN_SEEDS))
    ]


def flatten_forced_transitions(transitions, previous_factor):
    sequence = []
    current_factor = previous_factor
    for source, target in transitions:
        if source != current_factor:
            sequence.append(source)
            current_factor = source
        if target != current_factor:
            sequence.append(target)
            current_factor = target
    return sequence


def run_case(case, campaign_directory, settings):
    run_directory = campaign_directory / "runs" / case["label"]
    ensure_empty_directory(run_directory)

    input_path = campaign_directory / "inputs" / f"{case['label']}.csv"
    log_path = campaign_directory / "logs" / f"{case['label']}.log"
    ensure_file_does_not_exist(log_path)
    ensure_file_does_not_exist(campaign_directory / "case_results" / f"{case['label']}.json")
    write_input_timeseries_csv(input_path, case, settings)

    environment = os.environ.copy()
    environment.update({
        "KRATOS_FSI_RUN_LABEL": case["label"],
        "KRATOS_FSI_RUN_OUTPUT_DIRECTORY": str(run_directory.resolve()),
        "KRATOS_FSI_END_TIME": str(settings["end_time"]),
        "KRATOS_FSI_OUTPUT_INTERVAL": str(settings["output_interval"]),
        "KRATOS_FSI_WRITE_PARAVIEW": "1" if settings["write_paraview"] else "0",
        "KRATOS_FSI_CONTROLLER_TYPE": "csv",
        "KRATOS_FSI_ACTUATOR_CSV_FILE": str(input_path.resolve()),
        "KRATOS_FSI_ACTUATOR_CSV_TIME_COLUMN": "time",
        "KRATOS_FSI_ACTUATOR_CSV_VALUE_COLUMN": "value",
        "KRATOS_FSI_ACTUATOR_CSV_INTERPOLATION": "zoh",
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
    write_text_no_overwrite(log_path, completed.stdout)

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
    ensure_empty_directory(campaign_directory)
    write_manifest_once(campaign_directory, settings, cases)

    for i, case in enumerate(cases, start=1):
        print(f"[dry {i}/{len(cases)}] {case['label']}", flush=True)
        run_directory = campaign_directory / "runs" / case["label"]
        ensure_empty_directory(run_directory)
        input_path = campaign_directory / "inputs" / f"{case['label']}.csv"
        write_input_timeseries_csv(input_path, case, settings)
        shutil.copyfile(input_path, run_directory / "input_timeseries.csv")
        write_synthetic_beam_displacements(run_directory / "beam_displacement_timeseries.csv", settings)
        write_synthetic_actuator_timeseries(run_directory / "actuator_timeseries.csv", input_path)
        write_identification_snapshots_csv(
            run_directory / "beam_displacement_timeseries.csv",
            run_directory / "input_timeseries.csv",
            run_directory / "identification_snapshots.csv",
        )
        log_path = campaign_directory / "logs" / f"{case['label']}.log"
        write_text_no_overwrite(log_path, "dry run: Kratos was not launched\n")
        write_case_metadata(case, settings, campaign_directory, input_path, run_directory, log_path, 0)
        result = collect_case_result(case, campaign_directory, run_directory, settings)
        write_case_result(campaign_directory, result)


def write_input_timeseries_csv(path, case, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_file_does_not_exist(path)
    samples = iter_sample_times(settings["end_time"], settings["signal_dt"])
    schedule = SegmentSchedule(case["segments"])
    switch_times = {
        round(segment["start_time"], 12)
        for segment in case["segments"]
        if segment["segment_id"] > 0
    }

    with path.open("x", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["time", "value", "segment_id", "time_since_switch", "is_switch_sample"])
        for time in samples:
            segment = schedule.lookup(time)
            time_since_switch = max(0.0, time - segment["start_time"])
            writer.writerow([
                format_float(time),
                format_float(segment["value"]),
                segment["segment_id"],
                format_float(time_since_switch),
                int(round(time, 12) in switch_times),
            ])


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
    ensure_file_does_not_exist(output_path)
    with output_path.open("x", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "time",
            *[f"measurement_{name}" for name in measurement_columns],
            "control_u",
            "control_segment_id",
            "time_since_switch",
            "is_switch_sample",
        ])

        for beam_row in beam_rows:
            time = float(beam_row[0])
            input_row = find_input_row(input_rows, input_rows_by_time, time)
            writer.writerow([
                format_float(time),
                *beam_row[1:],
                input_row["value"],
                input_row["segment_id"],
                input_row["time_since_switch"],
                input_row["is_switch_sample"],
            ])

    metadata_path = output_path.with_suffix(".metadata.csv")
    ensure_file_does_not_exist(metadata_path)
    with metadata_path.open("x", newline="") as metadata_file:
        writer = csv.writer(metadata_file)
        writer.writerow(metadata)


def write_synthetic_beam_displacements(path, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_file_does_not_exist(path)
    metadata = ["time"]
    header = ["time"]
    for name, node_id, x0, y0 in [
        ("x_0_30", 1, 0.30, 0.20),
        ("x_0_40", 2, 0.40, 0.20),
        ("x_0_50", 3, 0.50, 0.20),
        ("tip", 4, 0.60, 0.20),
    ]:
        metadata.extend([
            f"{name}_node_id={node_id}",
            f"{name}_x0={x0:.12g}",
            f"{name}_y0={y0:.12g}",
        ])
        header.extend([
            f"{name}_DISPLACEMENT_X",
            f"{name}_DISPLACEMENT_Y",
            f"{name}_DISPLACEMENT_Z",
        ])

    with path.open("x", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(metadata)
        writer.writerow(header)
        for time in iter_sample_times(settings["end_time"], settings["output_interval"]):
            writer.writerow([format_float(time), *["0"] * (len(header) - 1)])


def write_synthetic_actuator_timeseries(path, input_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_file_does_not_exist(path)
    with input_path.open(newline="") as input_file, path.open("x", newline="") as output_file:
        input_reader = csv.DictReader(input_file)
        writer = csv.writer(output_file)
        writer.writerow([
            "time",
            "actuator_name",
            "control_value",
            "weighted_mean_velocity_x",
            "weighted_mean_velocity_y",
            "weighted_mean_velocity_z",
            "number_of_nodes",
        ])
        for row in input_reader:
            time = row["time"]
            value = float(row["value"])
            writer.writerow([time, "rabault_pair_upper", format_float(value), "0", "0", "0", 1])
            writer.writerow([time, "rabault_pair_lower", format_float(-value), "0", "0", "0", 1])


def write_case_metadata(case, settings, campaign_directory, input_path, run_directory, log_path, return_code):
    metadata = {
        "case": case,
        "campaign_settings": {
            "control_max": settings["control_max"],
            "control_levels": get_control_levels(settings["control_max"]),
            "natural_frequency_estimate_hz": NATURAL_FREQUENCY_ESTIMATE_HZ,
            "natural_period": NATURAL_PERIOD,
            "dwell_time_choices": settings["dwell_time_choices"],
            "warmup_duration": settings["warmup_duration"],
            "release_switch_duration": settings["release_switch_duration"],
            "zoh_convention": "value is held from each CSV time until the next CSV time",
        },
        "input_path": str(input_path.resolve()),
        "signal_path": str(input_path.resolve()),
        "identification_snapshots_path": str((run_directory / "identification_snapshots.csv").resolve()),
        "return_code": return_code,
        "log_path": str(log_path.resolve()),
    }
    write_text_no_overwrite(
        run_directory / "case_metadata.json",
        json.dumps(metadata, indent=4),
    )


def collect_case_result(case, campaign_directory, run_directory, settings):
    metrics = collect_metrics(run_directory, case, settings)
    return {
        **{key: value for key, value in case.items() if key != "segments"},
        **metrics,
        "run_directory": str(run_directory.resolve()),
        "input_timeseries": str((run_directory / "input_timeseries.csv").resolve()),
        "identification_snapshots": str((run_directory / "identification_snapshots.csv").resolve()),
        "case_result_path": str((campaign_directory / "case_results" / f"{case['label']}.json").resolve()),
    }


def collect_metrics(run_directory, case, settings):
    beam_path = run_directory / "beam_displacement_timeseries.csv"
    actuator_path = run_directory / "actuator_timeseries.csv"
    warmup_end_time = case.get("warmup_end_time", settings["warmup_duration"])
    near_tolerance = settings["near_undeformed_tolerance"]
    moving_window = settings["moving_window_duration"]

    metrics = {
        "last_time": 0.0,
        "max_abs_tip_y": 0.0,
        "rms_tip_y": 0.0,
        "post_warmup_rms_tip_y": 0.0,
        "min_moving_window_tip_rms": 0.0,
        "near_undeformed_fraction": 0.0,
        "final_tip_y": 0.0,
        "max_abs_control": 0.0,
        "rms_control": 0.0,
        "number_of_samples": 0,
        "post_warmup_number_of_samples": 0,
    }

    tip_square_sum = 0.0
    post_warmup_tip_square_sum = 0.0
    near_count = 0
    post_warmup_samples = []
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
            if time + 1e-12 >= warmup_end_time:
                post_warmup_tip_square_sum += tip_y * tip_y
                metrics["post_warmup_number_of_samples"] += 1
                post_warmup_samples.append((time, tip_y))
                if abs(tip_y) <= near_tolerance:
                    near_count += 1

    if metrics["number_of_samples"]:
        metrics["rms_tip_y"] = math.sqrt(tip_square_sum / metrics["number_of_samples"])
    if metrics["post_warmup_number_of_samples"]:
        metrics["post_warmup_rms_tip_y"] = math.sqrt(
            post_warmup_tip_square_sum / metrics["post_warmup_number_of_samples"]
        )
        metrics["near_undeformed_fraction"] = (
            near_count / metrics["post_warmup_number_of_samples"]
        )
    metrics["min_moving_window_tip_rms"] = calculate_minimum_moving_window_rms(
        post_warmup_samples,
        moving_window,
    )

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


def calculate_minimum_moving_window_rms(samples, window_duration):
    if not samples:
        return 0.0
    window = deque()
    square_sum = 0.0
    best = None
    for time, value in samples:
        window.append((time, value))
        square_sum += value * value
        while window and time - window[0][0] > window_duration + 1e-12:
            _, removed_value = window.popleft()
            square_sum -= removed_value * removed_value
        if window and time - window[0][0] >= 0.99 * window_duration:
            rms = math.sqrt(square_sum / len(window))
            best = rms if best is None else min(best, rms)
    if best is not None:
        return best
    return math.sqrt(sum(value * value for _, value in samples) / len(samples))


def write_case_result(campaign_directory, result):
    results_directory = campaign_directory / "case_results"
    results_directory.mkdir(parents=True, exist_ok=True)
    result_path = results_directory / f"{result['label']}.json"
    write_text_no_overwrite(result_path, json.dumps(result, indent=4))


def write_summary_files(campaign_directory, results, manifest):
    ordered_results = sort_results_by_manifest(results, manifest)
    (campaign_directory / "summary.json").write_text(json.dumps(ordered_results, indent=4))
    with (campaign_directory / "summary.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=SUMMARY_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered_results)
    print(f"wrote {campaign_directory / 'summary.csv'}")
    print(f"wrote {campaign_directory / 'summary.json'}")


def audit_campaign(campaign_directory, manifest, results):
    errors = []
    warnings = []
    expected_cases = manifest["cases"]
    expected_labels = [case["label"] for case in expected_cases]
    expected_label_set = set(expected_labels)
    result_by_label = {result["label"]: result for result in results}

    missing = [label for label in expected_labels if label not in result_by_label]
    unexpected = [label for label in result_by_label if label not in expected_label_set]
    if missing:
        errors.append(f"Missing case results: {', '.join(missing)}")
    if unexpected:
        errors.append(f"Unexpected case results: {', '.join(unexpected)}")

    run_directories = [
        result.get("run_directory", "")
        for result in results
        if result.get("run_directory")
    ]
    duplicate_directories = [
        directory
        for directory, count in Counter(run_directories).items()
        if count > 1
    ]
    if duplicate_directories:
        errors.append("Duplicate run directories: " + ", ".join(duplicate_directories))

    train_seeds = sorted(case["seed"] for case in expected_cases if case["role"] == "train")
    validation_seeds = sorted(case["seed"] for case in expected_cases if case["role"] == "validation")
    if set(train_seeds).intersection(validation_seeds):
        errors.append("Training and validation seeds are not distinct.")

    allowed_values = {
        round(value, 12)
        for value in manifest["control_levels"]
    }
    expected_end_time = float(manifest["end_time"])
    signal_dt = float(manifest["signal_dt"])
    for case in expected_cases:
        result = result_by_label.get(case["label"])
        if result is None:
            continue
        run_directory = Path(result["run_directory"])
        errors.extend(audit_case_files(
            run_directory,
            case,
            allowed_values,
            expected_end_time,
            signal_dt,
        ))

    errors.extend(audit_training_coverage(expected_cases))

    if errors:
        print("Campaign audit failed:")
        for error in errors:
            print(f"  ERROR: {error}")
        for warning in warnings:
            print(f"  WARNING: {warning}")
        raise RuntimeError("Control-identification campaign audit failed.")

    print("Campaign audit passed:")
    print(f"  expected cases completed: {len(expected_labels)}")
    print("  output schemas, ZOH alignment, mass balance, levels, seeds, and coverage are OK")


def audit_case_files(run_directory, case, allowed_values, expected_end_time, signal_dt):
    errors = []
    input_path = run_directory / "input_timeseries.csv"
    identification_path = run_directory / "identification_snapshots.csv"
    actuator_path = run_directory / "actuator_timeseries.csv"
    for path in [input_path, identification_path, actuator_path, run_directory / "case_metadata.json"]:
        if not path.exists():
            errors.append(f"{case['label']}: missing {path}")
            return errors

    input_rows = read_input_rows_by_time(input_path)
    identification_rows = read_csv_dicts(identification_path)
    actuator_rows = read_csv_dicts(actuator_path)

    errors.extend(require_columns(case["label"], input_path, input_rows[0], REQUIRED_INPUT_COLUMNS))
    errors.extend(require_columns(
        case["label"],
        identification_path,
        identification_rows[0],
        REQUIRED_IDENTIFICATION_COLUMNS,
    ))
    forbidden_present = FORBIDDEN_IDENTIFICATION_COLUMNS.intersection(identification_rows[0].keys())
    if forbidden_present:
        errors.append(
            f"{case['label']}: identification file contains periodic columns "
            f"{sorted(forbidden_present)}"
        )

    first_time = float(identification_rows[0]["time"])
    last_time = float(identification_rows[-1]["time"])
    if abs(first_time) > 1e-12:
        errors.append(f"{case['label']}: identification starts at {first_time}, not 0.")
    if abs(last_time - expected_end_time) > 5e-9:
        errors.append(
            f"{case['label']}: identification ends at {last_time}, not {expected_end_time}."
        )

    input_schedule = InputSchedule(input_rows)
    for row in input_rows:
        if round(float(row["value"]), 12) not in allowed_values:
            errors.append(
                f"{case['label']}: input file contains invalid control level "
                f"{row['value']} at t={row['time']}."
            )
            break

    for row in identification_rows:
        time = float(row["time"])
        expected = input_schedule.lookup(time)
        if abs(float(row["control_u"]) - float(expected["value"])) > 1e-10:
            errors.append(f"{case['label']}: control_u mismatch at t={time:.12g}.")
            break
        if row["control_segment_id"] != expected["segment_id"]:
            errors.append(f"{case['label']}: segment_id mismatch at t={time:.12g}.")
            break
        if abs(float(row["time_since_switch"]) - float(expected["time_since_switch"])) > 1e-10:
            errors.append(f"{case['label']}: time_since_switch mismatch at t={time:.12g}.")
            break
        if int(row["is_switch_sample"]) != int(expected["is_switch_sample"]):
            errors.append(f"{case['label']}: is_switch_sample mismatch at t={time:.12g}.")
            break
        if int(row["is_switch_sample"]) == 0 and round(float(row["control_u"]), 12) not in allowed_values:
            errors.append(f"{case['label']}: invalid control level at t={time:.12g}.")
            break

    if case["role"] == "passive":
        if any(abs(float(row["value"])) > 1e-12 for row in input_rows):
            errors.append(f"{case['label']}: passive input is not exactly zero.")

    actuator_by_time = defaultdict(dict)
    for row in actuator_rows:
        time = round(float(row["time"]), 12)
        if row["actuator_name"].endswith("_upper"):
            actuator_by_time[time]["upper"] = float(row["control_value"])
        elif row["actuator_name"].endswith("_lower"):
            actuator_by_time[time]["lower"] = float(row["control_value"])

    for time, pair in actuator_by_time.items():
        if "upper" not in pair or "lower" not in pair:
            errors.append(f"{case['label']}: incomplete actuator pair at t={time:.12g}.")
            break
        if abs(pair["upper"] + pair["lower"]) > 1e-10:
            errors.append(f"{case['label']}: actuator mass balance failed at t={time:.12g}.")
            break
        expected = input_schedule.lookup(time)
        if abs(pair["upper"] - float(expected["value"])) > 1e-10:
            if not actuator_value_is_consistent_near_switch(
                input_schedule,
                time,
                pair["upper"],
                signal_dt,
            ):
                errors.append(f"{case['label']}: actuator/input mismatch at t={time:.12g}.")
                break

    return errors


def actuator_value_is_consistent_near_switch(input_schedule, time, value, signal_dt):
    """Handle printed-time ambiguity at ZOH jumps.

    Kratos writes actuator times with finite precision. At a discontinuity, the
    internal time used by the CSV controller can be infinitesimally before the
    switch while the printed time rounds to the switch sample. Away from switch
    samples this still returns False, so real input mismatches remain audited.
    """
    center_index = bisect_right(input_schedule.times, round(time, 12)) - 1
    start_index = max(0, center_index - 3)
    end_index = min(len(input_schedule.rows), center_index + 5)
    nearby_rows = input_schedule.rows[start_index:end_index]

    near_switch = any(
        int(row.get("is_switch_sample", 0)) == 1
        and abs(row["time_float"] - time) <= 2.0 * signal_dt + 1e-12
        for row in nearby_rows
    )
    if not near_switch:
        return False

    return any(
        abs(value - float(row["value"])) <= 1e-10
        for row in nearby_rows
    )


def audit_training_coverage(cases):
    errors = []
    train_cases = [case for case in cases if case["role"] == "train"]
    occupancy = Counter()
    transitions = Counter()
    for case in train_cases:
        switch_start = case["switching_start_time"]
        switch_end = case["switching_end_time"]
        switch_segments = [
            segment
            for segment in case["segments"]
            if segment["end_time"] > switch_start + 1e-12
            and segment["start_time"] < switch_end - 1e-12
        ]
        previous_factor = None
        for segment in switch_segments:
            factor = segment["level_factor"]
            duration = min(segment["end_time"], switch_end) - max(segment["start_time"], switch_start)
            occupancy[factor] += max(0.0, duration)
            if previous_factor is not None and factor != previous_factor:
                transitions[(previous_factor, factor)] += 1
            previous_factor = factor

    missing_levels = [factor for factor in LEVEL_FACTORS if occupancy[factor] <= 0.0]
    if missing_levels:
        errors.append(f"Training coverage misses levels: {missing_levels}")

    positive_occupancies = [occupancy[factor] for factor in LEVEL_FACTORS if occupancy[factor] > 0.0]
    if positive_occupancies:
        ratio = max(positive_occupancies) / min(positive_occupancies)
        if ratio > 1.45:
            errors.append(f"Training level occupancy is imbalanced; max/min ratio is {ratio:.3f}.")

    missing_transitions = [
        (source, target)
        for source in LEVEL_FACTORS
        for target in LEVEL_FACTORS
        if source != target and transitions[(source, target)] == 0
    ]
    if missing_transitions:
        errors.append(f"Training coverage misses ordered transitions: {missing_transitions}")

    return errors


def print_authority_report(results, manifest):
    ordered_results = sort_results_by_manifest(results, manifest)
    print("Authority report:")
    for result in ordered_results:
        print(
            "  "
            f"{result['label']}: "
            f"max|tip_y|={result['max_abs_tip_y']:.6g}, "
            f"rms_tip_y={result['rms_tip_y']:.6g}, "
            f"post_warmup_rms_tip_y={result['post_warmup_rms_tip_y']:.6g}, "
            f"min_window_rms={result['min_moving_window_tip_rms']:.6g}, "
            f"near_zero_fraction={result['near_undeformed_fraction']:.3f}, "
            f"max|u|={result['max_abs_control']:.6g}, "
            f"rms_u={result['rms_control']:.6g}"
        )

    passive = next((result for result in ordered_results if result["role"] == "passive"), None)
    controlled = [result for result in ordered_results if result["role"] != "passive"]
    if (
        passive
        and controlled
        and not manifest.get("dry_run", False)
        and abs(float(manifest["control_max"]) - DEFAULT_CONTROL_MAX) < 1e-12
    ):
        passive_rms = max(passive["post_warmup_rms_tip_y"], 1e-14)
        largest_relative_departure = max(
            abs(result["post_warmup_rms_tip_y"] - passive["post_warmup_rms_tip_y"]) / passive_rms
            for result in controlled
        )
        if largest_relative_departure < 0.05:
            print(
                "WARNING: Umax=0.12 produced almost no departure from the passive "
                "post-warm-up RMS tip response. The control range is probably "
                "insufficient for stabilization data; rerun deliberately with a "
                "larger KRATOS_FSI_CONTROL_MAX if needed."
            )


def make_manifest(settings, cases):
    return {
        "campaign_label": settings["campaign_label"],
        "created_at": settings["created_at"],
        "natural_frequency_estimate_hz": NATURAL_FREQUENCY_ESTIMATE_HZ,
        "natural_period": NATURAL_PERIOD,
        "control_model": "eta_dot = R(eta, u)",
        "actuator_convention": "upper control = u, lower control = -u",
        "zoh_convention": "CSV value is held from each sample time until the next sample time.",
        "control_max": settings["control_max"],
        "control_levels": get_control_levels(settings["control_max"]),
        "level_factors": LEVEL_FACTORS,
        "random_seed": settings["random_seed"],
        "train_seeds": TRAIN_SEEDS,
        "validation_seeds": VALIDATION_SEEDS,
        "challenge_seeds": CHALLENGE_SEEDS,
        "dwell_multipliers_of_T0": DWELL_MULTIPLIERS,
        "dwell_time_choices": settings["dwell_time_choices"],
        "warmup_interval": [0.0, settings["warmup_duration"]],
        "switching_interval": "case-dependent: from warmup_end_time to switching_end_time",
        "release_interval": "challenge cases: from release_start_time to end_time",
        "end_time": settings["end_time"],
        "output_interval": settings["output_interval"],
        "signal_dt": settings["signal_dt"],
        "write_paraview": settings["write_paraview"],
        "dry_run": settings.get("dry_run", False),
        "identification_snapshot_columns": {
            "X": "beam displacement measurement columns prefixed with measurement_",
            "u": ["control_u"],
            "switch_metadata": ["control_segment_id", "time_since_switch", "is_switch_sample"],
        },
        "no_overwrite_policy": "Per-case input, metadata, result, and run directories are created only if absent.",
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
            raise RuntimeError(
                f"{manifest_path} already exists for a different campaign. "
                "Use a new KRATOS_FSI_CONTROL_LABEL."
            )


def read_manifest(campaign_directory):
    manifest_path = campaign_directory / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def read_case_results(campaign_directory):
    results_directory = campaign_directory / "case_results"
    results = []
    for result_path in sorted(results_directory.glob("*.json")):
        result = json.loads(result_path.read_text())
        localize_result_paths(campaign_directory, result, result_path)
        results.append(result)
    if not results:
        raise RuntimeError(f"No case result JSON files found in {results_directory}.")
    return results


def localize_result_paths(campaign_directory, result, result_path):
    """Make downloaded campaigns auditable away from the original cluster path."""
    label = result.get("label")
    if not label:
        return

    local_run_directory = (campaign_directory / "runs" / label).resolve()
    recorded_run_directory = Path(result.get("run_directory", ""))
    if recorded_run_directory.exists() or not local_run_directory.exists():
        return

    result["run_directory"] = str(local_run_directory)
    input_path = local_run_directory / "input_timeseries.csv"
    identification_path = local_run_directory / "identification_snapshots.csv"
    if input_path.exists():
        result["input_timeseries"] = str(input_path)
    if identification_path.exists():
        result["identification_snapshots"] = str(identification_path)
    result["case_result_path"] = str(result_path.resolve())


def write_case_selection(campaign_directory, case_index, case):
    selection_directory = campaign_directory / "case_selections"
    selection_directory.mkdir(parents=True, exist_ok=True)
    selection_path = selection_directory / f"{case_index:03d}_{case['label']}.json"
    write_text_no_overwrite(selection_path, json.dumps({
        "case_index": case_index,
        "case": case,
    }, indent=4))


def select_cases(all_cases, case_index):
    if case_index is None:
        return all_cases
    if case_index < 0 or case_index >= len(all_cases):
        raise IndexError(
            f"KRATOS_FSI_CONTROL_CASE_INDEX={case_index} is outside "
            f"the case range 0..{len(all_cases) - 1}."
        )
    return [all_cases[case_index]]


def sort_results_by_manifest(results, manifest):
    order = {
        case["label"]: index
        for index, case in enumerate(manifest.get("cases", []))
    }
    return sorted(results, key=lambda result: order.get(result["label"], len(order)))


def require_columns(label, path, row, required_columns):
    missing = [column for column in required_columns if column not in row]
    if missing:
        return [f"{label}: {path} misses columns {missing}."]
    return []


class SegmentSchedule:

    def __init__(self, segments):
        self.segments = segments
        self.start_times = [segment["start_time"] for segment in segments]

    def lookup(self, time):
        index = bisect_right(self.start_times, round(time, 12)) - 1
        if index < 0:
            return self.segments[0]
        return self.segments[index]


class InputSchedule:

    def __init__(self, input_rows):
        self.rows = input_rows
        self.times = [row["time_float"] for row in input_rows]

    def lookup(self, time):
        index = bisect_right(self.times, round(time, 12)) - 1
        if index < 0:
            return self.rows[0]
        return self.rows[index]


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


def read_csv_dicts(path):
    with path.open(newline="") as input_file:
        return list(csv.DictReader(input_file))


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


def iter_sample_times(end_time, dt):
    step = 0
    while True:
        time = round(step * dt, 12)
        if time >= end_time - 1e-12:
            yield round(end_time, 12)
            break
        yield time
        step += 1


def ensure_empty_directory(path):
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"{path} already exists and is not empty. "
            "Use a new campaign label or move the old files first."
        )
    path.mkdir(parents=True, exist_ok=True)


def ensure_file_does_not_exist(path):
    if path.exists():
        raise RuntimeError(f"{path} already exists; refusing to overwrite it.")


def write_text_no_overwrite(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output_file:
        output_file.write(contents)


def get_control_levels(control_max):
    return [round(factor * control_max, 12) for factor in LEVEL_FACTORS]


def quantize_time(time, dt):
    return round(round(time / dt) * dt, 12)


def format_float(value):
    if abs(value) < 5e-15:
        value = 0.0
    return f"{value:.12g}"


def format_result(result):
    return (
        f"{result['label']}: "
        f"max|tip_y|={result['max_abs_tip_y']:.6g}, "
        f"rms_tip_y={result['rms_tip_y']:.6g}, "
        f"post_warmup_rms_tip_y={result['post_warmup_rms_tip_y']:.6g}, "
        f"min_window_rms={result['min_moving_window_tip_rms']:.6g}, "
        f"near_zero_fraction={result['near_undeformed_fraction']:.3f}, "
        f"max|u|={result['max_abs_control']:.6g}, "
        f"rms_u={result['rms_control']:.6g}"
    )


def read_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


def read_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def read_optional_int(name):
    value = os.environ.get(name)
    return None if value is None else int(value)


def read_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no", "off")


if __name__ == "__main__":
    main()
