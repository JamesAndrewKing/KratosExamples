"""Run the deterministic-startup multistate Fourier-envelope FSI2 campaign.

The physical scalar actuator command is

    u(t) = a_c(t) cos(theta(t)) + a_s(t) sin(theta(t)),
    theta_dot = Omega_c,

and is applied to the existing Rabault actuator pair as upper=u, lower=-u.
The campaign samples both the early small-amplitude departure from the
unstable-equilibrium neighborhood and the mature passive limit cycle. Kratos
always starts from its standard initial condition; no restart is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import run_control_identification_campaign as base
import run_control_parametric_manifold_campaign as pm


DEFAULT_CAMPAIGN_LABEL = "fsi_control_fourier_envelope_multistate_t40_u20"
DEFAULT_END_TIME = 40.0
DEFAULT_OUTPUT_DT = 0.01
DEFAULT_INPUT_DT = 0.002
DEFAULT_WRITE_PARAVIEW = False
DEFAULT_CARRIER_FREQUENCY_HZ = 1.90
DEFAULT_ENVELOPE_BOUND = 2.0
DEFAULT_WARMUP_END_TIME = 20.0
DEFAULT_INITIAL_KICK_AMPLITUDE = 0.4
DEFAULT_INITIAL_KICK_END_TIME = 2.0
DEFAULT_INITIAL_KICK_RAMP_TIME = 0.25
DEFAULT_ACTIVATION_RAMP_DURATION = 2.0
DEFAULT_RADIAL_RAMP_DURATION = 10.0
DEFAULT_MAX_REASONABLE_TIP_DISPLACEMENT = 0.75
DEFAULT_MAX_ENVELOPE_ACCELERATION_FOR_AUDIT = 50.0
DEFAULT_STARTUP_END_TIME = 2.5
DEFAULT_NEAR_EQUILIBRIUM_PROBE_TIME = 3.0
DEFAULT_NEAR_EQUILIBRIUM_END_TIME = 4.5
DEFAULT_EARLY_BURST_AMPLITUDE = 0.1
DEFAULT_EARLY_BURST_DURATION = 2.0
DEFAULT_EARLY_BURST_RAMP_DURATION = 0.5
DEFAULT_END_WINDOW_DURATION = 5.0
DEFAULT_SMALL_AMPLITUDE_P2P_THRESHOLD = 0.02

PHASES = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi]
EARLY_PROBE_AMPLITUDES = [0.5, 1.5]
MATURE_CONSTANT_AMPLITUDES = [1.0, 2.0]
RADIAL_RAMP_SPECS = [
    (0.0, 2.0, 0.0),
    (2.0, 0.0, 0.0),
    (0.0, 2.0, 0.5 * math.pi),
    (2.0, 0.0, 0.5 * math.pi),
]
DETUNINGS_HZ = [-0.10, -0.05, 0.05, 0.10]
TRAIN_RANDOM_SEEDS = [1, 2, 3, 4]
VALIDATION_RANDOM_SEEDS = [101, 102]
CHALLENGE_CASES = [
    ("challenge_constant_A1p500_phi_pi4", "constant_quadrature", {"amplitude": 1.5, "phi": 0.25 * math.pi}),
    ("challenge_rotating_A1p500_df_m0p075", "rotating_quadrature", {"amplitude": 1.5, "detuning_hz": -0.075}),
    ("challenge_rotating_A1p500_df_p0p075", "rotating_quadrature", {"amplitude": 1.5, "detuning_hz": 0.075}),
]

PYTHONPATH = os.environ.get("KRATOS_FSI_PYTHONPATH")
DYLD_LIBRARY_PATH = os.environ.get("KRATOS_FSI_DYLD_LIBRARY_PATH")
LD_LIBRARY_PATH = os.environ.get("KRATOS_FSI_LD_LIBRARY_PATH")

CONTROL_COLUMNS = [
    "control_u",
    "control_u_dot",
    "carrier_theta_rad",
    "carrier_theta_unwrapped_rad",
    "carrier_omega_rad_s",
    "envelope_ac",
    "envelope_as",
    "envelope_ac_dot",
    "envelope_as_dot",
    "envelope_amplitude",
    "envelope_phase_rad",
    "use_for_identification",
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
    *CONTROL_COLUMNS,
]

SUMMARY_FIELDNAMES = [
    "label",
    "role",
    "signal_type",
    "random_seed",
    "initialization_protocol",
    "initial_state_source",
    "startup_end_time",
    "near_equilibrium_probe_time",
    "near_equilibrium_end_time",
    "run_directory",
    "identification_snapshots",
    "input_timeseries",
    "case_result_path",
    "end_time",
    "output_dt",
    "input_dt",
    "carrier_frequency_hz",
    "carrier_omega_rad_s",
    "envelope_bound",
    "envelope_bandwidth_hz",
    "controlled_start_time",
    "use_for_identification_start_time",
    "perturbation_amplitude",
    "perturbation_phase",
    "carrier_phase_at_control_activation",
    "amplitude",
    "phi",
    "amplitude_start",
    "amplitude_end",
    "ramp_duration",
    "detuning_hz",
    "physical_frequency_hz",
    "max_abs_u",
    "max_abs_u_dot",
    "max_abs_envelope_ac_dot",
    "max_abs_envelope_as_dot",
    "max_envelope_amplitude",
    "min_u",
    "max_u",
    "last_time",
    "number_of_samples",
    "max_abs_tip_y",
    "rms_tip_y",
    "final_tip_y",
    "tip_peak_to_peak",
    "early_probe_rms_tip_y",
    "early_probe_peak_to_peak",
    "early_probe_tip_min",
    "early_probe_tip_max",
    "end_window_rms_tip_y",
    "end_window_peak_to_peak",
    "end_window_tip_min",
    "end_window_tip_max",
    "leaves_small_amplitude_neighborhood",
    "moves_trajectory_inward",
    "early_mature_state_range_overlap",
    "max_abs_control",
    "rms_control",
]


def main() -> None:
    args = parse_arguments()
    source_directory = Path(__file__).resolve().parent
    os.chdir(source_directory)

    if args.audit is not None:
        campaign_directory = Path(args.audit)
        manifest = read_manifest(campaign_directory)
        results = read_case_results(campaign_directory)
        write_summary_files(campaign_directory, results, manifest)
        audit_campaign(campaign_directory, manifest, results)
        print_report(results, manifest)
        return

    if args.analyze_startup is not None:
        report = analyze_passive_startup(args.analyze_startup, read_settings())
        print_startup_report(report)
        return

    settings = read_settings()
    cases = build_case_library(settings)

    if args.pilot_case is not None:
        run_truncated_pilot(args, settings, cases)
        return

    if args.list_cases:
        for index, case in enumerate(cases):
            print(f"{index:03d} {case['label']} [{case['role']}/{case['signal_type']}]")
        print(f"number_of_cases={len(cases)}")
        return

    if args.dry_run is not None:
        campaign_directory = Path(args.dry_run)
        run_dry_campaign(campaign_directory, settings, cases)
        manifest = read_manifest(campaign_directory)
        results = read_case_results(campaign_directory)
        write_summary_files(campaign_directory, results, manifest)
        audit_campaign(campaign_directory, manifest, results)
        print_report(results, manifest)
        return

    campaign_directory = Path("run_outputs") / settings["campaign_label"]
    campaign_directory.mkdir(parents=True, exist_ok=True)
    write_manifest_once(campaign_directory, settings, cases)

    selected_cases = select_cases(cases)
    for index, case in selected_cases:
        print(f"[{index}] {case['label']}", flush=True)
        pm.write_case_selection(campaign_directory, index, case)
        run_directory = run_case(case, campaign_directory, settings)
        result = collect_case_result(case, campaign_directory, run_directory, settings)
        write_case_result(campaign_directory, result)
        print(format_result(result), flush=True)

    if len(selected_cases) == len(cases):
        manifest = make_manifest(settings, cases)
        results = read_case_results(campaign_directory)
        write_summary_files(campaign_directory, results, manifest)
        audit_campaign(campaign_directory, manifest, results)
        print_report(results, manifest)

    print(f"campaign={campaign_directory.resolve()}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or audit the FSI2 Fourier-envelope control campaign."
    )
    parser.add_argument("--dry-run", type=Path, help="Generate synthetic files and audit them.")
    parser.add_argument("--audit", type=Path, help="Audit an existing campaign directory.")
    parser.add_argument("--list-cases", action="store_true", help="List case indices and labels.")
    parser.add_argument(
        "--analyze-startup",
        type=Path,
        help="Analyze a passive beam_displacement_timeseries.csv and verify the early window.",
    )
    parser.add_argument("--pilot-case", help="Run one early-growth case as a truncated real Kratos pilot.")
    parser.add_argument("--pilot-end-time", type=float, default=6.0)
    parser.add_argument("--pilot-directory", type=Path)
    parser.add_argument(
        "--passive-reference",
        type=Path,
        help="Optional passive beam CSV used to quantify the controlled pilot departure.",
    )
    return parser.parse_args()


def run_truncated_pilot(args: argparse.Namespace, settings: dict, cases: list[dict]) -> None:
    matches = [case for case in cases if case["label"] == args.pilot_case]
    if not matches:
        raise ValueError(f"Unknown pilot case {args.pilot_case!r}.")
    case = matches[0]
    if case["initialization_protocol"] != "early_growth":
        raise ValueError("Truncated pilots are restricted to early-growth cases.")
    if args.pilot_end_time <= settings["near_equilibrium_end_time"]:
        raise ValueError("pilot-end-time must exceed near_equilibrium_end_time.")

    pilot_settings = dict(settings)
    pilot_settings["end_time"] = float(args.pilot_end_time)
    pilot_settings["campaign_label"] = f"{settings['campaign_label']}_pilot"
    pilot_directory = args.pilot_directory or (
        Path("run_outputs") / f"{pilot_settings['campaign_label']}_{case['label']}"
    )
    base.ensure_empty_directory(pilot_directory)
    run_directory = run_case(case, pilot_directory, pilot_settings)
    result = collect_case_result(case, pilot_directory, run_directory, pilot_settings)
    report = {
        "case_result": result,
        "pilot_end_time": pilot_settings["end_time"],
        "initial_state_source": "deterministic_standard_startup",
        "interpretation": "early controlled probe; no unstable-equilibrium restart",
    }
    if args.passive_reference is not None:
        report["controlled_vs_passive"] = compare_pilot_to_passive(
            run_directory / "beam_displacement_timeseries.csv",
            args.passive_reference,
            settings["near_equilibrium_probe_time"],
            pilot_settings["end_time"],
        )
    (pilot_directory / "pilot_report.json").write_text(json.dumps(report, indent=4))
    print(json.dumps(report, indent=2))
    print(f"pilot={pilot_directory.resolve()}")


def compare_pilot_to_passive(
    controlled_path: Path,
    passive_path: Path,
    start_time: float,
    end_time: float,
) -> dict:
    controlled = {
        round(time_value, 12): value
        for time_value, value in read_tip_samples(controlled_path)
        if start_time - 1e-12 <= time_value <= end_time + 1e-12
    }
    passive = {
        round(time_value, 12): value
        for time_value, value in read_tip_samples(passive_path)
        if start_time - 1e-12 <= time_value <= end_time + 1e-12
    }
    common_times = sorted(set(controlled) & set(passive))
    if not common_times:
        raise RuntimeError("Pilot and passive reference have no aligned samples.")
    differences = [controlled[time_value] - passive[time_value] for time_value in common_times]
    passive_values = [passive[time_value] for time_value in common_times]
    controlled_values = [controlled[time_value] for time_value in common_times]
    max_abs_difference = max(abs(value) for value in differences)
    passive_p2p = peak_to_peak(passive_values)
    measurable_threshold = max(5e-4, 0.05 * passive_p2p)
    return {
        "comparison_start_time": start_time,
        "comparison_end_time": end_time,
        "number_of_aligned_samples": len(common_times),
        "controlled_peak_to_peak_tip_y": peak_to_peak(controlled_values),
        "passive_peak_to_peak_tip_y": passive_p2p,
        "rms_controlled_minus_passive_tip_y": rms(differences),
        "max_abs_controlled_minus_passive_tip_y": max_abs_difference,
        "measurable_threshold": measurable_threshold,
        "measurable_departure": max_abs_difference > measurable_threshold,
        "immediate_mature_cycle_jump": peak_to_peak(controlled_values) > 0.8 * 0.16415,
    }


def read_settings() -> dict:
    carrier_frequency = read_float("KRATOS_FSI_FE_CARRIER_FREQUENCY_HZ", DEFAULT_CARRIER_FREQUENCY_HZ)
    output_dt = read_float("KRATOS_FSI_FE_OUTPUT_INTERVAL", DEFAULT_OUTPUT_DT)
    return {
        "campaign_label": os.environ.get("KRATOS_FSI_FE_LABEL", DEFAULT_CAMPAIGN_LABEL),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "end_time": read_float("KRATOS_FSI_FE_END_TIME", DEFAULT_END_TIME),
        "output_dt": output_dt,
        "input_dt": read_float("KRATOS_FSI_FE_SIGNAL_DT", DEFAULT_INPUT_DT),
        "write_paraview": read_bool("KRATOS_FSI_FE_WRITE_PARAVIEW", DEFAULT_WRITE_PARAVIEW),
        "carrier_frequency_hz": carrier_frequency,
        "carrier_omega_rad_s": 2.0 * math.pi * carrier_frequency,
        "carrier_period": 1.0 / carrier_frequency,
        "envelope_bound": read_float("KRATOS_FSI_FE_ENVELOPE_BOUND", DEFAULT_ENVELOPE_BOUND),
        "warmup_end_time": read_float("KRATOS_FSI_FE_WARMUP_END_TIME", DEFAULT_WARMUP_END_TIME),
        "initial_kick_amplitude": read_float("KRATOS_FSI_FE_INITIAL_KICK_AMPLITUDE", DEFAULT_INITIAL_KICK_AMPLITUDE),
        "initial_kick_end_time": read_float("KRATOS_FSI_FE_INITIAL_KICK_END_TIME", DEFAULT_INITIAL_KICK_END_TIME),
        "initial_kick_ramp_time": read_float("KRATOS_FSI_FE_INITIAL_KICK_RAMP_TIME", DEFAULT_INITIAL_KICK_RAMP_TIME),
        "activation_ramp_duration": read_float("KRATOS_FSI_FE_ACTIVATION_RAMP_DURATION", DEFAULT_ACTIVATION_RAMP_DURATION),
        "radial_ramp_duration": read_float("KRATOS_FSI_FE_RADIAL_RAMP_DURATION", DEFAULT_RADIAL_RAMP_DURATION),
        "max_reasonable_tip_displacement": read_float(
            "KRATOS_FSI_FE_MAX_REASONABLE_TIP_DISPLACEMENT",
            DEFAULT_MAX_REASONABLE_TIP_DISPLACEMENT,
        ),
        "max_envelope_acceleration_for_audit": DEFAULT_MAX_ENVELOPE_ACCELERATION_FOR_AUDIT,
        "startup_end_time": read_float("KRATOS_FSI_FE_MS_STARTUP_END_TIME", DEFAULT_STARTUP_END_TIME),
        "near_equilibrium_probe_time": read_float(
            "KRATOS_FSI_FE_MS_NEAR_EQUILIBRIUM_PROBE_TIME",
            DEFAULT_NEAR_EQUILIBRIUM_PROBE_TIME,
        ),
        "near_equilibrium_end_time": read_float(
            "KRATOS_FSI_FE_MS_NEAR_EQUILIBRIUM_END_TIME",
            DEFAULT_NEAR_EQUILIBRIUM_END_TIME,
        ),
        "early_burst_amplitude": read_float(
            "KRATOS_FSI_FE_MS_EARLY_BURST_AMPLITUDE",
            DEFAULT_EARLY_BURST_AMPLITUDE,
        ),
        "early_burst_duration": read_float(
            "KRATOS_FSI_FE_MS_EARLY_BURST_DURATION",
            DEFAULT_EARLY_BURST_DURATION,
        ),
        "early_burst_ramp_duration": read_float(
            "KRATOS_FSI_FE_MS_EARLY_BURST_RAMP_DURATION",
            DEFAULT_EARLY_BURST_RAMP_DURATION,
        ),
        "end_window_duration": DEFAULT_END_WINDOW_DURATION,
        "small_amplitude_p2p_threshold": read_float(
            "KRATOS_FSI_FE_MS_SMALL_AMPLITUDE_P2P_THRESHOLD",
            DEFAULT_SMALL_AMPLITUDE_P2P_THRESHOLD,
        ),
    }


def build_case_library(settings: dict) -> list[dict]:
    if not (
        2.0 <= settings["startup_end_time"]
        < settings["near_equilibrium_probe_time"]
        < settings["near_equilibrium_end_time"]
        < settings["warmup_end_time"]
        < settings["end_time"]
    ):
        raise ValueError(
            "Require 2 <= startup_end_time < near_equilibrium_probe_time < "
            "near_equilibrium_end_time < warmup_end_time < end_time."
        )
    if not 0.0 < settings["early_burst_amplitude"] <= 0.2 + 1e-12:
        raise ValueError("Early burst amplitude must be in (0, 0.2].")
    cases = [make_case(
        settings,
        "early_passive_reference",
        "reference",
        "passive",
        {},
        "early_growth",
    )]

    for phi, label_suffix in [(0.0, "phase0"), (0.5 * math.pi, "phase_pi2")]:
        cases.append(make_case(
            settings,
            f"early_growth_kick_{label_suffix}",
            "train",
            "early_burst",
            {
                "amplitude": settings["early_burst_amplitude"],
                "phi": phi,
                "burst_duration": settings["early_burst_duration"],
                "burst_ramp_duration": settings["early_burst_ramp_duration"],
            },
            "early_growth",
        ))

    for amplitude in EARLY_PROBE_AMPLITUDES:
        for phi in PHASES:
            label = f"early_probe_A{format_positive(amplitude)}_phi_{format_phase_label(phi)}"
            cases.append(make_case(
                settings,
                label,
                "train",
                "constant_quadrature",
                {"amplitude": amplitude, "phi": phi},
                "early_growth",
            ))

    for amplitude in MATURE_CONSTANT_AMPLITUDES:
        for phi in PHASES:
            label = f"mature_constant_A{format_positive(amplitude)}_phi_{format_phase_label(phi)}"
            cases.append(make_case(
                settings,
                label,
                "train",
                "constant_quadrature",
                {"amplitude": amplitude, "phi": phi},
                "mature_limit_cycle",
            ))

    for amplitude_start, amplitude_end, phi in RADIAL_RAMP_SPECS:
        label = (
            f"radial_ramp_A{format_positive(amplitude_start)}_to_"
            f"A{format_positive(amplitude_end)}_phi_{format_phase_label(phi)}"
        )
        cases.append(make_case(
            settings,
            label,
            "train",
            "radial_ramp",
            {
                "amplitude_start": amplitude_start,
                "amplitude_end": amplitude_end,
                "phi": phi,
                "ramp_duration": settings["radial_ramp_duration"],
            },
            "mature_limit_cycle",
        ))

    for detuning in DETUNINGS_HZ:
        label = f"mature_rotating_A1p500_df_{format_signed(detuning, 3)}"
        cases.append(make_case(
            settings,
            label,
            "train",
            "rotating_quadrature",
            {"amplitude": 1.5, "detuning_hz": detuning},
            "mature_limit_cycle",
        ))

    for seed in TRAIN_RANDOM_SEEDS:
        cases.append(make_case(
            settings,
            f"train_random_envelope_seed{seed:02d}",
            "train",
            "random_fourier_envelope",
            {"random_seed": seed},
            "mature_limit_cycle",
        ))
    for seed in VALIDATION_RANDOM_SEEDS:
        cases.append(make_case(
            settings,
            f"validation_random_envelope_seed{seed:03d}",
            "validation",
            "random_fourier_envelope",
            {"random_seed": seed},
            "mature_limit_cycle",
        ))

    for label, signal_type, metadata in CHALLENGE_CASES:
        cases.append(make_case(
            settings,
            label,
            "challenge",
            signal_type,
            metadata,
            "mature_limit_cycle",
        ))

    if len(cases) != 36:
        raise RuntimeError(f"Expected 36 multistate cases, built {len(cases)}.")
    return cases


def make_case(
    settings: dict,
    label: str,
    role: str,
    signal_type: str,
    metadata: dict,
    initialization_protocol: str,
) -> dict:
    if initialization_protocol not in ("early_growth", "mature_limit_cycle"):
        raise ValueError(f"Unknown initialization protocol {initialization_protocol!r}.")
    controlled_start = pm.quantize_time(
        settings["near_equilibrium_probe_time"]
        if initialization_protocol == "early_growth"
        else settings["warmup_end_time"],
        settings["output_dt"],
    )
    identification_start = (
        settings["startup_end_time"]
        if initialization_protocol == "early_growth"
        else controlled_start
    )
    metadata = dict(metadata)
    metadata["apply_initial_kick"] = initialization_protocol == "mature_limit_cycle"
    if signal_type == "random_fourier_envelope":
        controlled_duration = max(settings["input_dt"], settings["end_time"] - controlled_start)
        metadata["fourier_coefficients"] = random_coefficients(
            metadata["random_seed"],
            controlled_duration,
        )
    return {
        "label": label,
        "role": role,
        "kind": signal_type,
        "signal_type": signal_type,
        "controller": "csv",
        "csv_interpolation": "linear",
        "initialization_protocol": initialization_protocol,
        "initial_state_source": "deterministic_standard_startup",
        "startup_end_time": settings["startup_end_time"],
        "near_equilibrium_probe_time": settings["near_equilibrium_probe_time"],
        "near_equilibrium_end_time": settings["near_equilibrium_end_time"],
        "controlled_start_time": controlled_start,
        "use_for_identification_start_time": identification_start,
        "kick_or_perturbation_amplitude": (
            metadata.get("amplitude", 0.0)
            if initialization_protocol == "early_growth"
            else settings["initial_kick_amplitude"]
        ),
        "kick_or_perturbation_phase": (
            metadata.get("phi", 0.0) if initialization_protocol == "early_growth" else 0.0
        ),
        "carrier_phase_at_control_activation": wrap_phase(
            settings["carrier_omega_rad_s"] * controlled_start
        ),
        "metadata": {
            "carrier_frequency_hz": settings["carrier_frequency_hz"],
            "carrier_omega_rad_s": settings["carrier_omega_rad_s"],
            "physical_frequency_sign_convention": (
                "For rotating quadratures, physical frequency = "
                "f_c - psi_dot/(2*pi); detuning_hz = physical_frequency_hz - f_c."
            ),
            **metadata,
        },
    }


def run_case(case: dict, campaign_directory: Path, settings: dict) -> Path:
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
        "KRATOS_FSI_ACTUATOR_CSV_INTERPOLATION": "linear",
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


def run_dry_campaign(campaign_directory: Path, settings: dict, cases: list[dict]) -> None:
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


def write_input_timeseries_csv(path: Path, case: dict, settings: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base.ensure_file_does_not_exist(path)
    with path.open("x", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "time",
            "value",
            *CONTROL_COLUMNS,
            "control_kind",
        ])
        for time_value in base.iter_sample_times(settings["end_time"], settings["input_dt"]):
            row = evaluate_control(case, settings, time_value)
            writer.writerow([
                base.format_float(time_value),
                format_control_value(row["control_u"]),
                *[format_control_value(row[column]) for column in CONTROL_COLUMNS],
                row["control_kind"],
            ])


def write_identification_snapshots_csv(beam_path: Path, input_path: Path, output_path: Path) -> None:
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
            *CONTROL_COLUMNS,
            "control_kind",
        ])
        for beam_row in beam_rows:
            time_value = float(beam_row[0])
            input_row = base.find_input_row(input_rows, input_rows_by_time, time_value)
            writer.writerow([
                base.format_float(time_value),
                *beam_row[1:],
                *[input_row[column] for column in CONTROL_COLUMNS],
                input_row["control_kind"],
            ])

    metadata_path = output_path.with_suffix(".metadata.csv")
    base.ensure_file_does_not_exist(metadata_path)
    with metadata_path.open("x", newline="") as metadata_file:
        writer = csv.writer(metadata_file)
        writer.writerow(metadata)


def evaluate_control(case: dict, settings: dict, time_value: float) -> dict:
    theta_unwrapped = settings["carrier_omega_rad_s"] * time_value
    theta = theta_unwrapped % (2.0 * math.pi)
    ac, ac_dot, a_s, as_dot, control_kind = evaluate_envelope(case, settings, time_value)
    control = ac * math.cos(theta_unwrapped) + a_s * math.sin(theta_unwrapped)
    control_dot = (
        ac_dot * math.cos(theta_unwrapped)
        + as_dot * math.sin(theta_unwrapped)
        + settings["carrier_omega_rad_s"] * (-ac * math.sin(theta_unwrapped) + a_s * math.cos(theta_unwrapped))
    )
    amplitude = math.hypot(ac, a_s)
    phase = 0.0 if amplitude <= 1e-14 else math.atan2(a_s, ac)
    return {
        "control_u": control,
        "control_u_dot": control_dot,
        "carrier_theta_rad": theta,
        "carrier_theta_unwrapped_rad": theta_unwrapped,
        "carrier_omega_rad_s": settings["carrier_omega_rad_s"],
        "envelope_ac": ac,
        "envelope_as": a_s,
        "envelope_ac_dot": ac_dot,
        "envelope_as_dot": as_dot,
        "envelope_amplitude": amplitude,
        "envelope_phase_rad": phase,
        "use_for_identification": int(time_value + 1e-12 >= case["use_for_identification_start_time"]),
        "control_kind": control_kind,
    }


def evaluate_envelope(case: dict, settings: dict, time_value: float) -> tuple[float, float, float, float, str]:
    if case["metadata"].get("apply_initial_kick", True):
        kick = evaluate_initial_kick(settings, time_value)
        if kick is not None:
            amplitude, amplitude_dot = kick
            return amplitude, amplitude_dot, 0.0, 0.0, "fourier_initial_kick"
    if time_value < case["controlled_start_time"] - 1e-12:
        return 0.0, 0.0, 0.0, 0.0, "passive_warmup"

    metadata = case["metadata"]
    signal_type = case["signal_type"]
    local_time = max(0.0, time_value - case["controlled_start_time"])

    if signal_type == "passive":
        return 0.0, 0.0, 0.0, 0.0, "passive_reference"
    if signal_type == "early_burst":
        return early_burst(local_time, metadata)
    if signal_type == "constant_quadrature":
        return constant_quadrature(settings, local_time, metadata["amplitude"], metadata["phi"], "constant_quadrature")
    if signal_type == "radial_ramp":
        return radial_ramp(settings, local_time, metadata, "radial_ramp")
    if signal_type == "rotating_quadrature":
        return rotating_quadrature(settings, local_time, metadata, "rotating_quadrature")
    if signal_type == "random_fourier_envelope":
        return random_fourier_envelope(settings, local_time, metadata["fourier_coefficients"])
    raise ValueError(f"Unsupported signal_type {signal_type!r}.")


def early_burst(local_time: float, metadata: dict) -> tuple[float, float, float, float, str]:
    amplitude = float(metadata["amplitude"])
    phi = float(metadata["phi"])
    duration = float(metadata["burst_duration"])
    ramp_duration = min(float(metadata["burst_ramp_duration"]), 0.5 * duration)
    if local_time < 0.0 or local_time > duration:
        radial = 0.0
        radial_dot = 0.0
    elif local_time <= ramp_duration:
        shape, shape_prime = quintic_smoothstep(local_time / ramp_duration)
        radial = amplitude * shape
        radial_dot = amplitude * shape_prime / ramp_duration
    elif local_time >= duration - ramp_duration:
        shape, shape_prime = quintic_smoothstep(
            (local_time - (duration - ramp_duration)) / ramp_duration
        )
        radial = amplitude * (1.0 - shape)
        radial_dot = -amplitude * shape_prime / ramp_duration
    else:
        radial = amplitude
        radial_dot = 0.0
    control_kind = "post_burst_passive_growth" if local_time > duration else "early_smooth_burst"
    return (
        radial * math.cos(phi),
        radial_dot * math.cos(phi),
        radial * math.sin(phi),
        radial_dot * math.sin(phi),
        control_kind,
    )


def evaluate_initial_kick(settings: dict, time_value: float) -> tuple[float, float] | None:
    end_time = settings["initial_kick_end_time"]
    if time_value < -1e-12 or time_value > end_time + 1e-12:
        return None
    ramp = min(settings["initial_kick_ramp_time"], 0.5 * end_time)
    amplitude = settings["initial_kick_amplitude"]
    if time_value <= ramp + 1e-12:
        shape, shape_dot = quintic_smoothstep(time_value / ramp)
        return amplitude * shape, amplitude * shape_dot / ramp
    if time_value >= end_time - ramp - 1e-12:
        tau = (time_value - (end_time - ramp)) / ramp
        shape, shape_dot = quintic_smoothstep(tau)
        return amplitude * (1.0 - shape), -amplitude * shape_dot / ramp
    return amplitude, 0.0


def constant_quadrature(settings: dict, local_time: float, amplitude: float, phi: float, control_kind: str):
    ramp_duration = settings["activation_ramp_duration"]
    shape, shape_dot = activation_shape(local_time, ramp_duration)
    current_amplitude = amplitude * shape
    amplitude_dot = amplitude * shape_dot
    return (
        current_amplitude * math.cos(phi),
        amplitude_dot * math.cos(phi),
        current_amplitude * math.sin(phi),
        amplitude_dot * math.sin(phi),
        control_kind,
    )


def radial_ramp(settings: dict, local_time: float, metadata: dict, control_kind: str):
    phi = metadata["phi"]
    a0 = metadata["amplitude_start"]
    a1 = metadata["amplitude_end"]
    ramp_duration = metadata.get("ramp_duration", settings["radial_ramp_duration"])
    if a0 > 1e-12:
        activation_duration = settings["activation_ramp_duration"]
        if local_time < activation_duration - 1e-12:
            shape, shape_dot = activation_shape(local_time, activation_duration)
            amplitude = a0 * shape
            amplitude_dot = a0 * shape_dot
        else:
            ramp_time = local_time - activation_duration
            shape, shape_dot = activation_shape(ramp_time, ramp_duration)
            amplitude = a0 + (a1 - a0) * shape
            amplitude_dot = (a1 - a0) * shape_dot
    else:
        shape, shape_dot = activation_shape(local_time, ramp_duration)
        amplitude = a0 + (a1 - a0) * shape
        amplitude_dot = (a1 - a0) * shape_dot
    return (
        amplitude * math.cos(phi),
        amplitude_dot * math.cos(phi),
        amplitude * math.sin(phi),
        amplitude_dot * math.sin(phi),
        control_kind,
    )


def rotating_quadrature(settings: dict, local_time: float, metadata: dict, control_kind: str):
    amplitude = metadata["amplitude"]
    detuning_hz = metadata["detuning_hz"]
    psi_dot = -2.0 * math.pi * detuning_hz
    ramp_duration = settings["activation_ramp_duration"]
    shape, shape_dot = activation_shape(local_time, ramp_duration)
    psi = psi_dot * local_time
    radial = amplitude * shape
    radial_dot = amplitude * shape_dot
    return (
        radial * math.cos(psi),
        radial_dot * math.cos(psi) - radial * math.sin(psi) * psi_dot,
        radial * math.sin(psi),
        radial_dot * math.sin(psi) + radial * math.cos(psi) * psi_dot,
        control_kind,
    )


def random_fourier_envelope(settings: dict, local_time: float, coefficients: dict):
    controlled_duration = coefficients["duration"]
    raw_ac, raw_ac_dot, raw_as, raw_as_dot = evaluate_random_series(coefficients, local_time, controlled_duration)
    scale = coefficients["scale"]
    ramp_shape, ramp_dot = activation_shape(local_time, settings["activation_ramp_duration"])
    ac = scale * ramp_shape * raw_ac
    a_s = scale * ramp_shape * raw_as
    ac_dot = scale * (ramp_dot * raw_ac + ramp_shape * raw_ac_dot)
    as_dot = scale * (ramp_dot * raw_as + ramp_shape * raw_as_dot)
    return ac, ac_dot, a_s, as_dot, "random_fourier_envelope"


def random_coefficients(seed: int, duration: float) -> dict:
    rng = random.Random(seed)
    harmonics = [1, 2, 3]
    coefficients = {
        "harmonics": harmonics,
        "duration": duration,
        "ac_cos": [rng.uniform(-1.0, 1.0) for _ in harmonics],
        "ac_sin": [rng.uniform(-1.0, 1.0) for _ in harmonics],
        "as_cos": [rng.uniform(-1.0, 1.0) for _ in harmonics],
        "as_sin": [rng.uniform(-1.0, 1.0) for _ in harmonics],
        "ac_bias": rng.uniform(-0.4, 0.4),
        "as_bias": rng.uniform(-0.4, 0.4),
    }
    max_radius = 0.0
    for i in range(2001):
        time_value = duration * i / 2000
        ac, _, a_s, _ = evaluate_random_series({**coefficients, "scale": 1.0}, time_value, duration)
        max_radius = max(max_radius, math.hypot(ac, a_s))
    coefficients["scale"] = 1.8 / max(max_radius, 1e-12)
    return coefficients


def evaluate_random_series(coefficients: dict, local_time: float, duration: float) -> tuple[float, float, float, float]:
    ac = coefficients["ac_bias"]
    a_s = coefficients["as_bias"]
    ac_dot = 0.0
    as_dot = 0.0
    for index, harmonic in enumerate(coefficients["harmonics"]):
        omega = 2.0 * math.pi * harmonic / duration
        angle = omega * local_time
        cos_value = math.cos(angle)
        sin_value = math.sin(angle)
        ac += coefficients["ac_cos"][index] * cos_value + coefficients["ac_sin"][index] * sin_value
        a_s += coefficients["as_cos"][index] * cos_value + coefficients["as_sin"][index] * sin_value
        ac_dot += -omega * coefficients["ac_cos"][index] * sin_value + omega * coefficients["ac_sin"][index] * cos_value
        as_dot += -omega * coefficients["as_cos"][index] * sin_value + omega * coefficients["as_sin"][index] * cos_value
    return ac, ac_dot, a_s, as_dot


def activation_shape(local_time: float, duration: float) -> tuple[float, float]:
    if duration <= 0.0 or local_time >= duration:
        return 1.0, 0.0
    if local_time <= 0.0:
        return 0.0, 0.0
    shape, shape_prime = quintic_smoothstep(local_time / duration)
    return shape, shape_prime / duration


def quintic_smoothstep(tau: float) -> tuple[float, float]:
    tau = min(1.0, max(0.0, tau))
    value = tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)
    derivative = 30.0 * tau ** 2 * (1.0 - tau) ** 2
    return value, derivative


def collect_case_result(case: dict, campaign_directory: Path, run_directory: Path, settings: dict) -> dict:
    metadata = case["metadata"]
    metrics = collect_tip_metrics(run_directory, settings)
    input_metrics = collect_input_metrics(run_directory / "input_timeseries.csv")
    activation_phase = wrap_phase(
        settings["carrier_omega_rad_s"] * case["controlled_start_time"]
    )
    return {
        "label": case["label"],
        "role": case["role"],
        "signal_type": case["signal_type"],
        "random_seed": metadata.get("random_seed", ""),
        "initialization_protocol": case["initialization_protocol"],
        "initial_state_source": case["initial_state_source"],
        "startup_end_time": settings["startup_end_time"],
        "near_equilibrium_probe_time": settings["near_equilibrium_probe_time"],
        "near_equilibrium_end_time": settings["near_equilibrium_end_time"],
        "end_time": settings["end_time"],
        "output_dt": settings["output_dt"],
        "input_dt": settings["input_dt"],
        "carrier_frequency_hz": settings["carrier_frequency_hz"],
        "carrier_omega_rad_s": settings["carrier_omega_rad_s"],
        "envelope_bound": settings["envelope_bound"],
        "envelope_bandwidth_hz": envelope_bandwidth(case, settings),
        "controlled_start_time": case["controlled_start_time"],
        "use_for_identification_start_time": case["use_for_identification_start_time"],
        "perturbation_amplitude": metadata.get("amplitude", "") if case["initialization_protocol"] == "early_growth" else settings["initial_kick_amplitude"],
        "perturbation_phase": metadata.get("phi", "") if case["initialization_protocol"] == "early_growth" else 0.0,
        "carrier_phase_at_control_activation": activation_phase,
        "amplitude": metadata.get("amplitude", ""),
        "phi": metadata.get("phi", ""),
        "amplitude_start": metadata.get("amplitude_start", ""),
        "amplitude_end": metadata.get("amplitude_end", ""),
        "ramp_duration": metadata.get("ramp_duration", ""),
        "detuning_hz": metadata.get("detuning_hz", ""),
        "physical_frequency_hz": (
            settings["carrier_frequency_hz"] + metadata["detuning_hz"]
            if "detuning_hz" in metadata else ""
        ),
        **input_metrics,
        **metrics,
        "run_directory": str(run_directory.resolve()),
        "input_timeseries": str((run_directory / "input_timeseries.csv").resolve()),
        "identification_snapshots": str((run_directory / "identification_snapshots.csv").resolve()),
        "case_result_path": str((campaign_directory / "case_results" / f"{case['label']}.json").resolve()),
    }


def collect_tip_metrics(run_directory: Path, settings: dict) -> dict:
    beam_path = run_directory / "beam_displacement_timeseries.csv"
    actuator_path = run_directory / "actuator_timeseries.csv"
    metrics = {
        "last_time": 0.0,
        "number_of_samples": 0,
        "max_abs_tip_y": 0.0,
        "rms_tip_y": 0.0,
        "final_tip_y": 0.0,
        "tip_peak_to_peak": 0.0,
        "early_probe_rms_tip_y": 0.0,
        "early_probe_peak_to_peak": 0.0,
        "early_probe_tip_min": 0.0,
        "early_probe_tip_max": 0.0,
        "end_window_rms_tip_y": 0.0,
        "end_window_peak_to_peak": 0.0,
        "end_window_tip_min": 0.0,
        "end_window_tip_max": 0.0,
        "leaves_small_amplitude_neighborhood": False,
        "moves_trajectory_inward": "",
        "early_mature_state_range_overlap": "",
        "max_abs_control": 0.0,
        "rms_control": 0.0,
    }
    samples = []
    with beam_path.open(newline="") as beam_file:
        reader = csv.reader(beam_file)
        next(reader)
        header = next(reader)
        tip_y_index = header.index("tip_DISPLACEMENT_Y")
        for row in reader:
            time_value = float(row[0])
            tip_y = float(row[tip_y_index])
            metrics["last_time"] = time_value
            metrics["final_tip_y"] = tip_y
            metrics["max_abs_tip_y"] = max(metrics["max_abs_tip_y"], abs(tip_y))
            samples.append((time_value, tip_y))
            metrics["number_of_samples"] += 1
    if metrics["number_of_samples"]:
        tip_values = [value for _, value in samples]
        metrics["rms_tip_y"] = rms(tip_values)
        metrics["tip_peak_to_peak"] = peak_to_peak(tip_values)
        early_values = [
            value for time_value, value in samples
            if settings["startup_end_time"] - 1e-12 <= time_value <= settings["near_equilibrium_end_time"] + 1e-12
        ]
        end_values = [
            value for time_value, value in samples
            if time_value >= settings["end_time"] - settings["end_window_duration"] - 1e-12
        ]
        metrics["early_probe_rms_tip_y"] = rms(early_values)
        metrics["early_probe_peak_to_peak"] = peak_to_peak(early_values)
        metrics["early_probe_tip_min"] = min(early_values) if early_values else 0.0
        metrics["early_probe_tip_max"] = max(early_values) if early_values else 0.0
        metrics["end_window_rms_tip_y"] = rms(end_values)
        metrics["end_window_peak_to_peak"] = peak_to_peak(end_values)
        metrics["end_window_tip_min"] = min(end_values) if end_values else 0.0
        metrics["end_window_tip_max"] = max(end_values) if end_values else 0.0
        metrics["leaves_small_amplitude_neighborhood"] = (
            metrics["end_window_peak_to_peak"] > settings["small_amplitude_p2p_threshold"]
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


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values)) if values else 0.0


def peak_to_peak(values: list[float]) -> float:
    return max(values) - min(values) if values else 0.0


def collect_input_metrics(input_path: Path) -> dict:
    metrics = {
        "max_abs_u": 0.0,
        "max_abs_u_dot": 0.0,
        "max_abs_envelope_ac_dot": 0.0,
        "max_abs_envelope_as_dot": 0.0,
        "max_envelope_amplitude": 0.0,
        "min_u": float("inf"),
        "max_u": -float("inf"),
    }
    with input_path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            control = float(row["control_u"])
            metrics["max_abs_u"] = max(metrics["max_abs_u"], abs(control))
            metrics["max_abs_u_dot"] = max(metrics["max_abs_u_dot"], abs(float(row["control_u_dot"])))
            metrics["max_abs_envelope_ac_dot"] = max(metrics["max_abs_envelope_ac_dot"], abs(float(row["envelope_ac_dot"])))
            metrics["max_abs_envelope_as_dot"] = max(metrics["max_abs_envelope_as_dot"], abs(float(row["envelope_as_dot"])))
            metrics["max_envelope_amplitude"] = max(metrics["max_envelope_amplitude"], float(row["envelope_amplitude"]))
            metrics["min_u"] = min(metrics["min_u"], control)
            metrics["max_u"] = max(metrics["max_u"], control)
    return metrics


def write_case_metadata(
    case: dict,
    settings: dict,
    campaign_directory: Path,
    input_path: Path,
    run_directory: Path,
    log_path: Path,
    return_code: int,
) -> None:
    metadata = {
        "case": case,
        "campaign_settings": {
            "campaign_label": settings["campaign_label"],
            "control_convention": "upper control = u, lower control = -u",
            "physical_control": "u = envelope_ac*cos(theta) + envelope_as*sin(theta)",
            "carrier_frequency_hz": settings["carrier_frequency_hz"],
            "carrier_omega_rad_s": settings["carrier_omega_rad_s"],
            "envelope_bound": settings["envelope_bound"],
            "initialization_protocol": case["initialization_protocol"],
            "initial_state_source": "deterministic_standard_startup",
            "startup_end_time": settings["startup_end_time"],
            "near_equilibrium_probe_time": settings["near_equilibrium_probe_time"],
            "near_equilibrium_end_time": settings["near_equilibrium_end_time"],
            "controlled_start_time": case["controlled_start_time"],
            "use_for_identification_start_time": case["use_for_identification_start_time"],
            "initialization": (
                "standard Kratos initial condition without restart; early-growth cases repeat the "
                "unforced startup, while mature cases use the common smooth 0.4 kick on t=0..2 s"
            ),
            "scientific_scope": (
                "Samples near the unstable-equilibrium neighborhood are obtained during deterministic "
                "startup; this is not an exact unstable-equilibrium restart."
            ),
        },
        "input_path": str(input_path.resolve()),
        "signal_path": str(input_path.resolve()),
        "identification_snapshots_path": str((run_directory / "identification_snapshots.csv").resolve()),
        "return_code": return_code,
        "log_path": str(log_path.resolve()),
    }
    base.write_text_no_overwrite(run_directory / "case_metadata.json", json.dumps(metadata, indent=4))


def write_case_result(campaign_directory: Path, result: dict) -> None:
    result_path = campaign_directory / "case_results" / f"{result['label']}.json"
    base.write_text_no_overwrite(result_path, json.dumps(result, indent=4))


def write_summary_files(campaign_directory: Path, results: list[dict], manifest: dict) -> None:
    ordered = sort_results(results, manifest)
    augment_state_coverage_metrics(ordered)
    (campaign_directory / "summary.json").write_text(json.dumps(ordered, indent=4))
    with (campaign_directory / "summary.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=SUMMARY_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)
    print(f"wrote {campaign_directory / 'summary.csv'}")
    print(f"wrote {campaign_directory / 'summary.json'}")


def augment_state_coverage_metrics(results: list[dict]) -> None:
    if not results:
        return
    passive = next((result for result in results if result["label"] == "early_passive_reference"), None)
    passive_end_p2p = float(passive["end_window_peak_to_peak"]) if passive else 0.0
    early_results = [result for result in results if result["initialization_protocol"] == "early_growth"]
    mature_results = [result for result in results if result["initialization_protocol"] == "mature_limit_cycle"]
    early_range = pooled_range(early_results, "early_probe_tip_min", "early_probe_tip_max")
    mature_range = pooled_range(mature_results, "end_window_tip_min", "end_window_tip_max")
    global_overlap = interval_overlap_fraction(early_range, mature_range)
    for result in results:
        if result["initialization_protocol"] == "mature_limit_cycle" and passive_end_p2p > 1e-14:
            result["moves_trajectory_inward"] = (
                float(result["end_window_peak_to_peak"]) < 0.95 * passive_end_p2p
            )
        else:
            result["moves_trajectory_inward"] = "not_applicable"
        result["early_mature_state_range_overlap"] = global_overlap


def pooled_range(results: list[dict], minimum_key: str, maximum_key: str) -> tuple[float, float]:
    if not results:
        return (0.0, 0.0)
    return (
        min(float(result[minimum_key]) for result in results),
        max(float(result[maximum_key]) for result in results),
    )


def interval_overlap_fraction(first: tuple[float, float], second: tuple[float, float]) -> float:
    overlap = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return overlap / union if union > 1e-14 else 0.0


def audit_campaign(campaign_directory: Path, manifest: dict, results: list[dict]) -> None:
    errors = []
    expected_cases = manifest["cases"]
    if len(expected_cases) != 36 or int(manifest.get("total_number_of_cases", -1)) != 36:
        errors.append(f"expected exactly 36 cases, manifest has {len(expected_cases)}")
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
    for case in expected_cases:
        result = result_by_label.get(case["label"])
        if result is None:
            continue
        run_directory = Path(result["run_directory"])
        if not run_directory.exists():
            run_directory = campaign_directory / "runs" / case["label"]
        errors.extend(audit_case(campaign_directory, run_directory, case, result, manifest, expected_rows))

    if errors:
        print("Campaign audit failed:")
        for error in errors:
            print(f"  ERROR: {error}")
        raise RuntimeError("Fourier-envelope campaign audit failed.")
    print("Campaign audit passed:")
    print(f"  expected cases completed: {len(expected_labels)}")
    print("  schemas, timing, protocol masks, envelope identity, smoothness, metadata, and actuator balance are OK")


def audit_case(
    campaign_directory: Path,
    run_directory: Path,
    case: dict,
    result: dict,
    manifest: dict,
    expected_rows: int,
) -> list[str]:
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
    expected_input_rows = int(round(float(manifest["end_time"]) / float(manifest["input_dt"]))) + 1
    if len(input_rows) != expected_input_rows:
        errors.append(f"{case['label']}: input rows {len(input_rows)} != {expected_input_rows}")
    if len(identification_rows) != expected_rows:
        errors.append(f"{case['label']}: identification rows {len(identification_rows)} != {expected_rows}")
    if abs(float(identification_rows[0]["time"])) > 1e-12:
        errors.append(f"{case['label']}: first identification time is not 0")
    if abs(float(identification_rows[-1]["time"]) - float(manifest["end_time"])) > 5e-9:
        errors.append(f"{case['label']}: final identification time is not end_time")

    header = identification_rows[0].keys()
    missing_columns = [column for column in REQUIRED_IDENTIFICATION_COLUMNS if column not in header]
    if missing_columns:
        errors.append(f"{case['label']}: missing identification columns {missing_columns}")

    input_by_time = {round(row["time_float"], 12): row for row in input_rows}
    for row in identification_rows:
        time_value = round(float(row["time"]), 12)
        input_row = input_by_time.get(time_value)
        if input_row is None:
            errors.append(f"{case['label']}: missing input sample at t={time_value:g}")
            break
        for column in CONTROL_COLUMNS + ["control_kind"]:
            if column == "control_kind":
                if row[column] != input_row[column]:
                    errors.append(f"{case['label']}: {column} mismatch at t={time_value:g}")
                    return errors
            elif abs(float(row[column]) - float(input_row[column])) > 1e-10:
                errors.append(f"{case['label']}: {column} mismatch at t={time_value:g}")
                return errors

    expected_mask_start = float(case["use_for_identification_start_time"])
    for row in input_rows:
        expected_mask = int(float(row["time"]) + 1e-12 >= expected_mask_start)
        if int(float(row["use_for_identification"])) != expected_mask:
            errors.append(f"{case['label']}: incorrect identification mask at t={float(row['time']):g}")
            break

    errors.extend(audit_envelope_rows(case, input_rows, manifest))
    input_metrics = collect_input_metrics(run_directory / "input_timeseries.csv")
    if input_metrics["max_envelope_amplitude"] > float(manifest["envelope_bound"]) + 1e-10:
        errors.append(f"{case['label']}: envelope amplitude exceeds bound")
    for column, expected in input_metrics.items():
        if abs(float(result[column]) - expected) > 1e-10:
            errors.append(f"{case['label']}: summary {column} mismatch")
            break
    if float(result["max_abs_tip_y"]) > float(manifest["max_reasonable_tip_displacement"]):
        errors.append(f"{case['label']}: max_abs_tip_y exceeds sanity limit")

    errors.extend(pm.audit_actuator_balance(
        run_directory,
        {"label": case["label"], "csv_interpolation": "linear"},
        input_rows,
        float(manifest["input_dt"]),
    ))
    errors.extend(audit_case_metadata(run_directory, case, result))
    return errors


def audit_envelope_rows(case: dict, rows: list[dict], manifest: dict) -> list[str]:
    errors = []
    max_formula_error = 0.0
    max_ac_jump = 0.0
    max_as_jump = 0.0
    max_ac_dot_jump_rate = 0.0
    max_as_dot_jump_rate = 0.0
    previous = None
    for row in rows:
        theta = float(row["carrier_theta_unwrapped_rad"])
        ac = float(row["envelope_ac"])
        a_s = float(row["envelope_as"])
        control = float(row["control_u"])
        expected = ac * math.cos(theta) + a_s * math.sin(theta)
        max_formula_error = max(max_formula_error, abs(control - expected))
        if math.hypot(ac, a_s) > float(manifest["envelope_bound"]) + 1e-10:
            errors.append(f"{case['label']}: envelope radius exceeds bound")
            break
        if previous is not None:
            dt = float(row["time"]) - float(previous["time"])
            max_ac_jump = max(max_ac_jump, abs(ac - float(previous["envelope_ac"])) / max(dt, 1e-15))
            max_as_jump = max(max_as_jump, abs(a_s - float(previous["envelope_as"])) / max(dt, 1e-15))
            max_ac_dot_jump_rate = max(
                max_ac_dot_jump_rate,
                abs(float(row["envelope_ac_dot"]) - float(previous["envelope_ac_dot"])) / max(dt, 1e-15),
            )
            max_as_dot_jump_rate = max(
                max_as_dot_jump_rate,
                abs(float(row["envelope_as_dot"]) - float(previous["envelope_as_dot"])) / max(dt, 1e-15),
            )
        previous = row
    if max_formula_error > 1e-10:
        errors.append(f"{case['label']}: u != ac*cos(theta)+as*sin(theta), max error {max_formula_error:g}")

    max_ac_dot = max(abs(float(row["envelope_ac_dot"])) for row in rows)
    max_as_dot = max(abs(float(row["envelope_as_dot"])) for row in rows)
    if max_ac_jump > 1.25 * max_ac_dot + 1e-6:
        errors.append(f"{case['label']}: possible envelope_ac discontinuity")
    if max_as_jump > 1.25 * max_as_dot + 1e-6:
        errors.append(f"{case['label']}: possible envelope_as discontinuity")
    acceleration_limit = float(manifest["max_envelope_acceleration_for_audit"])
    if max_ac_dot_jump_rate > acceleration_limit + 1e-6:
        errors.append(
            f"{case['label']}: envelope_ac_dot jump implies acceleration "
            f"{max_ac_dot_jump_rate:g} > {acceleration_limit:g}"
        )
    if max_as_dot_jump_rate > acceleration_limit + 1e-6:
        errors.append(
            f"{case['label']}: envelope_as_dot jump implies acceleration "
            f"{max_as_dot_jump_rate:g} > {acceleration_limit:g}"
        )
    return errors


def audit_case_metadata(run_directory: Path, case: dict, result: dict) -> list[str]:
    metadata = json.loads((run_directory / "case_metadata.json").read_text())
    stored_case = metadata.get("case", {})
    errors = []
    if stored_case.get("label") != case["label"]:
        errors.append(f"{case['label']}: case_metadata label mismatch")
    if stored_case != case:
        errors.append(f"{case['label']}: stored case metadata differs from manifest case")
    if result.get("signal_type") != case["signal_type"] or result.get("role") != case["role"]:
        errors.append(f"{case['label']}: summary role/signal_type mismatch")
    if stored_case.get("initialization_protocol") != case["initialization_protocol"]:
        errors.append(f"{case['label']}: initialization protocol mismatch")
    if stored_case.get("initial_state_source") != "deterministic_standard_startup":
        errors.append(f"{case['label']}: invalid initial_state_source")
    for key in (
        "initialization_protocol",
        "controlled_start_time",
        "use_for_identification_start_time",
        "startup_end_time",
        "near_equilibrium_probe_time",
        "near_equilibrium_end_time",
    ):
        if result.get(key) != case.get(key):
            errors.append(f"{case['label']}: summary/manifest mismatch for {key}")
    serialized = json.dumps(metadata).lower()
    if "restart_directory" in serialized or "restart_path" in serialized:
        errors.append(f"{case['label']}: restart metadata is forbidden")
    return errors


def make_manifest(settings: dict, cases: list[dict]) -> dict:
    return {
        "campaign_label": settings["campaign_label"],
        "created_at": settings["created_at"],
        "purpose": "Fourier-envelope FSI2 identification across early growth and mature limit-cycle dynamics",
        "base_campaign": "fsi_control_fourier_envelope_t40_u20",
        "modeling_goal": "eta_dot = R(eta, theta, a_c, a_s)",
        "control_convention": "physical scalar u applied as upper=u, lower=-u",
        "physical_control": "u(t)=a_c(t)*cos(theta(t))+a_s(t)*sin(theta(t))",
        "carrier_frequency_hz": settings["carrier_frequency_hz"],
        "carrier_omega_rad_s": settings["carrier_omega_rad_s"],
        "detuning_sign_convention": (
            "For a_c=A*cos(psi), a_s=A*sin(psi), physical frequency is "
            "f_c - psi_dot/(2*pi)."
        ),
        "envelope_bound": settings["envelope_bound"],
        "envelope_bandwidth_hz": {
            "random_fourier_max": 0.15,
            "radial_ramp_approx": 1.0 / settings["radial_ramp_duration"],
        },
        "initial_state_source": "deterministic_standard_startup",
        "restart_available": False,
        "scientific_scope": (
            "Early cases sample a neighborhood of the unstable equilibrium during repeated deterministic "
            "startup. They are not exact unstable-equilibrium restarts."
        ),
        "state_coverage_report_definition": (
            "early_mature_state_range_overlap is the intersection-over-union of the pooled scalar tip-Y "
            "ranges in the early probe windows and mature end windows"
        ),
        "startup_window": {
            "inlet_ramp_end_time": 2.0,
            "startup_end_time": settings["startup_end_time"],
            "near_equilibrium_probe_time": settings["near_equilibrium_probe_time"],
            "near_equilibrium_end_time": settings["near_equilibrium_end_time"],
            "calibration_source": "deterministic passive FSI2 startup trajectory",
            "small_amplitude_p2p_threshold": settings["small_amplitude_p2p_threshold"],
        },
        "initialization": {
            "early_growth": {
                "type": "standard_startup_without_strong_kick",
                "controlled_start_time": settings["near_equilibrium_probe_time"],
                "use_for_identification_start_time": settings["startup_end_time"],
            },
            "mature_limit_cycle": {
                "type": "common_smooth_kick_then_passive_warmup",
                "controlled_start_time": settings["warmup_end_time"],
                "use_for_identification_start_time": settings["warmup_end_time"],
            },
            "initial_kick_amplitude": settings["initial_kick_amplitude"],
            "initial_kick_end_time": settings["initial_kick_end_time"],
            "initial_kick_ramp_time": settings["initial_kick_ramp_time"],
            "activation_ramp_duration": settings["activation_ramp_duration"],
            "note": "No restart paths are used; forcing phase is independent of activation time.",
        },
        "early_probe_amplitudes": EARLY_PROBE_AMPLITUDES,
        "mature_constant_amplitudes": MATURE_CONSTANT_AMPLITUDES,
        "constant_phases": ["0", "pi/2", "pi", "3pi/2"],
        "radial_ramp_specs": RADIAL_RAMP_SPECS,
        "rotating_amplitude": 1.5,
        "detunings_hz": DETUNINGS_HZ,
        "train_random_seeds": TRAIN_RANDOM_SEEDS,
        "validation_random_seeds": VALIDATION_RANDOM_SEEDS,
        "challenge_cases": CHALLENGE_CASES,
        "end_time": settings["end_time"],
        "output_dt": settings["output_dt"],
        "input_dt": settings["input_dt"],
        "write_paraview": settings["write_paraview"],
        "max_reasonable_tip_displacement": settings["max_reasonable_tip_displacement"],
        "max_envelope_acceleration_for_audit": settings["max_envelope_acceleration_for_audit"],
        "identification_snapshot_columns": REQUIRED_IDENTIFICATION_COLUMNS,
        "total_number_of_cases": len(cases),
        "cases": cases,
    }


def write_manifest_once(campaign_directory: Path, settings: dict, cases: list[dict]) -> None:
    campaign_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = campaign_directory / "manifest.json"
    manifest = make_manifest(settings, cases)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=campaign_directory,
            prefix=".manifest.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(manifest, output_file, indent=4)
            output_file.flush()
            os.fsync(output_file.fileno())
        try:
            os.link(temporary_path, manifest_path)
        except FileExistsError:
            pass

        existing = json.loads(manifest_path.read_text())
        existing_labels = [case["label"] for case in existing.get("cases", [])]
        new_labels = [case["label"] for case in cases]
        if existing_labels != new_labels:
            raise RuntimeError(f"{manifest_path} exists for a different campaign.")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def read_manifest(campaign_directory: Path) -> dict:
    manifest_path = campaign_directory / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def read_case_results(campaign_directory: Path) -> list[dict]:
    return pm.read_case_results(campaign_directory)


def select_cases(cases: list[dict]) -> list[tuple[int, dict]]:
    labels = os.environ.get("KRATOS_FSI_FE_MS_CASE_LABELS") or os.environ.get("KRATOS_FSI_FE_CASE_LABELS")
    index = os.environ.get("KRATOS_FSI_FE_MS_CASE_INDEX")
    limit = os.environ.get("KRATOS_FSI_FE_MS_LIMIT")
    if labels:
        wanted = {label.strip() for label in labels.split(",") if label.strip()}
        return [(i, case) for i, case in enumerate(cases) if case["label"] in wanted]
    if index is not None:
        index_value = int(index)
        if index_value < 0 or index_value >= len(cases):
            raise IndexError(f"KRATOS_FSI_FE_MS_CASE_INDEX={index_value} outside 0..{len(cases)-1}.")
        return [(index_value, cases[index_value])]
    selected = list(enumerate(cases))
    if limit is not None:
        selected = selected[:int(limit)]
    return selected


def sort_results(results: list[dict], manifest: dict) -> list[dict]:
    order = {case["label"]: index for index, case in enumerate(manifest["cases"])}
    return sorted(results, key=lambda result: order.get(result["label"], len(order)))


def print_report(results: list[dict], manifest: dict) -> None:
    ordered = sort_results(results, manifest)
    augment_state_coverage_metrics(ordered)
    print("Fourier-envelope multistate coverage report:")
    print(f"  number_of_cases={len(ordered)}")
    print(
        "  early samples are near the unstable-equilibrium neighborhood during deterministic startup; "
        "they are not exact unstable-equilibrium restarts"
    )
    for result in ordered:
        print(
            f"  {result['label']}: "
            f"protocol={result['initialization_protocol']}, type={result['signal_type']}, "
            f"tip_rms={float(result['rms_tip_y']):.6g}, "
            f"tip_p2p={float(result['tip_peak_to_peak']):.6g}, "
            f"early_p2p={float(result['early_probe_peak_to_peak']):.6g}, "
            f"end_p2p={float(result['end_window_peak_to_peak']):.6g}, "
            f"leaves_small_neighborhood={result['leaves_small_amplitude_neighborhood']}, "
            f"moves_inward={result['moves_trajectory_inward']}, "
            f"early/mature_overlap={float(result['early_mature_state_range_overlap']):.3f}"
        )


def format_result(result: dict) -> str:
    return (
        f"{result['label']}: "
        f"max|u|={float(result['max_abs_u']):.6g}, "
        f"max|a|={float(result['max_envelope_amplitude']):.6g}, "
        f"max|tip_y|={float(result['max_abs_tip_y']):.6g}"
    )


def envelope_bandwidth(case: dict, settings: dict) -> float:
    if case["signal_type"] == "random_fourier_envelope":
        return 3.0 / max(settings["end_time"] - settings["warmup_end_time"], settings["input_dt"])
    if case["signal_type"] == "rotating_quadrature":
        return abs(float(case["metadata"]["detuning_hz"]))
    if case["signal_type"] == "radial_ramp":
        return 1.0 / float(case["metadata"].get("ramp_duration", settings["radial_ramp_duration"]))
    if case["signal_type"] == "constant_quadrature":
        return 1.0 / settings["activation_ramp_duration"]
    if case["signal_type"] == "early_burst":
        return 1.0 / float(case["metadata"]["burst_ramp_duration"])
    return 0.0


def analyze_passive_startup(beam_path: Path, settings: dict) -> dict:
    samples = read_tip_samples(beam_path)
    if not samples:
        raise RuntimeError(f"No beam samples found in {beam_path}.")
    last_time = samples[-1][0]
    if last_time + 1e-9 < settings["near_equilibrium_end_time"]:
        raise RuntimeError(
            f"Passive pilot ends at t={last_time:g}, before near_equilibrium_end_time="
            f"{settings['near_equilibrium_end_time']:g}."
        )
    early_values = values_in_window(
        samples,
        settings["startup_end_time"],
        settings["near_equilibrium_end_time"],
    )
    probe_half_width = 0.25
    probe_values = values_in_window(
        samples,
        settings["near_equilibrium_probe_time"] - probe_half_width,
        settings["near_equilibrium_probe_time"] + probe_half_width,
    )
    mature_values = values_in_window(
        samples,
        max(settings["near_equilibrium_end_time"], last_time - settings["end_window_duration"]),
        last_time,
    )
    early_p2p = peak_to_peak(early_values)
    mature_p2p = peak_to_peak(mature_values)
    usable = (
        settings["startup_end_time"] >= 2.0
        and settings["startup_end_time"] < settings["near_equilibrium_probe_time"] < settings["near_equilibrium_end_time"]
        and early_p2p <= settings["small_amplitude_p2p_threshold"]
        and (mature_p2p <= 1e-14 or early_p2p < 0.25 * mature_p2p)
    )
    return {
        "beam_path": str(beam_path.resolve()),
        "mandatory_inlet_ramp_end_time": 2.0,
        "startup_end_time": settings["startup_end_time"],
        "near_equilibrium_probe_time": settings["near_equilibrium_probe_time"],
        "near_equilibrium_end_time": settings["near_equilibrium_end_time"],
        "last_time": last_time,
        "early_window_rms_tip_y": rms(early_values),
        "early_window_peak_to_peak_tip_y": early_p2p,
        "probe_window_peak_to_peak_tip_y": peak_to_peak(probe_values),
        "mature_window_peak_to_peak_tip_y": mature_p2p,
        "early_to_mature_peak_to_peak_ratio": early_p2p / mature_p2p if mature_p2p > 1e-14 else None,
        "usable_near_equilibrium_interval": usable,
        "interpretation": (
            "samples near the unstable-equilibrium neighborhood; not an exact unstable-equilibrium restart"
        ),
    }


def read_tip_samples(beam_path: Path) -> list[tuple[float, float]]:
    with beam_path.open(newline="") as beam_file:
        reader = csv.reader(beam_file)
        next(reader)
        header = next(reader)
        tip_index = header.index("tip_DISPLACEMENT_Y")
        return [(float(row[0]), float(row[tip_index])) for row in reader]


def values_in_window(samples: list[tuple[float, float]], start: float, end: float) -> list[float]:
    return [value for time_value, value in samples if start - 1e-12 <= time_value <= end + 1e-12]


def print_startup_report(report: dict) -> None:
    print(json.dumps(report, indent=2))
    if not report["usable_near_equilibrium_interval"]:
        raise RuntimeError(
            "No usable post-startup near-equilibrium interval was verified. Do not submit the full campaign."
        )


def format_positive(value: float) -> str:
    return f"{abs(value):.3f}".replace(".", "p")


def wrap_phase(value: float) -> float:
    wrapped = value % (2.0 * math.pi)
    if wrapped < 1e-12 or 2.0 * math.pi - wrapped < 1e-12:
        return 0.0
    return wrapped


def format_signed(value: float, digits: int = 3) -> str:
    prefix = "m" if value < -1e-12 else "p"
    return prefix + f"{abs(value):.{digits}f}".replace(".", "p")


def format_phase_label(phi: float) -> str:
    normalized = phi % (2.0 * math.pi)
    if abs(normalized) < 1e-12:
        return "0"
    if abs(normalized - 0.5 * math.pi) < 1e-12:
        return "pi2"
    if abs(normalized - math.pi) < 1e-12:
        return "pi"
    if abs(normalized - 1.5 * math.pi) < 1e-12:
        return "3pi2"
    if abs(normalized - 0.25 * math.pi) < 1e-12:
        return "pi4"
    return format_positive(normalized)


def format_control_value(value) -> str:
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if abs(value) < 5e-15:
        value = 0.0
    return f"{value:.17g}"


def read_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def read_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "no", "off")


if __name__ == "__main__":
    main()
