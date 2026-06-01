import csv
import os
import subprocess
import sys
from pathlib import Path


PYTHONPATH = "/Users/jaking/Kratos/bin/Release"
DYLD_LIBRARY_PATH = "/Users/jaking/Kratos/bin/Release/libs"


CASES = [
    {"label": "passive_probe", "controller": "beam_tip_feedback", "sign": -1.0, "kp": 0.0, "kd": 0.0, "qmax": 0.0, "delay": 0.0},
    {"label": "phase_f38_g200_phi_m90", "controller": "beam_tip_phase", "sign": -1.0, "qmax": 0.08, "frequency": 3.8, "phase": -1.57079632679, "gain": 200.0},
    {"label": "phase_f38_g300_phi_m90", "controller": "beam_tip_phase", "sign": -1.0, "qmax": 0.08, "frequency": 3.8, "phase": -1.57079632679, "gain": 300.0},
    {"label": "phase_f38_g500_phi_m90", "controller": "beam_tip_phase", "sign": -1.0, "qmax": 0.08, "frequency": 3.8, "phase": -1.57079632679, "gain": 500.0},
    {"label": "phase_f38_g300_phi_m120", "controller": "beam_tip_phase", "sign": -1.0, "qmax": 0.08, "frequency": 3.8, "phase": -2.09439510239, "gain": 300.0},
    {"label": "phase_f38_g300_phi_m60", "controller": "beam_tip_phase", "sign": -1.0, "qmax": 0.08, "frequency": 3.8, "phase": -1.0471975512, "gain": 300.0},
]


def main():
    end_time = float(os.environ.get("KRATOS_FSI_SWEEP_END_TIME", "0.25"))
    output_interval = float(os.environ.get("KRATOS_FSI_SWEEP_OUTPUT_INTERVAL", "0.01"))
    results = []

    for case in CASES:
        run_directory = run_case(case, end_time, output_interval)
        metrics = collect_metrics(run_directory)
        results.append({
            **case,
            "end_time": end_time,
            "run_directory": str(run_directory),
            **metrics,
        })
        print(format_result(results[-1]), flush=True)

    output_path = Path("run_outputs") / "feedback_gain_sweep_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_summary(output_path, results)
    print(f"summary={output_path.resolve()}")


def run_case(case, end_time, output_interval):
    before = set(Path("run_outputs").glob("run_*"))
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": PYTHONPATH,
        "DYLD_LIBRARY_PATH": DYLD_LIBRARY_PATH,
        "KRATOS_FSI_RUN_LABEL": case["label"],
        "KRATOS_FSI_END_TIME": str(end_time),
        "KRATOS_FSI_OUTPUT_INTERVAL": str(output_interval),
        "KRATOS_FSI_WRITE_PARAVIEW": "0",
        "KRATOS_FSI_CONTROLLER_TYPE": case["controller"],
        "KRATOS_FSI_CONTROL_SIGN": str(case["sign"]),
        "KRATOS_FSI_KP": str(case.get("kp", 0.0)),
        "KRATOS_FSI_KD": str(case.get("kd", 0.0)),
        "KRATOS_FSI_QMAX": str(case["qmax"]),
        "KRATOS_FSI_FEEDBACK_DELAY": str(case.get("delay", 0.0)),
        "KRATOS_FSI_OSCILLATOR_FREQUENCY": str(case.get("frequency", 3.8)),
        "KRATOS_FSI_OSCILLATOR_PHASE_SHIFT": str(case.get("phase", 0.0)),
        "KRATOS_FSI_OSCILLATOR_GAIN": str(case.get("gain", 0.0)),
    })

    subprocess.run([sys.executable, "MainKratos.py"], check=True, env=environment)

    after = set(Path("run_outputs").glob("run_*"))
    created = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if not created:
        raise RuntimeError(f"No run directory was created for case {case['label']}.")
    return created[-1]


def collect_metrics(run_directory):
    beam_path = run_directory / "beam_displacement_timeseries.csv"
    actuator_path = run_directory / "actuator_timeseries.csv"

    max_abs_tip_y = 0.0
    final_tip_y = 0.0
    last_beam_time = 0.0
    with beam_path.open(newline="") as beam_file:
        reader = csv.reader(beam_file)
        next(reader)
        header = next(reader)
        tip_y_index = header.index("tip_DISPLACEMENT_Y")
        for row in reader:
            last_beam_time = float(row[0])
            final_tip_y = float(row[tip_y_index])
            max_abs_tip_y = max(max_abs_tip_y, abs(final_tip_y))

    max_abs_q = 0.0
    saturated_rows = 0
    total_rows = 0
    last_actuator_time = 0.0
    with actuator_path.open(newline="") as actuator_file:
        reader = csv.DictReader(actuator_file)
        for row in reader:
            q = abs(float(row["control_value"]))
            max_abs_q = max(max_abs_q, q)
            last_actuator_time = float(row["time"])
            total_rows += 1

    qmax = read_qmax(run_directory)
    if qmax > 0.0:
        with actuator_path.open(newline="") as actuator_file:
            reader = csv.DictReader(actuator_file)
            saturated_rows = sum(
                1
                for row in reader
                if abs(float(row["control_value"])) >= 0.999 * qmax
            )

    return {
        "last_beam_time": last_beam_time,
        "last_actuator_time": last_actuator_time,
        "max_abs_tip_y": max_abs_tip_y,
        "final_tip_y": final_tip_y,
        "max_abs_q": max_abs_q,
        "saturation_fraction": saturated_rows / total_rows if total_rows else 0.0,
    }


def read_qmax(run_directory):
    parameters_path = run_directory / "ProjectParameters.effective.json"
    text = parameters_path.read_text()
    marker = '"max_abs_control":'
    if marker not in text:
        return 0.0
    return float(text.split(marker, 1)[1].split(",", 1)[0].strip())


def write_summary(output_path, results):
    fieldnames = [
        "label",
        "controller",
        "sign",
        "kp",
        "kd",
        "qmax",
        "delay",
        "frequency",
        "phase",
        "gain",
        "end_time",
        "last_beam_time",
        "last_actuator_time",
        "max_abs_tip_y",
        "final_tip_y",
        "max_abs_q",
        "saturation_fraction",
        "run_directory",
    ]
    with output_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def format_result(result):
    return (
        f"{result['label']}: "
        f"max_abs_tip_y={result['max_abs_tip_y']:.6g}, "
        f"max_abs_q={result['max_abs_q']:.6g}, "
        f"saturation={result['saturation_fraction']:.3f}, "
        f"run={result['run_directory']}"
    )


if __name__ == "__main__":
    main()
