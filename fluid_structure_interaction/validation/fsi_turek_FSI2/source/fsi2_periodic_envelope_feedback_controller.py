"""Fast phase-periodic envelope feedback for the FSI2 actuator pair."""

import csv
import json
import math
import time
from collections import deque
from pathlib import Path

try:
    import KratosMultiphysics
except ImportError:  # Artifact validation does not require Kratos.
    KratosMultiphysics = None


FORMAT_VERSION = 1
CONTROLLER_TYPE = "periodic_envelope_feedback"


def _vector(values, length=None):
    result = [float(value) for value in values]
    if length is not None and len(result) != length:
        raise ValueError(f"Expected vector length {length}, received {len(result)}.")
    return result


def _matrix(values, rows=None, columns=None):
    result = [_vector(row) for row in values]
    if rows is not None and len(result) != rows:
        raise ValueError(f"Expected {rows} rows, received {len(result)}.")
    if columns is not None and any(len(row) != columns for row in result):
        raise ValueError(f"Expected matrix width {columns}.")
    return result


def _columns(matrix):
    return [[float(matrix[i][j]) for i in range(len(matrix))]
            for j in range(len(matrix[0]))]


def _norm(values):
    return math.sqrt(sum(value * value for value in values))


def _project_ball(values, radius):
    magnitude = _norm(values)
    if magnitude <= radius or magnitude == 0.0:
        return values[:]
    return [radius * value / magnitude for value in values]


def validate_artifact(data):
    if int(data.get("format_version", 0)) != FORMAT_VERSION:
        raise ValueError(f"Expected format_version {FORMAT_VERSION}.")
    if data.get("controller_type") != CONTROLLER_TYPE:
        raise ValueError(f"Artifact is not a {CONTROLLER_TYPE} controller.")
    if data.get("parameter_coordinates") != ["envelope_ac", "envelope_as"]:
        raise ValueError("Unexpected envelope coordinates.")

    names = list(data["observable_names"])
    scales = _vector(data["observable_scale"], len(names))
    if any(value <= 0.0 for value in scales):
        raise ValueError("Observable scales must be positive.")
    delay_count = int(data["delay_count"])
    shift_steps = int(data["shift_steps"])
    if delay_count < 2 or shift_steps < 1:
        raise ValueError("Invalid delay embedding settings.")
    delay_basis = _matrix(data["delay_basis"], delay_count * len(names), 2)
    del delay_basis

    phase = _vector(data["phase_nodes"])
    gains = data["gain_table"]
    if len(phase) < 4 or len(gains) != len(phase):
        raise ValueError("Phase nodes and gain table are incompatible.")
    if any(next_phase <= current for current, next_phase in zip(phase, phase[1:])):
        raise ValueError("Phase nodes must be strictly increasing.")
    for gain in gains:
        _matrix(gain, 2, 4)
    _vector(data["eta_equilibrium"], 2)
    state_scale = _vector(data["state_scale"], 4)
    if any(value <= 0.0 for value in state_scale):
        raise ValueError("State scales must be positive.")
    for name in ["carrier_frequency_hz", "sample_interval", "control_interval",
                 "envelope_bound", "envelope_rate_bound"]:
        if float(data[name]) <= 0.0:
            raise ValueError(f"{name} must be positive.")
    if float(data["sampled_floquet_radius"]) >= 1.0:
        raise ValueError("Exported linear feedback is not Floquet stable.")


class PeriodicEnvelopeFeedbackLaw:
    """Interpolated periodic LQR with circular envelope and rate limits."""

    def __init__(self, data):
        validate_artifact(data)
        self.eta_equilibrium = _vector(data["eta_equilibrium"])
        self.state_scale = _vector(data["state_scale"])
        self.phase_nodes = _vector(data["phase_nodes"])
        self.gains = [_matrix(gain) for gain in data["gain_table"]]
        self.rate_bound = float(data["envelope_rate_bound"])
        self.envelope_bound = float(data["envelope_bound"])
        self.control_interval = float(data["control_interval"])

    def control(self, eta, theta, envelope):
        gain = self._interpolated_gain(theta)
        state = [
            (eta[i] - self.eta_equilibrium[i]) / self.state_scale[i]
            for i in range(2)
        ] + [envelope[i] / self.state_scale[2 + i] for i in range(2)]
        rate = [-sum(gain[i][j] * state[j] for j in range(4))
                * self.rate_bound for i in range(2)]
        rate = _project_ball(rate, self.rate_bound)
        endpoint = [envelope[i] + self.control_interval * rate[i]
                    for i in range(2)]
        endpoint = _project_ball(endpoint, self.envelope_bound)
        return [(endpoint[i] - envelope[i]) / self.control_interval
                for i in range(2)]

    def normalized_radius(self, eta):
        return _norm([
            (eta[i] - self.eta_equilibrium[i]) / self.state_scale[i]
            for i in range(2)
        ])

    def _interpolated_gain(self, theta):
        position = (theta % (2.0 * math.pi)) / (2.0 * math.pi) * len(self.gains)
        lower = int(math.floor(position))
        upper = (lower + 1) % len(self.gains)
        fraction = position - lower
        return [[(1.0 - fraction) * self.gains[lower][i][j]
                 + fraction * self.gains[upper][i][j]
                 for j in range(4)] for i in range(2)]


class PeriodicEnvelopeFeedbackController:
    """Kratos bridge from causal probe delays to periodic envelope feedback."""

    SAMPLE_POINTS = {
        "x_0_30": (0.30, 0.20),
        "x_0_40": (0.40, 0.20),
        "x_0_50": (0.50, 0.20),
        "tip": (0.60, 0.20),
    }

    def __init__(self, model, settings):
        if KratosMultiphysics is None:
            raise RuntimeError("This controller must run inside Kratos.")
        artifact_path = Path(settings["rom_file_name"].GetString())
        data = json.loads(artifact_path.read_text())
        self.law = PeriodicEnvelopeFeedbackLaw(data)
        self.model = model
        self.observable_names = list(data["observable_names"])
        self.observable_scale = _vector(data["observable_scale"])
        self.delay_basis = _matrix(data["delay_basis"])
        self.sample_interval = float(data["sample_interval"])
        self.shift_steps = int(data["shift_steps"])
        self.delay_count = int(data["delay_count"])
        history_length = (self.delay_count - 1) * self.shift_steps + 1
        self.history = deque(maxlen=history_length)
        self.omega = 2.0 * math.pi * float(data["carrier_frequency_hz"])
        self.phase_offset = float(data.get("carrier_phase", 0.0))
        self.activation_time = settings["mpc_activation_time"].GetDouble()
        self.kick_value = settings["mpc_initial_kick_value"].GetDouble()
        self.kick_end_time = settings["mpc_initial_kick_end_time"].GetDouble()
        self.envelope = [0.0, 0.0]
        self.envelope_rate = [0.0, 0.0]
        self.envelope_time = self.activation_time
        self.current_control = 0.0
        self.next_sample_time = 0.0
        self.next_control_time = self.activation_time
        self.last_compute_time = None
        self.nodes = self._find_measurement_nodes()

        output_path = Path(settings["rom_log_file_name"].GetString())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_file = output_path.open("w", newline="")
        self.writer = csv.writer(self.output_file)
        self.writer.writerow([
            "time", "eta_1", "eta_2", "carrier_theta", "envelope_ac",
            "envelope_as", "envelope_ac_dot", "envelope_as_dot", "control_u",
            "normalized_radius", "envelope_norm", "envelope_rate_norm",
        ])
        self._append_observation()
        self.next_sample_time = self.sample_interval

    def ComputeControl(self, current_time):
        if self.last_compute_time is not None and abs(
                current_time - self.last_compute_time) < 1e-12:
            return self.current_control
        self.last_compute_time = current_time

        while current_time + 1e-10 >= self.next_sample_time:
            self._append_observation()
            self.next_sample_time += self.sample_interval

        if current_time + 1e-10 < self.activation_time:
            self.current_control = (self.kick_value
                                    if current_time < self.kick_end_time - 1e-10
                                    else 0.0)
            return self.current_control

        self._advance_envelope(current_time)
        theta = self.omega * current_time + self.phase_offset
        if len(self.history) < self.history.maxlen:
            self.current_control = self._physical_control(theta)
            return self.current_control

        if current_time + 1e-10 >= self.next_control_time:
            eta = self._reduced_state()
            self.envelope_rate = self.law.control(eta, theta, self.envelope)
            while self.next_control_time <= current_time + 1e-10:
                self.next_control_time += self.law.control_interval
            self.writer.writerow([
                f"{current_time:.12g}", *[f"{value:.12g}" for value in eta],
                f"{theta:.12g}", *[f"{value:.12g}" for value in self.envelope],
                *[f"{value:.12g}" for value in self.envelope_rate],
                f"{self._physical_control(theta):.12g}",
                f"{self.law.normalized_radius(eta):.12g}",
                f"{_norm(self.envelope):.12g}", f"{_norm(self.envelope_rate):.12g}",
            ])
            self.output_file.flush()

        self.current_control = self._physical_control(theta)
        return self.current_control

    def Finalize(self):
        if self.output_file is not None:
            self.output_file.close()
            self.output_file = None

    def _advance_envelope(self, current_time):
        if current_time <= self.envelope_time:
            return
        dt = current_time - self.envelope_time
        self.envelope = [self.envelope[i] + dt * self.envelope_rate[i]
                         for i in range(2)]
        self.envelope = _project_ball(self.envelope, self.law.envelope_bound)
        self.envelope_time = current_time

    def _physical_control(self, theta):
        return self.envelope[0] * math.cos(theta) + self.envelope[1] * math.sin(theta)

    def _append_observation(self):
        observation = []
        for name in self.observable_names:
            stem = name.removeprefix("measurement_")
            axis = 0
            if name.endswith("_DISPLACEMENT_Y"):
                stem = stem.removesuffix("_DISPLACEMENT_Y")
                axis = 1
            else:
                stem = stem.removesuffix("_DISPLACEMENT_X")
            displacement = self.nodes[stem].GetSolutionStepValue(
                KratosMultiphysics.DISPLACEMENT)
            observation.append(float(displacement[axis]))
        self.history.append([
            observation[i] / self.observable_scale[i]
            for i in range(len(observation))
        ])

    def _reduced_state(self):
        delayed = []
        for index in range(0, len(self.history), self.shift_steps):
            delayed.extend(self.history[index])
        if len(delayed) != len(self.delay_basis):
            raise RuntimeError("Online delay vector does not match the POD basis.")
        return [sum(self.delay_basis[i][j] * delayed[i]
                    for i in range(len(delayed)))
                for j in range(2)]

    def _find_measurement_nodes(self):
        structure = self.model["Structure"]
        expected = set()
        for name in self.observable_names:
            stem = name.removeprefix("measurement_")
            stem = stem.removesuffix("_DISPLACEMENT_X").removesuffix("_DISPLACEMENT_Y")
            expected.add(stem)
        if not expected.issubset(self.SAMPLE_POINTS):
            raise RuntimeError(f"Unknown ROM sample points: {sorted(expected)}")
        return {
            name: min(structure.Nodes, key=lambda node: (
                node.X0 - self.SAMPLE_POINTS[name][0]) ** 2
                + (node.Y0 - self.SAMPLE_POINTS[name][1]) ** 2)
            for name in expected
        }


def validate_export(data):
    law = PeriodicEnvelopeFeedbackLaw(data)
    reference = data["validation"]
    maximum_rate_error = 0.0
    for eta, theta, envelope, expected in zip(
            _columns(reference["eta"]), reference["theta"],
            _columns(reference["envelope"]), _columns(reference["rate"])):
        rate = law.control(eta, float(theta), envelope)
        maximum_rate_error = max(maximum_rate_error, max(
            abs(rate[i] - expected[i]) for i in range(2)))

    basis = _matrix(data["delay_basis"])
    maximum_projection_error = 0.0
    for delayed, expected in zip(
            _columns(reference["delayed"]),
            _columns(reference["reconstructed_eta"])):
        eta = [sum(basis[i][j] * delayed[i] for i in range(len(delayed)))
               for j in range(2)]
        maximum_projection_error = max(maximum_projection_error, max(
            abs(eta[i] - expected[i]) for i in range(2)))
    return maximum_rate_error, maximum_projection_error


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate periodic FSI2 feedback.")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--benchmark-evaluations", type=int, default=10000)
    arguments = parser.parse_args()
    data = json.loads(arguments.artifact.read_text())
    errors = validate_export(data)
    law = PeriodicEnvelopeFeedbackLaw(data)
    start = time.perf_counter()
    for i in range(arguments.benchmark_evaluations):
        law.control([0.3, -0.2], 0.01 * i, [0.4, -0.1])
    elapsed = time.perf_counter() - start
    print("max export errors: rate={:.3e}, projection={:.3e}".format(*errors))
    print("ideal/sample-held Floquet radii: {:.6f}/{:.6f}".format(
        float(data["ideal_floquet_radius"]),
        float(data["sampled_floquet_radius"])))
    print("feedback time: {:.3f} us/evaluation".format(
        1e6 * elapsed / arguments.benchmark_evaluations))
