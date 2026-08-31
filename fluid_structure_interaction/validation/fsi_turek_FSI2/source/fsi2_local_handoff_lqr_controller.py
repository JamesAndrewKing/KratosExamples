"""Dependency-free guarded local stroboscopic LQR for the FSI2 actuator pair."""

import csv
import json
import math
from collections import deque
from pathlib import Path

try:
    import KratosMultiphysics
except ImportError:  # Artifact and control-law tests do not require Kratos.
    KratosMultiphysics = None


FORMAT_VERSION = 1
CONTROLLER_TYPE = "local_handoff_lqr"


def _vector(value, length=None):
    result = [float(item) for item in value]
    if length is not None and len(result) != length:
        raise ValueError(f"Expected vector length {length}, received {len(result)}.")
    return result


def _matrix(value, rows=None, columns=None):
    result = [_vector(row) for row in value]
    if rows is not None and len(result) != rows:
        raise ValueError(f"Expected {rows} matrix rows, received {len(result)}.")
    if columns is not None and any(len(row) != columns for row in result):
        raise ValueError(f"Expected matrix width {columns}.")
    return result


def _matvec(matrix, vector):
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]


def _norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def _clip(value, lower, upper):
    return max(lower, min(upper, value))


def _clip_norm(vector, limit):
    magnitude = _norm(vector)
    if magnitude <= limit or magnitude == 0.0:
        return vector[:]
    return [limit * value / magnitude for value in vector]


def validate_artifact_schema(data):
    if int(data.get("format_version", 0)) < FORMAT_VERSION:
        raise ValueError("Local handoff LQR requires format_version >= 1.")
    if data.get("controller_type") != CONTROLLER_TYPE:
        raise ValueError("Artifact is not a local_handoff_lqr controller.")
    if data.get("parameter_coordinates") != [
            "carrier_relative_ac", "carrier_relative_as"]:
        raise ValueError("Unexpected local envelope coordinates.")
    if not data.get("validated", False):
        raise ValueError("Local controller artifact is not marked validated.")

    observable_names = list(data["observable_names"])
    observable_scale = _vector(data["observable_scale"], len(observable_names))
    if any(value <= 0.0 for value in observable_scale):
        raise ValueError("Observable scales must be positive.")
    delay_count = int(data["delay_count"])
    shift_steps = int(data["shift_steps"])
    if delay_count < 2 or shift_steps < 1 or float(data["sample_interval"]) <= 0.0:
        raise ValueError("Invalid delay sampling settings.")
    delay_size = delay_count * len(observable_names)
    _matrix(data["delay_basis"], delay_size, 4)
    _vector(data["delay_reference"], delay_size)
    _matrix(data["F"], 4, 4)
    _matrix(data["B"], 4, 2)
    _vector(data["c"], 4)
    _vector(data["eta_star"], 4)
    _matrix(data["K"], 2, 6)
    state_scale = _vector(data["state_scale"], 6)
    input_scale = _vector(data["input_scale"], 2)
    if any(value <= 0.0 for value in state_scale + input_scale):
        raise ValueError("Controller scales must be positive.")
    if float(data["period"]) <= 0.0 or float(data["envelope_limit"]) <= 0.0:
        raise ValueError("Period and envelope limit must be positive.")
    if float(data["rate_limit"]) <= 0.0 or float(data["guard_radius"]) <= 0.0:
        raise ValueError("Rate and guard limits must be positive.")
    if data.get("interpolation") != "quintic_smoothstep_between_stroboscopic_updates":
        raise ValueError("Local controller requires quintic smoothstep interpolation.")
    if abs(float(data.get("interpolation_peak_rate_factor", 0.0)) - 1.875) > 1e-12:
        raise ValueError("Unexpected interpolation peak-rate factor.")
    expected_omega = 2.0 * math.pi / float(data["period"])
    if abs(float(data["carrier_omega_rad_s"]) - expected_omega) > 1e-11:
        raise ValueError("Carrier frequency and period are inconsistent.")


class LocalHandoffLaw:
    """Exported normalized LQR, constraints, local chart, and map."""

    def __init__(self, data, feedback_gain_multiplier=1.0):
        validate_artifact_schema(data)
        self.feedback_gain_multiplier = float(feedback_gain_multiplier)
        if not math.isfinite(self.feedback_gain_multiplier) \
                or self.feedback_gain_multiplier <= 0.0:
            raise ValueError("Local feedback gain multiplier must be positive and finite.")
        self.observable_names = list(data["observable_names"])
        self.observable_scale = _vector(data["observable_scale"])
        self.sample_interval = float(data["sample_interval"])
        self.shift_steps = int(data["shift_steps"])
        self.delay_count = int(data["delay_count"])
        self.delay_basis = _matrix(data["delay_basis"])
        self.delay_reference = _vector(data["delay_reference"])
        self.F = _matrix(data["F"])
        self.B = _matrix(data["B"])
        self.c = _vector(data["c"])
        self.eta_star = _vector(data["eta_star"])
        self.K = _matrix(data["K"])
        self.period = float(data["period"])
        self.carrier_omega = float(data["carrier_omega_rad_s"])
        self.carrier_phase = float(data.get("carrier_phase_rad", 0.0))
        self.state_scale = _vector(data["state_scale"])
        self.input_scale = _vector(data["input_scale"])
        self.envelope_limit = float(data["envelope_limit"])
        self.rate_limit = float(data["rate_limit"])
        self.guard_radius = float(data["guard_radius"])
        self.interpolation_peak_rate_factor = float(
            data["interpolation_peak_rate_factor"])

    def reconstruct_eta(self, delayed):
        centered = [delayed[i] - self.delay_reference[i]
                    for i in range(len(delayed))]
        return [sum(self.delay_basis[i][j] * centered[i]
                    for i in range(len(centered)))
                for j in range(4)]

    def local_radius(self, eta):
        return _norm([(eta[i] - self.eta_star[i]) / self.state_scale[i]
                      for i in range(4)])

    def control(self, eta, envelope):
        normalized_state = [
            (eta[i] - self.eta_star[i]) / self.state_scale[i]
            for i in range(4)
        ] + [envelope[i] / self.state_scale[4 + i] for i in range(2)]
        normalized_rate = [
            _clip(-self.feedback_gain_multiplier * value, -1.0, 1.0)
            for value in _matvec(self.K, normalized_state)
        ]
        rate = [self.input_scale[i] * normalized_rate[i] for i in range(2)]
        average_rate_limit = self.rate_limit / self.interpolation_peak_rate_factor
        rate = _clip_norm(rate, average_rate_limit)
        next_envelope = [envelope[i] + self.period * rate[i] for i in range(2)]
        if _norm(next_envelope) > self.envelope_limit:
            next_envelope = _clip_norm(next_envelope, self.envelope_limit)
            rate = [(next_envelope[i] - envelope[i]) / self.period for i in range(2)]
            if _norm(rate) > average_rate_limit:
                rate = _clip_norm(rate, average_rate_limit)
                next_envelope = [
                    envelope[i] + self.period * rate[i] for i in range(2)
                ]
        average = [0.5 * (envelope[i] + next_envelope[i]) for i in range(2)]
        autonomous = _matvec(self.F, eta)
        controlled = _matvec(self.B, average)
        predicted = [autonomous[i] + controlled[i] + self.c[i] for i in range(4)]
        return rate, next_envelope, predicted


def validate_export(data):
    law = LocalHandoffLaw(data)
    reference = data["validation"]
    maximum_errors = {
        "rate": 0.0,
        "next_envelope": 0.0,
        "next_eta": 0.0,
        "reconstructed_eta_offset": 0.0,
    }
    for eta, envelope, expected_rate, expected_envelope, expected_eta in zip(
            reference["eta"], reference["envelope"], reference["rate"],
            reference["next_envelope"], reference["next_eta"]):
        rate, next_envelope, next_eta = law.control(eta, envelope)
        maximum_errors["rate"] = max(maximum_errors["rate"], max(
            abs(rate[i] - expected_rate[i]) for i in range(2)))
        maximum_errors["next_envelope"] = max(
            maximum_errors["next_envelope"], max(
                abs(next_envelope[i] - expected_envelope[i]) for i in range(2)))
        maximum_errors["next_eta"] = max(maximum_errors["next_eta"], max(
            abs(next_eta[i] - expected_eta[i]) for i in range(4)))
    for delayed, expected in zip(
            reference["delayed"], reference["reconstructed_eta_offset"]):
        eta = law.reconstruct_eta(delayed)
        maximum_errors["reconstructed_eta_offset"] = max(
            maximum_errors["reconstructed_eta_offset"], max(
                abs(eta[i] - expected[i]) for i in range(4)))
    return maximum_errors


class LocalHandoffLqrController:
    """Kratos bridge from probe history to guarded stroboscopic actuation."""

    SAMPLE_POINTS = {
        "x_0_30": (0.30, 0.20),
        "x_0_40": (0.40, 0.20),
        "x_0_50": (0.50, 0.20),
        "tip": (0.60, 0.20),
    }

    def __init__(self, model, settings):
        if KratosMultiphysics is None:
            raise RuntimeError("LocalHandoffLqrController must run inside Kratos.")
        artifact_path = Path(settings["local_controller_file_name"].GetString())
        data = json.loads(artifact_path.read_text())
        gain_multiplier = settings[
            "local_controller_gain_multiplier"].GetDouble()
        self.law = LocalHandoffLaw(data, gain_multiplier)
        self.model = model
        self.activation_time = settings["local_controller_activation_time"].GetDouble()
        self.nodes = self._find_measurement_nodes()
        history_length = (self.law.delay_count - 1) * self.law.shift_steps + 1
        self.history = deque(maxlen=history_length)
        self.next_sample_time = 0.0
        self.last_compute_time = None
        self.current_control = 0.0
        self.started = False
        self.engaged = False
        self.guard_tripped = False
        self.phase_reference = 0.0
        self.next_update_time = self.activation_time
        self.segment_start_time = self.activation_time
        self.segment_start_envelope = [0.0, 0.0]
        self.segment_target_envelope = [0.0, 0.0]
        self.envelope = [0.0, 0.0]
        self.envelope_rate = [0.0, 0.0]

        output_path = Path(settings["local_controller_log_file_name"].GetString())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_file = output_path.open("w", newline="")
        self.writer = csv.writer(self.output_file)
        self.writer.writerow([
            "time", "scheduled_update_time", "eta_1", "eta_2", "eta_3", "eta_4",
            "local_radius", "carrier_theta_rad", "carrier_phase_reference_rad",
            "envelope_relative_c", "envelope_relative_s", "envelope_ac", "envelope_as",
            "envelope_average_rate_relative_c", "envelope_average_rate_relative_s",
            "envelope_peak_rate_relative_c", "envelope_peak_rate_relative_s",
            "envelope_amplitude", "control_u", "predicted_next_radius",
            "feedback_gain_multiplier", "is_update", "engaged",
            "guard_tripped", "status",
        ])
        self._append_observation()
        self.next_sample_time = self.law.sample_interval

    def ComputeControl(self, current_time):
        if self.last_compute_time is not None \
                and abs(current_time - self.last_compute_time) < 1e-12:
            return self.current_control
        self.last_compute_time = current_time

        appended = False
        while current_time + 1e-10 >= self.next_sample_time:
            self._append_observation()
            self.next_sample_time += self.law.sample_interval
            appended = True

        if current_time + 1e-10 < self.activation_time:
            self.current_control = 0.0
            return self.current_control
        if self.guard_tripped:
            self.current_control = 0.0
            return self.current_control
        if len(self.history) < self.history.maxlen:
            self._trip_guard(current_time, None, math.inf, "insufficient_history")
            return self.current_control

        if not self.started:
            self.started = True
            self.engaged = True
            self.phase_reference = self._carrier_phase(self.activation_time)

        self._interpolate_envelope(current_time)
        if appended:
            eta = self._reduced_state()
            radius = self.law.local_radius(eta)
            if not all(math.isfinite(value) for value in eta) or not math.isfinite(radius):
                self._trip_guard(current_time, eta, radius, "nonfinite_local_state")
                return self.current_control
            if radius > self.law.guard_radius + 1e-12:
                self._trip_guard(current_time, eta, radius, "local_radius_exceeded")
                return self.current_control

        while current_time + 1e-10 >= self.next_update_time and self.engaged:
            self._stroboscopic_update(current_time, self.next_update_time)
            self.next_update_time += self.law.period

        self.current_control = self._physical_control(current_time)
        return self.current_control

    def Finalize(self):
        if self.output_file is not None:
            self.output_file.close()
            self.output_file = None

    def _stroboscopic_update(self, current_time, scheduled_time):
        eta = self._reduced_state()
        radius = self.law.local_radius(eta)
        if not all(math.isfinite(value) for value in eta) or not math.isfinite(radius):
            self._trip_guard(current_time, eta, radius, "nonfinite_local_state")
            return
        if radius > self.law.guard_radius + 1e-12:
            self._trip_guard(current_time, eta, radius, "local_radius_exceeded")
            return

        self.envelope = self.segment_target_envelope[:]
        rate, target, predicted = self.law.control(eta, self.envelope)
        self.segment_start_time = scheduled_time
        self.segment_start_envelope = self.envelope[:]
        self.segment_target_envelope = target
        self.envelope_rate = rate
        predicted_radius = self.law.local_radius(predicted)
        self._write_log(
            current_time, scheduled_time, eta, radius, predicted_radius,
            True, "engaged")

    def _interpolate_envelope(self, current_time):
        fraction = _clip(
            (current_time - self.segment_start_time) / self.law.period, 0.0, 1.0)
        smooth_fraction = fraction ** 3 * (
            10.0 + fraction * (-15.0 + 6.0 * fraction))
        self.envelope = [
            self.segment_start_envelope[i] + smooth_fraction * (
                self.segment_target_envelope[i] - self.segment_start_envelope[i])
            for i in range(2)
        ]

    def _trip_guard(self, current_time, eta, radius, reason):
        self.engaged = False
        self.guard_tripped = True
        self.envelope = [0.0, 0.0]
        self.envelope_rate = [0.0, 0.0]
        self.segment_start_envelope = [0.0, 0.0]
        self.segment_target_envelope = [0.0, 0.0]
        self.current_control = 0.0
        if eta is None:
            eta = [float("nan")] * 4
        self._write_log(
            current_time, self.next_update_time, eta, radius, float("nan"),
            False, reason)

    def _physical_control(self, current_time):
        relative_phase = self._carrier_phase(current_time) - self.phase_reference
        return (self.envelope[0] * math.cos(relative_phase)
                + self.envelope[1] * math.sin(relative_phase))

    def _laboratory_envelope(self):
        cosine = math.cos(self.phase_reference)
        sine = math.sin(self.phase_reference)
        return [
            cosine * self.envelope[0] - sine * self.envelope[1],
            sine * self.envelope[0] + cosine * self.envelope[1],
        ]

    def _carrier_phase(self, current_time):
        return self.law.carrier_omega * current_time + self.law.carrier_phase

    def _append_observation(self):
        observation = []
        for name in self.law.observable_names:
            stem = name.removeprefix("measurement_").removesuffix("_DISPLACEMENT_X")
            axis = 0
            if name.endswith("_DISPLACEMENT_Y"):
                stem = name.removeprefix("measurement_").removesuffix("_DISPLACEMENT_Y")
                axis = 1
            displacement = self.nodes[stem].GetSolutionStepValue(
                KratosMultiphysics.DISPLACEMENT)
            observation.append(float(displacement[axis]))
        self.history.append([
            observation[i] / self.law.observable_scale[i]
            for i in range(len(observation))
        ])

    def _reduced_state(self):
        samples = list(self.history)
        delayed = []
        for index in range(0, len(samples), self.law.shift_steps):
            delayed.extend(samples[index])
        if len(delayed) != len(self.law.delay_basis):
            raise RuntimeError("Online delay vector does not match local delay basis.")
        return self.law.reconstruct_eta(delayed)

    def _write_log(self, current_time, scheduled_time, eta, radius,
                   predicted_radius, is_update, status):
        laboratory = self._laboratory_envelope()
        self.writer.writerow([
            f"{current_time:.12g}", f"{scheduled_time:.12g}",
            *[f"{value:.12g}" for value in eta], f"{radius:.12g}",
            f"{self._carrier_phase(current_time):.12g}",
            f"{self.phase_reference:.12g}",
            f"{self.envelope[0]:.12g}", f"{self.envelope[1]:.12g}",
            f"{laboratory[0]:.12g}", f"{laboratory[1]:.12g}",
            f"{self.envelope_rate[0]:.12g}",
            f"{self.envelope_rate[1]:.12g}",
            f"{self.law.interpolation_peak_rate_factor * self.envelope_rate[0]:.12g}",
            f"{self.law.interpolation_peak_rate_factor * self.envelope_rate[1]:.12g}",
            f"{_norm(self.envelope):.12g}",
            f"{self._physical_control(current_time):.12g}",
            f"{predicted_radius:.12g}",
            f"{self.law.feedback_gain_multiplier:.12g}", int(is_update),
            int(self.engaged), int(self.guard_tripped), status,
        ])
        self.output_file.flush()

    def _find_measurement_nodes(self):
        structure = self.model["Structure"]
        expected = set()
        for name in self.law.observable_names:
            stem = name.removeprefix("measurement_")
            stem = stem.removesuffix("_DISPLACEMENT_X").removesuffix("_DISPLACEMENT_Y")
            expected.add(stem)
        if not expected.issubset(self.SAMPLE_POINTS):
            raise RuntimeError(f"Unknown local-controller sample points: {sorted(expected)}")
        return {
            name: min(structure.Nodes, key=lambda node: (
                node.X0 - self.SAMPLE_POINTS[name][0]) ** 2
                + (node.Y0 - self.SAMPLE_POINTS[name][1]) ** 2)
            for name in expected
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate an FSI2 local LQR artifact.")
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    data = json.loads(args.artifact.read_text())
    errors = validate_export(data)
    print(f"artifact={args.artifact.resolve()}")
    print(f"format_version={data['format_version']}")
    print(f"controller_type={data['controller_type']}")
    print(f"nominal_spectral_radius={data['nominal_closed_loop_spectral_radius']:.12g}")
    print(f"guard_radius={data['guard_radius']:.12g}")
    print("reference_errors=" + ",".join(
        f"{name}:{value:.3e}" for name, value in errors.items()))
    if max(errors.values()) > 1e-10:
        raise RuntimeError(f"MATLAB/Python local-controller mismatch: {errors}")


if __name__ == "__main__":
    main()
