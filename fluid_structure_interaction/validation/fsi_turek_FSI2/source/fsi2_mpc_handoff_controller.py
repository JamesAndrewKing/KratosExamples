"""Amplitude-frequency MPC with a guarded handoff to the local FSI2 LQR."""

import csv
import json
import math
import time
from collections import deque
from pathlib import Path

try:
    import KratosMultiphysics
except ImportError:  # Artifact and transition tests do not require Kratos.
    KratosMultiphysics = None

from fsi2_fourier_envelope_mpc_controller import (
    AmplitudeFrequencyMpc,
    FourierEnvelopeRom,
    ParameterizedOutputMap,
    validate_artifact_schema as validate_mpc_artifact,
    validate_export as validate_mpc_export,
)
from fsi2_local_handoff_lqr_controller import (
    LocalHandoffLaw,
    validate_artifact_schema as validate_local_artifact,
    validate_export as validate_local_export,
)


FORMAT_VERSION = 1
CONTROLLER_TYPE = "mpc_local_handoff"


def _norm(values):
    return math.sqrt(sum(value * value for value in values))


def _clip(value, lower, upper):
    return max(lower, min(upper, value))


def _maximum_difference(first, second):
    return max((abs(float(a) - float(b)) for a, b in zip(first, second)), default=0.0)


class LocalTerminalObjective:
    """Terminal attraction to the measured local chart and zero actuation."""

    def __init__(self, predictor, mpc_data, local_data):
        self.mapping = FourierEnvelopeRom({
            "dynamics_coefficients": predictor["coefficients"],
            "eta_exponents": predictor["eta_exponents"],
            "delta_exponents": predictor["delta_exponents"],
            "harmonic_indices": predictor["harmonic_indices"],
            "eta_center": predictor["eta_center"],
            "eta_scale": predictor["eta_scale"],
            "delta_center": predictor["delta_center"],
            "delta_scale": predictor["delta_scale"],
            "feature_limit": predictor["feature_limit"],
        })
        self.offset = [float(value) for value in predictor["offset"]]
        self.target = [float(value) for value in predictor["target"]]
        self.state_scale = [float(value) for value in predictor["state_scale"]]
        self.state_weight = float(predictor["terminal_state_weight"])
        self.amplitude_weight = float(predictor["terminal_amplitude_weight"])
        self.frequency_weight = float(predictor["terminal_frequency_weight"])
        self.amplitude_scale = (
            float(mpc_data["parameter_upper"][0])
            - float(mpc_data["parameter_lower"][0])
        )
        self.frequency_scale = (
            float(mpc_data["parameter_upper"][1])
            - float(mpc_data["parameter_lower"][1])
        )
        self.frequency_target = float(local_data["carrier_omega_rad_s"])

    def evaluate(self, state):
        value = self.mapping.evaluate(state[:2], state[2], state[3:5])
        return [value[i] + self.offset[i] for i in range(4)]

    def cost(self, state):
        local_eta = self.evaluate(state)
        error = [
            (local_eta[i] - self.target[i]) / self.state_scale[i]
            for i in range(4)
        ]
        return (
            self.state_weight * sum(value * value for value in error)
            + self.amplitude_weight * (state[3] / self.amplitude_scale) ** 2
            + self.frequency_weight
            * ((state[4] - self.frequency_target) / self.frequency_scale) ** 2
        )

    def cost_gradient(self, state):
        value, jac_eta, jac_theta, jac_delta = self.mapping.evaluate_with_jacobian(
            state[:2], state[2], state[3:5]
        )
        local_eta = [value[i] + self.offset[i] for i in range(4)]
        weighted_error = [
            (local_eta[i] - self.target[i]) / self.state_scale[i] ** 2
            for i in range(4)
        ]
        gradient = [0.0] * 5
        for i in range(4):
            factor = 2.0 * self.state_weight * weighted_error[i]
            for j in range(2):
                gradient[j] += factor * jac_eta[i][j]
            gradient[2] += factor * jac_theta[i]
            for j in range(2):
                gradient[3 + j] += factor * jac_delta[i][j]
        gradient[3] += (
            2.0 * self.amplitude_weight * state[3] / self.amplitude_scale ** 2
        )
        gradient[4] += (
            2.0
            * self.frequency_weight
            * (state[4] - self.frequency_target)
            / self.frequency_scale ** 2
        )
        return self.cost(state), gradient


class HandoffGuard:
    """Require a sustained entry into the local controller's fitted domain."""

    def __init__(self, settings, local_omega):
        self.entry_radius = float(settings["entry_radius"])
        self.exit_radius = float(settings["exit_radius"])
        self.maximum_amplitude = float(settings["maximum_entry_amplitude"])
        self.zero_amplitude_tolerance = float(
            settings["zero_amplitude_tolerance"]
        )
        self.maximum_frequency_error = float(
            settings["maximum_entry_frequency_error_rad_s"]
        )
        self.maximum_amplitude_rate = float(
            settings["maximum_entry_amplitude_rate"]
        )
        self.maximum_frequency_rate = float(
            settings["maximum_entry_frequency_rate_rad_s2"]
        )
        self.required_updates = int(settings["required_consecutive_updates"])
        self.local_omega = float(local_omega)
        self.consecutive_updates = 0

    def assess(self, radius, parameters, rates):
        carrier_is_active = parameters[0] > self.zero_amplitude_tolerance
        tests = {
            "radius": radius <= self.entry_radius,
            "amplitude": parameters[0] <= self.maximum_amplitude,
            "frequency": (not carrier_is_active)
            or abs(parameters[1] - self.local_omega)
            <= self.maximum_frequency_error,
            "amplitude_rate": abs(rates[0]) <= self.maximum_amplitude_rate,
            "frequency_rate": (not carrier_is_active)
            or abs(rates[1]) <= self.maximum_frequency_rate,
        }
        if all(tests.values()):
            self.consecutive_updates += 1
        else:
            self.consecutive_updates = 0
        return self.consecutive_updates >= self.required_updates, tests

    def reset(self):
        self.consecutive_updates = 0


class HandoffCaptureMpc(AmplitudeFrequencyMpc):
    """Fixed-deadline capture MPC tailored to a zero-input handoff."""

    def __init__(self, rom, output, mpc_data, handoff_data, warm_starts):
        super().__init__(rom, output, mpc_data)
        self.force_zero_terminal_amplitude = bool(
            handoff_data["force_zero_terminal_amplitude"]
        )
        self.initialize_with_reachability_guesses = bool(
            handoff_data["initialize_with_reachability_guesses"]
        )
        self.capture_warm_starts = warm_starts
        self.correction_bound = [
            float(handoff_data["amplitude_rate_correction_bound"]),
            float(handoff_data["frequency_rate_correction_bound_rad_s2"]),
        ]
        self.initialized_capture_guess = False
        self.reference_guess = None
        self.full_blocks = self.blocks
        self.hold_updates = self.updates_per_block
        self.capture_block = 0
        self.updates_in_block = 0
        self.current_rate = [0.0, 0.0]
        self.current_cost = math.inf

        # The base optimizer normally shifts an infinite-horizon warm start.
        # Capture instead shortens the finite horizon explicitly below.
        self.updates_per_block = self.full_blocks * self.hold_updates + 1

    @property
    def deadline_reached(self):
        return (self.capture_block >= self.full_blocks
                and self.updates_in_block >= self.hold_updates)

    def reset_capture_episode(self):
        self.blocks = self.full_blocks
        self.horizon = self.blocks * self.block_duration
        self.guess = [0.0] * (2 * self.blocks)
        self.update_count = 0
        self.capture_block = 0
        self.updates_in_block = 0
        self.current_rate = [0.0, 0.0]
        self.current_cost = math.inf
        self.initialized_capture_guess = False
        self.reference_guess = None

    def control(self, eta, theta, parameters):
        if 0 < self.updates_in_block < self.hold_updates:
            self.updates_in_block += 1
            return self.current_rate[:], self.current_cost, 0
        if self.deadline_reached:
            raise RuntimeError("Capture deadline reached; assess handoff before replanning.")

        if self.capture_block > 0:
            self.guess = self.guess[2:]
            self.reference_guess = self.reference_guess[2:]
        self.blocks = self.full_blocks - self.capture_block
        self.horizon = self.blocks * self.block_duration
        expected = 2 * self.blocks
        if len(self.guess) != expected:
            raise RuntimeError("Shrinking-horizon warm start has inconsistent length.")
        self.update_count = 0
        rate, cost, iterations = super().control(eta, theta, parameters)
        self.current_rate = rate[:]
        self.current_cost = cost
        self.capture_block += 1
        self.updates_in_block = 1
        return rate, cost, iterations

    def initialize_capture_guess(self, eta, theta, parameters):
        if self.initialized_capture_guess or not self.initialize_with_reachability_guesses:
            return
        half = self.blocks // 2
        peak = 0.9 * self.parameter_upper[0]
        up_count = max(1, half)
        down_count = max(1, self.blocks - up_count)
        up_rate = min(self.rate_bound[0], peak / (up_count * self.block_duration))
        down_rate = max(
            -self.rate_bound[0],
            -(parameters[0] + up_count * self.block_duration * up_rate)
            / (down_count * self.block_duration),
        )
        base = []
        for block in range(self.blocks):
            base.extend([up_rate if block < up_count else down_rate, 0.0])
        guesses = [[0.0] * (2 * self.blocks), base]
        for direction in [-1.0, 1.0]:
            candidate = base[:]
            frequency_rate = direction * 0.5 * self.rate_bound[1]
            for block in range(self.blocks):
                candidate[2 * block + 1] = frequency_rate
            guesses.append(candidate)
        guesses.append([
            value for block in range(self.blocks)
            for value in [
                up_rate if block < up_count else down_rate,
                (0.5 * self.rate_bound[1]
                 if (block < up_count) == (math.sin(theta) >= 0.0)
                 else -0.5 * self.rate_bound[1]),
            ]
        ])
        guesses.extend([
            [value for row in plan for value in row]
            for plan in self.capture_warm_starts["rate_plans"]
        ])
        projected = [self._project(guess, parameters) for guess in guesses]
        self.guess = min(
            projected,
            key=lambda guess: self._objective(eta, theta, parameters, guess),
        )
        self.reference_guess = self.guess[:]
        self.initialized_capture_guess = True

    def _project(self, decision, initial_parameters):
        if self.reference_guess is not None:
            if len(decision) != len(self.reference_guess):
                raise RuntimeError("Capture trust region has inconsistent length.")
            decision = [
                _clip(
                    decision[2 * block + component],
                    self.reference_guess[2 * block + component]
                    - self.correction_bound[component],
                    self.reference_guess[2 * block + component]
                    + self.correction_bound[component],
                )
                for block in range(self.blocks)
                for component in range(2)
            ]
        if not self.force_zero_terminal_amplitude:
            return super()._project(decision, initial_parameters)
        projected = []
        parameters = initial_parameters[:]
        for block in range(self.blocks):
            rates = [
                _clip(
                    decision[2 * block + i], -self.rate_bound[i], self.rate_bound[i]
                )
                for i in range(2)
            ]
            endpoints = [
                _clip(
                    parameters[i] + self.block_duration * rates[i],
                    self.parameter_lower[i], self.parameter_upper[i],
                )
                for i in range(2)
            ]
            remaining = (self.blocks - block - 1) * self.block_duration
            endpoints[0] = min(endpoints[0], self.rate_bound[0] * remaining)
            rates = [
                (endpoints[i] - parameters[i]) / self.block_duration
                for i in range(2)
            ]
            projected.extend(rates)
            parameters = endpoints
        return projected


def validate_artifact_schema(data):
    if int(data.get("format_version", 0)) < FORMAT_VERSION:
        raise ValueError("MPC handoff controller requires format_version >= 1.")
    if data.get("controller_type") != CONTROLLER_TYPE:
        raise ValueError("Artifact is not an mpc_local_handoff controller.")
    mpc_data = data["mpc"]
    local_data = data["local"]
    validate_mpc_artifact(mpc_data)
    validate_local_artifact(local_data)

    for name in ["observable_names", "shift_steps", "delay_count"]:
        if mpc_data[name] != local_data[name]:
            raise ValueError(f"MPC and local artifact disagree on {name}.")
    for name in ["observable_scale"]:
        if _maximum_difference(mpc_data[name], local_data[name]) > 1e-12:
            raise ValueError(f"MPC and local artifact disagree on {name}.")
    if abs(float(mpc_data["sample_interval"]) - float(local_data["sample_interval"])) > 1e-12:
        raise ValueError("MPC and local sample intervals differ.")
    if abs(float(mpc_data["parameter_reference"][1])
           - float(local_data["carrier_omega_rad_s"])) > 1e-11:
        raise ValueError("MPC reference and local carrier frequencies differ.")

    predictor = data["local_predictor"]
    if len(predictor["coefficients"]) != 4 or len(predictor["offset"]) != 4:
        raise ValueError("Local predictor must have four outputs.")
    if len(predictor["target"]) != 4 or len(predictor["state_scale"]) != 4:
        raise ValueError("Local predictor target and scale must have length four.")
    if any(float(value) <= 0.0 for value in predictor["state_scale"]):
        raise ValueError("Local predictor scales must be positive.")
    for name in [
            "terminal_state_weight",
            "terminal_amplitude_weight", "terminal_frequency_weight"]:
        if float(predictor[name]) < 0.0:
            raise ValueError(f"{name} must be nonnegative.")

    warm_starts = data["capture_warm_starts"]
    plans = warm_starts["rate_plans"]
    if len(plans) < 3 or any(len(plan) != round(
            float(mpc_data["prediction_horizon"])
            / float(mpc_data["move_block_duration"])) for plan in plans):
        raise ValueError("Capture library must contain full-horizon plans.")
    if any(len(rate) != 2 for plan in plans for rate in plan):
        raise ValueError("Each capture-library rate must have two entries.")
    if abs(float(warm_starts["block_duration"])
           - float(mpc_data["move_block_duration"])) > 1e-12:
        raise ValueError("Capture-library and MPC block durations differ.")
    if max(float(value) for value in warm_starts["terminal_radius_ratio"]) >= 0.01:
        raise ValueError("A capture warm start terminates outside the local domain.")

    handoff = data["handoff"]
    if not 0.0 < float(handoff["entry_radius"]) < float(handoff["exit_radius"]):
        raise ValueError("Handoff radii must satisfy 0 < entry < exit.")
    if float(handoff["exit_radius"]) > float(local_data["guard_radius"]) + 1e-12:
        raise ValueError("Handoff exit radius exceeds the validated local guard.")
    if not 0.0 < float(handoff["maximum_entry_amplitude"]) \
            <= float(local_data["envelope_limit"]):
        raise ValueError("Entry amplitude must lie in the local envelope domain.")
    if not 0.0 < float(handoff["zero_amplitude_tolerance"]) \
            < float(handoff["maximum_entry_amplitude"]):
        raise ValueError("Zero-amplitude tolerance must lie inside entry bounds.")
    if int(handoff["required_consecutive_updates"]) < 1:
        raise ValueError("At least one consecutive handoff update is required.")
    for name in [
            "amplitude_rate_correction_bound",
            "frequency_rate_correction_bound_rad_s2"]:
        if float(handoff[name]) <= 0.0:
            raise ValueError(f"{name} must be positive.")
    for name in [
            "force_zero_terminal_amplitude",
            "initialize_with_reachability_guesses", "fallback_to_mpc"]:
        if not isinstance(handoff[name], bool):
            raise ValueError(f"{name} must be boolean.")


def validate_export(data):
    validate_artifact_schema(data)
    predictor = LocalTerminalObjective(
        data["local_predictor"], data["mpc"], data["local"]
    )
    maximum_predictor_error = 0.0
    maximum_terminal_gradient_error = 0.0
    for eta, theta, delta, expected in zip(
            data["validation"]["eta"], data["validation"]["theta"],
            data["validation"]["delta"], data["validation"]["local_eta"]):
        state = [float(value) for value in eta] + [float(theta)] \
            + [float(value) for value in delta]
        actual = predictor.evaluate(state)
        maximum_predictor_error = max(
            maximum_predictor_error,
            max(abs(actual[i] - float(expected[i])) for i in range(4)),
        )
        _, analytic = predictor.cost_gradient(state)
        step = 1e-6
        for j in range(5):
            plus = state[:]
            minus = state[:]
            plus[j] += step
            minus[j] -= step
            finite_difference = (predictor.cost(plus) - predictor.cost(minus)) \
                / (2.0 * step)
            maximum_terminal_gradient_error = max(
                maximum_terminal_gradient_error,
                abs(finite_difference - analytic[j]),
            )
    mpc_errors = validate_mpc_export(data["mpc"])
    local_errors = validate_local_export(data["local"])
    return {
        "local_predictor": maximum_predictor_error,
        "terminal_gradient": maximum_terminal_gradient_error,
        "mpc": max(mpc_errors),
        "local": max(local_errors.values()),
    }


class MpcLocalHandoffController:
    """Kratos bridge for nonlinear capture followed by local stabilization."""

    SAMPLE_POINTS = {
        "x_0_30": (0.30, 0.20),
        "x_0_40": (0.40, 0.20),
        "x_0_50": (0.50, 0.20),
        "tip": (0.60, 0.20),
    }

    def __init__(self, model, settings):
        if KratosMultiphysics is None:
            raise RuntimeError("MpcLocalHandoffController must run inside Kratos.")
        data = json.loads(Path(settings["handoff_controller_file_name"].GetString()).read_text())
        validate_artifact_schema(data)
        self.data = data
        self.fallback_to_mpc = bool(data["handoff"]["fallback_to_mpc"])
        self.model = model
        self.local = LocalHandoffLaw(data["local"])
        self.terminal = LocalTerminalObjective(
            data["local_predictor"], data["mpc"], data["local"]
        )
        rom = FourierEnvelopeRom(data["mpc"])
        self.output = ParameterizedOutputMap(data["mpc"])
        self.mpc = HandoffCaptureMpc(
            rom, self.output, data["mpc"], data["handoff"],
            data["capture_warm_starts"],
        )
        self.mpc.set_additional_terminal_objective(self.terminal)
        self.guard = HandoffGuard(data["handoff"], self.local.carrier_omega)

        self.observable_names = list(data["mpc"]["observable_names"])
        self.observable_scale = [float(value) for value in data["mpc"]["observable_scale"]]
        self.mpc_delay_basis = data["mpc"]["delay_basis"]
        self.sample_interval = float(data["mpc"]["sample_interval"])
        self.shift_steps = int(data["mpc"]["shift_steps"])
        self.delay_count = int(data["mpc"]["delay_count"])
        history_length = (self.delay_count - 1) * self.shift_steps + 1
        self.history = deque(maxlen=history_length)
        self.nodes = self._find_measurement_nodes()

        self.activation_time = settings["mpc_activation_time"].GetDouble()
        self.initial_kick_value = settings["mpc_initial_kick_value"].GetDouble()
        self.initial_kick_end_time = settings["mpc_initial_kick_end_time"].GetDouble()
        self.mode = "mpc"
        self.parameters = self.mpc.parameter_reference[:]
        self.parameter_rate = [0.0, 0.0]
        self.parameter_time = self.activation_time
        self.phase = (
            self.parameters[1] * self.activation_time
            + float(data["mpc"].get("carrier_phase", 0.0))
        )
        self.next_mpc_time = self.activation_time

        self.phase_reference = 0.0
        self.local_envelope = [0.0, 0.0]
        self.segment_start_envelope = [0.0, 0.0]
        self.segment_target_envelope = [0.0, 0.0]
        self.segment_start_time = self.activation_time
        self.local_rate = [0.0, 0.0]
        self.next_local_time = math.inf

        self.next_sample_time = 0.0
        self.last_compute_time = None
        self.current_control = 0.0
        self.transition_count = 0

        output_path = Path(settings["handoff_controller_log_file_name"].GetString())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_file = output_path.open("w", newline="")
        self.writer = csv.writer(self.output_file)
        self.writer.writerow([
            "time", "mode", "event", "eta_1", "eta_2",
            "local_eta_1", "local_eta_2", "local_eta_3", "local_eta_4",
            "local_radius", "theta", "A", "Omega", "frequency_hz",
            "A_dot", "Omega_dot", "envelope_relative_c", "envelope_relative_s",
            "envelope_amplitude", "control_u", "objective", "solve_time_seconds",
            "optimizer_iterations", "handoff_streak", "predicted_next_radius",
        ])
        self._append_observation()
        self.next_sample_time = self.sample_interval

    def ComputeControl(self, current_time):
        if self.last_compute_time is not None \
                and abs(current_time - self.last_compute_time) < 1e-12:
            return self.current_control
        self.last_compute_time = current_time

        appended = False
        while current_time + 1e-10 >= self.next_sample_time:
            self._append_observation()
            self.next_sample_time += self.sample_interval
            appended = True

        if current_time + 1e-10 < self.activation_time:
            self.current_control = (
                self.initial_kick_value
                if current_time < self.initial_kick_end_time - 1e-10 else 0.0
            )
            return self.current_control
        if len(self.history) < self.history.maxlen:
            self.current_control = 0.0
            return self.current_control

        if self.mode == "mpc":
            self._advance_mpc_parameters(current_time)
            if current_time + 1e-10 >= self.next_mpc_time:
                self._mpc_update(current_time)
        elif self.mode == "local":
            self._interpolate_local_envelope(current_time)
            if appended:
                local_eta = self._local_eta()
                radius = self.local.local_radius(local_eta)
                if not math.isfinite(radius):
                    self._fail_safe(current_time, local_eta, radius)
                elif radius > self.guard.exit_radius:
                    self._exit_local_domain(current_time, local_eta, radius)
            while self.mode == "local" \
                    and current_time + 1e-10 >= self.next_local_time:
                self._local_update(current_time, self.next_local_time)
                self.next_local_time += self.local.period

        self.current_control = self._physical_control(current_time)
        return self.current_control

    def Finalize(self):
        if self.output_file is not None:
            self.output_file.close()
            self.output_file = None

    def _mpc_update(self, current_time):
        eta = self._mpc_eta()
        local_eta = self._local_eta()
        radius = self.local.local_radius(local_eta)
        if self.mpc.deadline_reached:
            self.parameter_rate = [0.0, 0.0]
        ready, tests = self.guard.assess(radius, self.parameters, self.parameter_rate)
        if ready:
            self._enter_local(current_time, eta, local_eta, radius)
            return
        if self.mpc.deadline_reached and all(tests.values()):
            self._write_log(
                current_time, "handoff_dwell", eta, local_eta, radius,
                self.mpc.current_cost, 0.0, 0, math.nan,
            )
            while self.next_mpc_time <= current_time + 1e-10:
                self.next_mpc_time += self.mpc.control_interval
            return
        if self.mpc.deadline_reached:
            self.mpc.reset_capture_episode()

        start = time.perf_counter()
        self.mpc.initialize_capture_guess(eta, self.phase, self.parameters)
        rate, objective, iterations = self.mpc.control(
            eta, self.phase, self.parameters
        )
        elapsed = time.perf_counter() - start
        self.parameter_rate = rate
        while self.next_mpc_time <= current_time + 1e-10:
            self.next_mpc_time += self.mpc.control_interval
        self._write_log(
            current_time, "mpc_update", eta, local_eta, radius,
            objective, elapsed, iterations, math.nan,
        )

    def _enter_local(self, current_time, eta, local_eta, radius):
        self.mode = "local"
        self.transition_count += 1
        self.phase_reference = self._local_carrier_phase(current_time)
        self.local_envelope = [
            self.parameters[0] * math.cos(self.phase),
            -self.parameters[0] * math.sin(self.phase),
        ]
        self.segment_start_envelope = self.local_envelope[:]
        self.segment_target_envelope = self.local_envelope[:]
        self.segment_start_time = current_time
        self.next_local_time = current_time
        self.local_rate = [0.0, 0.0]
        self._write_log(
            current_time, "mpc_to_local", eta, local_eta, radius,
            math.nan, 0.0, 0, math.nan,
        )

    def _local_update(self, current_time, scheduled_time):
        self.local_envelope = self.segment_target_envelope[:]
        local_eta = self._local_eta()
        radius = self.local.local_radius(local_eta)
        if not math.isfinite(radius):
            self._fail_safe(current_time, local_eta, radius)
            return
        if radius > self.guard.exit_radius:
            self._exit_local_domain(current_time, local_eta, radius)
            return
        rate, target, predicted = self.local.control(local_eta, self.local_envelope)
        self.segment_start_time = scheduled_time
        self.segment_start_envelope = self.local_envelope[:]
        self.segment_target_envelope = target
        self.local_rate = rate
        eta = self._mpc_eta()
        self._write_log(
            current_time, "local_update", eta, local_eta, radius,
            math.nan, 0.0, 0, self.local.local_radius(predicted),
        )

    def _return_to_mpc(self, current_time, local_eta, radius):
        self._interpolate_local_envelope(current_time)
        relative_phase = self._local_carrier_phase(current_time) - self.phase_reference
        amplitude = _norm(self.local_envelope)
        envelope_phase = math.atan2(self.local_envelope[1], self.local_envelope[0])
        self.phase = relative_phase - envelope_phase if amplitude > 1e-14 \
            else relative_phase
        self.parameters = [amplitude, self.local.carrier_omega]
        self.parameter_rate = [0.0, 0.0]
        self.parameter_time = current_time
        self.next_mpc_time = current_time
        self.mpc.reset_capture_episode()
        self.guard.reset()
        self.mode = "mpc"
        self.transition_count += 1
        eta = self._mpc_eta()
        self._write_log(
            current_time, "local_to_mpc", eta, local_eta, radius,
            math.nan, 0.0, 0, math.nan,
        )
        self._mpc_update(current_time)

    def _exit_local_domain(self, current_time, local_eta, radius):
        if self.fallback_to_mpc:
            self._return_to_mpc(current_time, local_eta, radius)
        else:
            self._fail_safe(current_time, local_eta, radius)

    def _fail_safe(self, current_time, local_eta, radius):
        self.mode = "failed"
        self.parameters = [0.0, self.local.carrier_omega]
        self.parameter_rate = [0.0, 0.0]
        self.local_envelope = [0.0, 0.0]
        self.segment_start_envelope = [0.0, 0.0]
        self.segment_target_envelope = [0.0, 0.0]
        eta = [math.nan, math.nan]
        self._write_log(
            current_time, "nonfinite_fail_safe", eta, local_eta, radius,
            math.nan, 0.0, 0, math.nan,
        )

    def _advance_mpc_parameters(self, current_time):
        if current_time <= self.parameter_time:
            return
        dt = current_time - self.parameter_time
        self.phase += (
            self.parameters[1] * dt + 0.5 * self.parameter_rate[1] * dt * dt
        )
        self.parameters = [
            _clip(
                self.parameters[i] + dt * self.parameter_rate[i],
                self.mpc.parameter_lower[i], self.mpc.parameter_upper[i],
            )
            for i in range(2)
        ]
        self.parameter_time = current_time

    def _interpolate_local_envelope(self, current_time):
        fraction = _clip(
            (current_time - self.segment_start_time) / self.local.period, 0.0, 1.0
        )
        smooth = fraction ** 3 * (10.0 + fraction * (-15.0 + 6.0 * fraction))
        self.local_envelope = [
            self.segment_start_envelope[i]
            + smooth * (self.segment_target_envelope[i] - self.segment_start_envelope[i])
            for i in range(2)
        ]

    def _physical_control(self, current_time):
        if self.mode == "mpc":
            return self.parameters[0] * math.cos(self.phase)
        if self.mode == "failed":
            return 0.0
        relative_phase = self._local_carrier_phase(current_time) - self.phase_reference
        return (
            self.local_envelope[0] * math.cos(relative_phase)
            + self.local_envelope[1] * math.sin(relative_phase)
        )

    def _local_carrier_phase(self, current_time):
        return self.local.carrier_omega * current_time + self.local.carrier_phase

    def _append_observation(self):
        observation = []
        for name in self.observable_names:
            stem = name.removeprefix("measurement_").removesuffix("_DISPLACEMENT_X")
            axis = 0
            if name.endswith("_DISPLACEMENT_Y"):
                stem = name.removeprefix("measurement_").removesuffix("_DISPLACEMENT_Y")
                axis = 1
            displacement = self.nodes[stem].GetSolutionStepValue(
                KratosMultiphysics.DISPLACEMENT
            )
            observation.append(float(displacement[axis]))
        self.history.append([
            observation[i] / self.observable_scale[i]
            for i in range(len(observation))
        ])

    def _delayed_state(self):
        samples = list(self.history)
        delayed = []
        for index in range(0, len(samples), self.shift_steps):
            delayed.extend(samples[index])
        if len(delayed) != len(self.mpc_delay_basis):
            raise RuntimeError("Online delay vector does not match exported bases.")
        return delayed

    def _mpc_eta(self):
        delayed = self._delayed_state()
        return [
            sum(self.mpc_delay_basis[i][j] * delayed[i] for i in range(len(delayed)))
            for j in range(len(self.mpc_delay_basis[0]))
        ]

    def _local_eta(self):
        return self.local.reconstruct_eta(self._delayed_state())

    def _write_log(self, current_time, event, eta, local_eta, radius,
                   objective, solve_time, iterations, predicted_radius):
        envelope = self.local_envelope if self.mode == "local" else [math.nan, math.nan]
        amplitude = _norm(envelope) if self.mode == "local" else self.parameters[0]
        self.writer.writerow([
            f"{current_time:.12g}", self.mode, event,
            *[f"{value:.12g}" for value in eta],
            *[f"{value:.12g}" for value in local_eta], f"{radius:.12g}",
            f"{self.phase:.12g}", f"{self.parameters[0]:.12g}",
            f"{self.parameters[1]:.12g}",
            f"{self.parameters[1] / (2.0 * math.pi):.12g}",
            f"{self.parameter_rate[0]:.12g}", f"{self.parameter_rate[1]:.12g}",
            f"{envelope[0]:.12g}", f"{envelope[1]:.12g}",
            f"{amplitude:.12g}", f"{self._physical_control(current_time):.12g}",
            f"{objective:.12g}", f"{solve_time:.12g}", iterations,
            self.guard.consecutive_updates, f"{predicted_radius:.12g}",
        ])
        self.output_file.flush()

    def _find_measurement_nodes(self):
        structure = self.model["Structure"]
        expected = set()
        for name in self.observable_names:
            stem = name.removeprefix("measurement_")
            stem = stem.removesuffix("_DISPLACEMENT_X").removesuffix("_DISPLACEMENT_Y")
            expected.add(stem)
        if not expected.issubset(self.SAMPLE_POINTS):
            raise RuntimeError(f"Unknown handoff-controller sample points: {sorted(expected)}")
        return {
            name: min(
                structure.Nodes,
                key=lambda node: (node.X0 - self.SAMPLE_POINTS[name][0]) ** 2
                + (node.Y0 - self.SAMPLE_POINTS[name][1]) ** 2,
            )
            for name in expected
        }


class EquilibriumCaptureMpcController(MpcLocalHandoffController):
    """Run one fixed-deadline capture episode, then coast at zero input."""

    def __init__(self, model, settings):
        super().__init__(model, settings)
        self.capture_deadline = self.activation_time + self.mpc.horizon

    def ComputeControl(self, current_time):
        control = super().ComputeControl(current_time)
        if self.mode == "coast" and current_time + 1e-10 >= self.next_mpc_time:
            eta = self._mpc_eta()
            local_eta = self._local_eta()
            radius = self.local.local_radius(local_eta)
            self._write_log(
                current_time, "coast_observation", eta, local_eta, radius,
                math.nan, 0.0, 0, math.nan,
            )
            while self.next_mpc_time <= current_time + 1e-10:
                self.next_mpc_time += self.mpc.control_interval
        return control

    def _mpc_update(self, current_time):
        eta = self._mpc_eta()
        local_eta = self._local_eta()
        radius = self.local.local_radius(local_eta)
        if self.mpc.deadline_reached:
            self.parameters[0] = 0.0
            self.parameter_rate = [0.0, 0.0]
            self.parameter_time = current_time
            self.mode = "coast"
            self.next_mpc_time = current_time + self.mpc.control_interval
            self._write_log(
                current_time, "capture_to_coast", eta, local_eta, radius,
                self.mpc.current_cost, 0.0, 0, math.nan,
            )
            return

        start = time.perf_counter()
        self.mpc.initialize_capture_guess(eta, self.phase, self.parameters)
        rate, objective, iterations = self.mpc.control(
            eta, self.phase, self.parameters
        )
        elapsed = time.perf_counter() - start
        self.parameter_rate = rate
        while self.next_mpc_time <= current_time + 1e-10:
            self.next_mpc_time += self.mpc.control_interval
        self._write_log(
            current_time, "capture_update", eta, local_eta, radius,
            objective, elapsed, iterations, math.nan,
        )

    def _physical_control(self, current_time):
        if self.mode == "coast":
            return 0.0
        return super()._physical_control(current_time)


def validate_transition_contract(data):
    """Check handoff dwell, actuator continuity, and phase round trips."""
    validate_artifact_schema(data)
    local = LocalHandoffLaw(data["local"])
    guard = HandoffGuard(data["handoff"], local.carrier_omega)
    settings = data["handoff"]
    parameters = [0.8 * settings["maximum_entry_amplitude"], local.carrier_omega]
    rates = [0.0, 0.0]
    ready_history = [
        guard.assess(0.5 * settings["entry_radius"], parameters, rates)[0]
        for _ in range(settings["required_consecutive_updates"])
    ]
    if any(ready_history[:-1]) or not ready_history[-1]:
        raise ValueError("Handoff dwell contract failed.")

    maximum_identity_error = 0.0
    maximum_roundtrip_error = 0.0
    for theta in [0.0, 0.7, 2.4, 7.1]:
        amplitude = 0.08
        envelope = [amplitude * math.cos(theta), -amplitude * math.sin(theta)]
        mpc_control = amplitude * math.cos(theta)
        local_control = envelope[0]
        maximum_identity_error = max(
            maximum_identity_error, abs(mpc_control - local_control)
        )
        recovered_amplitude = _norm(envelope)
        recovered_phase = -math.atan2(envelope[1], envelope[0])
        maximum_roundtrip_error = max(
            maximum_roundtrip_error,
            abs(recovered_amplitude * math.cos(recovered_phase) - mpc_control),
        )
    return {
        "handoff_identity": maximum_identity_error,
        "fallback_identity": maximum_roundtrip_error,
    }


def simulate_capture(data, maximum_updates):
    """Run capture and local stabilization from measured passive-cycle phases."""
    starts = data["capture_validation"]
    results = [
        _simulate_capture_start(data, maximum_updates, index)
        for index in range(len(starts["eta"]))
    ]
    return {
        "ready": all(result["ready"] for result in results),
        "starts": len(results),
        "maximum_updates": max(result["updates"] for result in results),
        "worst_final_radius": max(result["final_radius"] for result in results),
        "worst_local_radius": max(result["maximum_local_radius"] for result in results),
        "maximum_solve_time": max(result["maximum_solve_time"] for result in results),
    }


def _simulate_capture_start(data, maximum_updates, start_index):
    rom = FourierEnvelopeRom(data["mpc"])
    output = ParameterizedOutputMap(data["mpc"])
    mpc = HandoffCaptureMpc(
        rom, output, data["mpc"], data["handoff"], data["capture_warm_starts"]
    )
    terminal = LocalTerminalObjective(
        data["local_predictor"], data["mpc"], data["local"]
    )
    mpc.set_additional_terminal_objective(terminal)
    local = LocalHandoffLaw(data["local"])
    guard = HandoffGuard(data["handoff"], local.carrier_omega)
    initial = data["capture_validation"]
    state = [float(value) for value in initial["eta"][start_index]] + [
        float(initial["theta"][start_index]),
        *[float(value) for value in mpc.parameter_reference],
    ]
    rate = [0.0, 0.0]
    minimum_radius = math.inf
    ready = False
    elapsed = []
    for update in range(maximum_updates + 1):
        local_eta = terminal.evaluate(state)
        radius = local.local_radius(local_eta)
        minimum_radius = min(minimum_radius, radius)
        if mpc.deadline_reached:
            rate = [0.0, 0.0]
        ready, tests = guard.assess(radius, state[3:5], rate)
        print(
            f"capture[{start_index}:{update:02d}] r={radius:.5g}, "
            f"A={state[3]:.5g}, "
            f"f={state[4] / (2.0 * math.pi):.5g}Hz, "
            f"Adot={rate[0]:.5g}, streak={guard.consecutive_updates}"
        )
        if ready or update == maximum_updates:
            break
        if mpc.deadline_reached and all(tests.values()):
            steps = round(mpc.control_interval / mpc.internal_step)
            for _ in range(steps):
                state = mpc.step(state, rate, mpc.internal_step)
            continue
        if mpc.deadline_reached:
            mpc.reset_capture_episode()
        start = time.perf_counter()
        mpc.initialize_capture_guess(state[:2], state[2], state[3:5])
        rate, _, _ = mpc.control(state[:2], state[2], state[3:5])
        elapsed.append(time.perf_counter() - start)
        steps = round(mpc.control_interval / mpc.internal_step)
        for _ in range(steps):
            state = mpc.step(state, rate, mpc.internal_step)

    maximum_local_radius = radius
    if ready:
        local_eta = terminal.evaluate(state)
        envelope = [
            state[3] * math.cos(state[2]),
            -state[3] * math.sin(state[2]),
        ]
        for _ in range(20):
            _, envelope, local_eta = local.control(local_eta, envelope)
            maximum_local_radius = max(
                maximum_local_radius, local.local_radius(local_eta)
            )
    return {
        "ready": ready,
        "updates": update,
        "minimum_radius": minimum_radius,
        "final_radius": radius,
        "final_amplitude": state[3],
        "maximum_local_radius": maximum_local_radius,
        "maximum_solve_time": max(elapsed, default=0.0),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate the FSI2 MPC-LQR artifact.")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--benchmark-updates", type=int, default=0)
    parser.add_argument("--simulate-capture-updates", type=int, default=0)
    args = parser.parse_args()
    data = json.loads(args.artifact.read_text())
    errors = validate_export(data)
    transitions = validate_transition_contract(data)
    print(f"artifact={args.artifact.resolve()}")
    print("export_errors=" + ",".join(
        f"{name}:{value:.3e}" for name, value in errors.items()))
    print("transition_errors=" + ",".join(
        f"{name}:{value:.3e}" for name, value in transitions.items()))
    print(
        "handoff="
        f"entry_radius:{data['handoff']['entry_radius']},"
        f"exit_radius:{data['handoff']['exit_radius']},"
        f"max_A:{data['handoff']['maximum_entry_amplitude']},"
        f"dwell_updates:{data['handoff']['required_consecutive_updates']}"
    )
    if max([*errors.values(), *transitions.values()]) > 1e-7:
        raise RuntimeError("Combined artifact validation failed.")
    if args.benchmark_updates > 0:
        rom = FourierEnvelopeRom(data["mpc"])
        output = ParameterizedOutputMap(data["mpc"])
        mpc = HandoffCaptureMpc(
            rom, output, data["mpc"], data["handoff"],
            data["capture_warm_starts"],
        )
        mpc.set_additional_terminal_objective(LocalTerminalObjective(
            data["local_predictor"], data["mpc"], data["local"]
        ))
        eta = [row[2] for row in data["mpc"]["validation"]["eta"]]
        theta = data["mpc"]["validation"]["theta"][2]
        parameters = [row[2] for row in data["mpc"]["validation"]["delta"]]
        elapsed = []
        for _ in range(args.benchmark_updates):
            start = time.perf_counter()
            mpc.initialize_capture_guess(eta, theta, parameters)
            mpc.control(eta, theta, parameters)
            elapsed.append(time.perf_counter() - start)
        print(
            f"handoff-aware MPC solve time: first={elapsed[0]:.3f}s, "
            f"max={max(elapsed):.3f}s"
        )
    if args.simulate_capture_updates > 0:
        summary = simulate_capture(data, args.simulate_capture_updates)
        print("capture_summary=" + ",".join(
            f"{name}:{value}" for name, value in summary.items()))


if __name__ == "__main__":
    main()
