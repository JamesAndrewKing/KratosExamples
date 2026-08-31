"""Dependency-free Fourier-envelope ROM MPC for the FSI2 actuator pair."""

import csv
import json
import math
import time
from collections import deque
from pathlib import Path

try:
    import KratosMultiphysics
except ImportError:  # ROM and optimizer tests do not require Kratos.
    KratosMultiphysics = None


FORMAT_VERSION = 3
REQUIRED_ARTIFACT_FIELDS = {
    "carrier_frequency_hz",
    "carrier_phase",
    "control_interval",
    "delay_basis",
    "delay_count",
    "delta_center",
    "delta_exponents",
    "delta_scale",
    "dynamics_coefficients",
    "envelope_bound",
    "envelope_rate_bound",
    "envelope_weight",
    "eta_center",
    "eta_exponents",
    "eta_scale",
    "feature_limit",
    "harmonic_indices",
    "internal_step",
    "move_block_duration",
    "observable_names",
    "observable_scale",
    "optimizer_iterations",
    "output_coefficients",
    "output_eta_center",
    "output_eta_exponents",
    "output_eta_scale",
    "output_feature_limit",
    "output_names",
    "output_target",
    "output_weights",
    "prediction_horizon",
    "rate_weight",
    "sample_interval",
    "shift_steps",
    "terminal_envelope_weight",
    "terminal_weight",
    "validation",
}


def validate_artifact_contract(data):
    """Require the recovered format-v3 quadrature controller contract."""
    if data.get("controller_type") != "fourier_envelope_mpc":
        raise ValueError("Expected controller_type='fourier_envelope_mpc'.")
    if data.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Expected Fourier-envelope artifact format_version={FORMAT_VERSION}, "
            f"got {data.get('format_version')!r}."
        )
    missing = sorted(REQUIRED_ARTIFACT_FIELDS.difference(data))
    if missing:
        raise ValueError(
            "Format-v3 quadrature artifact is missing: " + ", ".join(missing)
        )

    delta_center = _vector(data["delta_center"])
    delta_scale = _vector(data["delta_scale"])
    delta_exponents = _integer_matrix(data["delta_exponents"])
    if len(delta_center) != 2 or len(delta_scale) != 2 or any(
            len(exponent) != 2 for exponent in delta_exponents):
        raise ValueError(
            "Format-v3 parameters must be the two quadratures envelope_ac/envelope_as."
        )
    if len(data["dynamics_coefficients"]) != 2:
        raise ValueError("Format-v3 controller requires two reduced coordinates.")


class FourierEnvelopeRom:
    """Evaluate a pretransformed Fourier-Legendre ROM and its Jacobian."""

    def __init__(self, data):
        self.coefficients = data["dynamics_coefficients"]
        self.eta_exponents = _integer_matrix(data["eta_exponents"])
        self.delta_exponents = _integer_matrix(data["delta_exponents"])
        self.harmonics = [int(value) for value in _vector(data["harmonic_indices"])]
        self.eta_center = _vector(data["eta_center"])
        self.eta_scale = _vector(data["eta_scale"])
        self.delta_center = _vector(data["delta_center"])
        self.delta_scale = _vector(data["delta_scale"])
        self.feature_limit = float(data["feature_limit"])
        self.n_eta_basis = len(self.eta_exponents)
        self.n_delta_basis = len(self.delta_exponents)
        expected = self.n_eta_basis * self.n_delta_basis * len(self.harmonics)
        if any(len(row) != expected for row in self.coefficients):
            raise ValueError("Dynamics coefficient count does not match the tensor basis.")

        # Reshape once so runtime evaluation contracts eta, delta, and theta
        # without assembling the full feature vector.
        self.tensor = []
        for row in self.coefficients:
            harmonic_blocks = []
            for h in range(len(self.harmonics)):
                delta_blocks = []
                for d in range(self.n_delta_basis):
                    first = (h * self.n_delta_basis + d) * self.n_eta_basis
                    delta_blocks.append(row[first:first + self.n_eta_basis])
                harmonic_blocks.append(delta_blocks)
            self.tensor.append(harmonic_blocks)

    def evaluate(self, eta, theta, envelope):
        eta_basis = _legendre_product_values(
            eta, self.eta_exponents, self.eta_center, self.eta_scale,
            self.feature_limit)
        delta_basis = _legendre_product_values(
            envelope, self.delta_exponents, self.delta_center, self.delta_scale,
            self.feature_limit)
        fourier = _fourier_basis(theta, self.harmonics)[0]
        values = []
        for output_blocks in self.tensor:
            value = 0.0
            for h, delta_blocks in enumerate(output_blocks):
                harmonic_value = 0.0
                for d, eta_coefficients in enumerate(delta_blocks):
                    harmonic_value += delta_basis[d] * _dot(
                        eta_coefficients, eta_basis)
                value += fourier[h] * harmonic_value
            values.append(value)
        return values

    def evaluate_with_jacobian(self, eta, theta, envelope):
        eta_basis, eta_derivatives = _legendre_product_basis(
            eta, self.eta_exponents, self.eta_center, self.eta_scale,
            self.feature_limit)
        delta_basis, delta_derivatives = _legendre_product_basis(
            envelope, self.delta_exponents, self.delta_center, self.delta_scale,
            self.feature_limit)
        fourier, fourier_derivative = _fourier_basis(theta, self.harmonics)

        values = []
        jac_eta = []
        jac_theta = []
        jac_delta = []
        for output_blocks in self.tensor:
            value = 0.0
            d_eta = [0.0] * len(eta)
            d_theta = 0.0
            d_delta = [0.0] * len(envelope)
            for h, delta_blocks in enumerate(output_blocks):
                harmonic_value = 0.0
                harmonic_eta = [0.0] * len(eta)
                harmonic_delta = [0.0] * len(envelope)
                for d, eta_coefficients in enumerate(delta_blocks):
                    eta_value = _dot(eta_coefficients, eta_basis)
                    harmonic_value += delta_basis[d] * eta_value
                    for j in range(len(eta)):
                        harmonic_eta[j] += delta_basis[d] * _dot(
                            eta_coefficients, eta_derivatives[j])
                    for j in range(len(envelope)):
                        harmonic_delta[j] += delta_derivatives[j][d] * eta_value
                value += fourier[h] * harmonic_value
                d_theta += fourier_derivative[h] * harmonic_value
                for j in range(len(eta)):
                    d_eta[j] += fourier[h] * harmonic_eta[j]
                for j in range(len(envelope)):
                    d_delta[j] += fourier[h] * harmonic_delta[j]
            values.append(value)
            jac_eta.append(d_eta)
            jac_theta.append(d_theta)
            jac_delta.append(d_delta)
        return values, jac_eta, jac_theta, jac_delta


class EnvelopeOutputMap:
    """Map reduced coordinates to the current normalized probe outputs."""

    def __init__(self, data):
        self.coefficients = data["output_coefficients"]
        self.exponents = _integer_matrix(data["output_eta_exponents"])
        self.center = _vector(data["output_eta_center"])
        self.scale = _vector(data["output_eta_scale"])
        self.feature_limit = float(data["output_feature_limit"])
        self.target = _vector(data["output_target"])
        self.weights = _vector(data["output_weights"])
        self.names = data["output_names"]
        if not (len(self.coefficients) == len(self.target) == len(self.weights)):
            raise ValueError("Output coefficients, target, and weights are incompatible.")

    def evaluate(self, eta):
        basis, _ = _legendre_product_basis(
            eta, self.exponents, self.center, self.scale, self.feature_limit)
        return [_dot(row, basis) for row in self.coefficients]

    def cost(self, eta):
        output = self.evaluate(eta)
        return sum(self.weights[i] * (output[i] - self.target[i]) ** 2
                   for i in range(len(output)))

    def cost_gradient(self, eta):
        basis, derivatives = _legendre_product_basis(
            eta, self.exponents, self.center, self.scale, self.feature_limit)
        output = [_dot(row, basis) for row in self.coefficients]
        jacobian = [[_dot(row, derivatives[j]) for j in range(len(eta))]
                    for row in self.coefficients]
        error = [output[i] - self.target[i] for i in range(len(output))]
        cost = sum(self.weights[i] * error[i] ** 2 for i in range(len(output)))
        gradient = [2.0 * sum(
            self.weights[i] * error[i] * jacobian[i][j]
            for i in range(len(output))) for j in range(len(eta))]
        return cost, gradient


class EnvelopeMpc:
    """Warm-started envelope-rate MPC with exact RK4 sensitivities."""

    def __init__(self, rom, output, data):
        self.rom = rom
        self.output = output
        self.omega = 2.0 * math.pi * float(data["carrier_frequency_hz"])
        self.control_interval = float(data["control_interval"])
        self.internal_step = float(data["internal_step"])
        self.block_duration = float(data["move_block_duration"])
        self.horizon = float(data["prediction_horizon"])
        self.envelope_bound = float(data["envelope_bound"])
        self.rate_bound = float(data["envelope_rate_bound"])
        self.envelope_weight = float(data["envelope_weight"])
        self.rate_weight = float(data["rate_weight"])
        self.terminal_weight = float(data["terminal_weight"])
        self.terminal_envelope_weight = float(
            data.get("terminal_envelope_weight", 0.0)
        )
        self.optimizer_iterations = int(data["optimizer_iterations"])
        self.blocks = round(self.horizon / self.block_duration)
        self.steps_per_block = round(self.block_duration / self.internal_step)
        self.updates_per_block = round(self.block_duration / self.control_interval)
        if self.blocks < 1 or self.steps_per_block < 1 or self.updates_per_block < 1:
            raise ValueError("MPC time scales must define positive integer step counts.")
        if abs(self.blocks * self.block_duration - self.horizon) > 1e-10:
            raise ValueError("Prediction horizon must be a multiple of move_block_duration.")
        if abs(self.steps_per_block * self.internal_step - self.block_duration) > 1e-10:
            raise ValueError("Move-block duration must be a multiple of internal_step.")
        self.guess = [0.0] * (2 * self.blocks)
        self.update_count = 0

    def control(self, eta, theta, envelope):
        decision = self._project(self.guess, envelope)
        best, gradient = self._objective_gradient(eta, theta, envelope, decision)
        inverse_hessian = _identity(len(decision))
        iterations = 0
        for iterations in range(1, self.optimizer_iterations + 1):
            projected = self._project(
                [decision[i] - gradient[i] for i in range(len(decision))], envelope)
            projected_gradient = [decision[i] - projected[i]
                                  for i in range(len(decision))]
            if max(abs(value) for value in projected_gradient) < 1e-7:
                break
            direction = [-value for value in _matrix_vector(
                inverse_hessian, projected_gradient)]
            if _dot(gradient, direction) >= 0.0:
                direction = [-value for value in projected_gradient]
                inverse_hessian = _identity(len(decision))

            accepted = False
            step = 1.0
            for _ in range(12):
                trial = self._project(
                    [decision[i] + step * direction[i]
                     for i in range(len(decision))], envelope)
                value = self._objective(eta, theta, envelope, trial)
                displacement = [trial[i] - decision[i] for i in range(len(decision))]
                slope = _dot(gradient, displacement)
                if slope < 0.0 and value <= best + 1e-4 * slope:
                    new_cost, new_gradient = self._objective_gradient(
                        eta, theta, envelope, trial)
                    change = [new_gradient[i] - gradient[i]
                              for i in range(len(decision))]
                    curvature = _dot(displacement, change)
                    if curvature > 1e-10:
                        inverse_hessian = _inverse_bfgs_update(
                            inverse_hessian, displacement, change, curvature)
                    decision = trial
                    best = new_cost
                    gradient = new_gradient
                    accepted = True
                    break
                step *= 0.5
            if not accepted:
                break

        self.update_count += 1
        self.guess = decision[:]
        if self.update_count % self.updates_per_block == 0:
            self.guess = self.guess[2:] + self.guess[-2:]
        return decision[:2], best, iterations

    def step(self, state, rate, dt=None):
        return self._rk4_state_step(
            state, rate, self.internal_step if dt is None else dt)

    def _rhs(self, state, rate):
        return self.rom.evaluate(state[:2], state[2], state[3:5]) + [
            self.omega, rate[0], rate[1]]

    def _rhs_with_jacobian(self, state, rate):
        eta = state[:2]
        theta = state[2]
        envelope = state[3:5]
        value, jac_eta, jac_theta, jac_delta = self.rom.evaluate_with_jacobian(
            eta, theta, envelope)
        rhs = value + [self.omega, rate[0], rate[1]]
        jac_state = [[0.0] * 5 for _ in range(5)]
        jac_rate = [[0.0] * 2 for _ in range(5)]
        for i in range(2):
            jac_state[i][:2] = jac_eta[i]
            jac_state[i][2] = jac_theta[i]
            jac_state[i][3:5] = jac_delta[i]
        jac_rate[3][0] = 1.0
        jac_rate[4][1] = 1.0
        return rhs, jac_state, jac_rate

    def _rk4_step(self, state, rate, dt):
        identity = _identity(5)
        k1, a1, b1 = self._rhs_with_jacobian(state, rate)

        x2 = _add(state, k1, 0.5 * dt)
        dx2 = _matrix_add(identity, a1, 0.5 * dt)
        du2 = _matrix_scale(b1, 0.5 * dt)
        k2, af2, bf2 = self._rhs_with_jacobian(x2, rate)
        a2 = _matrix_multiply(af2, dx2)
        b2 = _matrix_add(_matrix_multiply(af2, du2), bf2)

        x3 = _add(state, k2, 0.5 * dt)
        dx3 = _matrix_add(identity, a2, 0.5 * dt)
        du3 = _matrix_scale(b2, 0.5 * dt)
        k3, af3, bf3 = self._rhs_with_jacobian(x3, rate)
        a3 = _matrix_multiply(af3, dx3)
        b3 = _matrix_add(_matrix_multiply(af3, du3), bf3)

        x4 = _add(state, k3, dt)
        dx4 = _matrix_add(identity, a3, dt)
        du4 = _matrix_scale(b3, dt)
        k4, af4, bf4 = self._rhs_with_jacobian(x4, rate)
        a4 = _matrix_multiply(af4, dx4)
        b4 = _matrix_add(_matrix_multiply(af4, du4), bf4)

        next_state = [state[i] + dt * (
            k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0
                      for i in range(5)]
        jac_state = [[identity[i][j] + dt * (
            a1[i][j] + 2.0 * a2[i][j] + 2.0 * a3[i][j] + a4[i][j]) / 6.0
                      for j in range(5)] for i in range(5)]
        jac_rate = [[dt * (
            b1[i][j] + 2.0 * b2[i][j] + 2.0 * b3[i][j] + b4[i][j]) / 6.0
                    for j in range(2)] for i in range(5)]
        return next_state, jac_state, jac_rate

    def _rk4_state_step(self, state, rate, dt):
        k1 = self._rhs(state, rate)
        k2 = self._rhs(_add(state, k1, 0.5 * dt), rate)
        k3 = self._rhs(_add(state, k2, 0.5 * dt), rate)
        k4 = self._rhs(_add(state, k3, dt), rate)
        return [state[i] + dt * (
            k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0
                for i in range(5)]

    def _objective(self, eta, theta, envelope, decision):
        state = eta[:] + [theta] + envelope[:]
        cost = 0.0
        for block in range(self.blocks):
            rate = decision[2 * block:2 * block + 2]
            for _ in range(self.steps_per_block):
                cost += self.internal_step * self._stage_cost(state, rate)
                state = self._rk4_state_step(state, rate, self.internal_step)
        return (
            cost
            + self.terminal_weight * self.output.cost(state[:2])
            + self.terminal_envelope_weight
            * _norm_squared(state[3:5]) / self.envelope_bound ** 2
        )

    def _objective_gradient(self, eta, theta, envelope, decision):
        state = eta[:] + [theta] + envelope[:]
        sensitivity = [[0.0] * len(decision) for _ in range(5)]
        gradient = [0.0] * len(decision)
        cost = 0.0
        for block in range(self.blocks):
            rate = decision[2 * block:2 * block + 2]
            for _ in range(self.steps_per_block):
                output_cost, output_gradient = self.output.cost_gradient(state[:2])
                cost += self.internal_step * (
                    output_cost
                    + self.envelope_weight * _norm_squared(state[3:5])
                    / self.envelope_bound ** 2
                    + self.rate_weight * _norm_squared(rate) / self.rate_bound ** 2)
                state_gradient = output_gradient + [0.0] + [
                    2.0 * self.envelope_weight * state[3 + j]
                    / self.envelope_bound ** 2 for j in range(2)]
                for j in range(len(decision)):
                    gradient[j] += self.internal_step * sum(
                        state_gradient[i] * sensitivity[i][j] for i in range(5))
                for j in range(2):
                    gradient[2 * block + j] += self.internal_step * (
                        2.0 * self.rate_weight * rate[j] / self.rate_bound ** 2)

                state, jac_state, jac_rate = self._rk4_step(
                    state, rate, self.internal_step)
                sensitivity = _matrix_multiply(jac_state, sensitivity)
                for i in range(5):
                    for j in range(2):
                        sensitivity[i][2 * block + j] += jac_rate[i][j]

        terminal_cost, terminal_gradient = self.output.cost_gradient(state[:2])
        cost += (
            self.terminal_weight * terminal_cost
            + self.terminal_envelope_weight
            * _norm_squared(state[3:5]) / self.envelope_bound ** 2
        )
        for j in range(len(decision)):
            gradient[j] += self.terminal_weight * sum(
                terminal_gradient[i] * sensitivity[i][j] for i in range(2))
            gradient[j] += self.terminal_envelope_weight * sum(
                2.0 * state[3 + i] * sensitivity[3 + i][j]
                / self.envelope_bound ** 2 for i in range(2)
            )
        return cost, gradient

    def _stage_cost(self, state, rate):
        return (self.output.cost(state[:2])
                + self.envelope_weight * _norm_squared(state[3:5])
                / self.envelope_bound ** 2
                + self.rate_weight * _norm_squared(rate) / self.rate_bound ** 2)

    def _project(self, decision, initial_envelope):
        projected = []
        envelope = initial_envelope[:]
        for block in range(self.blocks):
            rate = _project_ball(decision[2 * block:2 * block + 2], self.rate_bound)
            endpoint = [envelope[i] + self.block_duration * rate[i] for i in range(2)]
            endpoint = _project_ball(endpoint, self.envelope_bound)
            rate = [(endpoint[i] - envelope[i]) / self.block_duration for i in range(2)]
            projected.extend(rate)
            envelope = endpoint
        return projected


class FourierEnvelopeMpcController:
    """Kratos bridge: probe history -> eta -> envelope MPC -> actuator value."""

    SAMPLE_POINTS = {
        "x_0_30": (0.30, 0.20),
        "x_0_40": (0.40, 0.20),
        "x_0_50": (0.50, 0.20),
        "tip": (0.60, 0.20),
    }

    def __init__(self, model, settings):
        if KratosMultiphysics is None:
            raise RuntimeError("FourierEnvelopeMpcController must run inside Kratos.")
        with Path(settings["rom_file_name"].GetString()).open() as input_file:
            data = json.load(input_file)
        validate_artifact_contract(data)

        self.model = model
        self.rom = FourierEnvelopeRom(data)
        self.output = EnvelopeOutputMap(data)
        self.mpc = EnvelopeMpc(self.rom, self.output, data)
        self.observable_names = data["observable_names"]
        self.observable_scale = _vector(data["observable_scale"])
        self.delay_basis = data["delay_basis"]
        self.sample_interval = float(data["sample_interval"])
        self.shift_steps = int(data["shift_steps"])
        self.delay_count = int(data["delay_count"])
        history_length = (self.delay_count - 1) * self.shift_steps + 1
        self.history = deque(maxlen=history_length)

        self.omega = self.mpc.omega
        self.phase_offset = float(data.get("carrier_phase", 0.0))
        self.activation_time = settings["mpc_activation_time"].GetDouble()
        self.initial_kick_value = settings["mpc_initial_kick_value"].GetDouble()
        self.initial_kick_end_time = settings["mpc_initial_kick_end_time"].GetDouble()
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
            "objective", "solve_time_seconds", "optimizer_iterations",
            "output_error", "envelope_norm", "envelope_rate_norm"
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
            self.current_control = (self.initial_kick_value
                                    if current_time < self.initial_kick_end_time - 1e-10
                                    else 0.0)
            return self.current_control

        self._advance_envelope(current_time)
        theta = self.omega * current_time + self.phase_offset
        if len(self.history) < self.history.maxlen:
            self.current_control = self._physical_control(theta)
            return self.current_control

        if current_time + 1e-10 >= self.next_control_time:
            eta = self._reduced_state()
            start = time.perf_counter()
            rate, objective, iterations = self.mpc.control(
                eta, theta, self.envelope)
            elapsed = time.perf_counter() - start
            self.envelope_rate = rate
            while self.next_control_time <= current_time + 1e-10:
                self.next_control_time += self.mpc.control_interval

            output_error = math.sqrt(self.output.cost(eta))
            self.writer.writerow([
                f"{current_time:.12g}", *[f"{value:.12g}" for value in eta],
                f"{theta:.12g}", *[f"{value:.12g}" for value in self.envelope],
                *[f"{value:.12g}" for value in self.envelope_rate],
                f"{self._physical_control(theta):.12g}", f"{objective:.12g}",
                f"{elapsed:.12g}", iterations, f"{output_error:.12g}",
                f"{math.sqrt(_norm_squared(self.envelope)):.12g}",
                f"{math.sqrt(_norm_squared(self.envelope_rate)):.12g}"
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
        self.envelope = _project_ball(self.envelope, self.mpc.envelope_bound)
        self.envelope_time = current_time

    def _physical_control(self, theta):
        return self.envelope[0] * math.cos(theta) + self.envelope[1] * math.sin(theta)

    def _append_observation(self):
        observation = []
        for name in self.observable_names:
            stem = name.removeprefix("measurement_").removesuffix("_DISPLACEMENT_X")
            axis = 0
            if name.endswith("_DISPLACEMENT_Y"):
                stem = name.removeprefix("measurement_").removesuffix("_DISPLACEMENT_Y")
                axis = 1
            displacement = self.nodes[stem].GetSolutionStepValue(
                KratosMultiphysics.DISPLACEMENT)
            observation.append(float(displacement[axis]))
        self.history.append([
            observation[i] / self.observable_scale[i] for i in range(len(observation))
        ])

    def _reduced_state(self):
        samples = list(self.history)
        delayed = []
        for index in range(0, len(samples), self.shift_steps):
            delayed.extend(samples[index])
        if len(delayed) != len(self.delay_basis):
            raise RuntimeError("Online delay vector does not match the exported POD basis.")
        return [sum(self.delay_basis[i][j] * delayed[i]
                    for i in range(len(delayed)))
                for j in range(len(self.delay_basis[0]))]

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
    """Check exported basis ordering and analytic derivatives."""
    validate_artifact_contract(data)
    rom = FourierEnvelopeRom(data)
    output = EnvelopeOutputMap(data)
    reference = data["validation"]
    eta_columns = _columns(reference["eta"])
    delta_columns = _columns(reference["delta"])
    theta = _vector(reference["theta"])
    dynamics_columns = _columns(reference["dynamics"])
    output_columns = _columns(reference["output"])
    max_dynamics_error = 0.0
    max_output_error = 0.0
    max_jacobian_error = 0.0
    for eta, envelope, angle, expected_dynamics, expected_output in zip(
            eta_columns, delta_columns, theta, dynamics_columns, output_columns):
        value, jac_eta, jac_theta, jac_delta = rom.evaluate_with_jacobian(
            eta, angle, envelope)
        max_dynamics_error = max(max_dynamics_error, max(
            abs(value[i] - expected_dynamics[i]) for i in range(len(value))))
        predicted_output = output.evaluate(eta)
        max_output_error = max(max_output_error, max(
            abs(predicted_output[i] - expected_output[i])
            for i in range(len(predicted_output))))
        step = 1e-6
        variables = eta + [angle] + envelope
        analytic = [row[:] for row in jac_eta]
        for i in range(len(value)):
            analytic[i].append(jac_theta[i])
            analytic[i].extend(jac_delta[i])
        for j in range(len(variables)):
            plus = variables[:]
            minus = variables[:]
            plus[j] += step
            minus[j] -= step
            fp = rom.evaluate(plus[:2], plus[2], plus[3:5])
            fm = rom.evaluate(minus[:2], minus[2], minus[3:5])
            for i in range(len(value)):
                finite_difference = (fp[i] - fm[i]) / (2.0 * step)
                max_jacobian_error = max(
                    max_jacobian_error, abs(finite_difference - analytic[i][j]))
    return max_dynamics_error, max_output_error, max_jacobian_error


def benchmark_controller(data, updates=8):
    rom = FourierEnvelopeRom(data)
    output = EnvelopeOutputMap(data)
    mpc = EnvelopeMpc(rom, output, data)
    eta = [0.5 * value for value in _vector(data["eta_scale"])]
    theta = 0.0
    envelope = [0.0, 0.0]
    times = []
    for _ in range(updates):
        start = time.perf_counter()
        rate, _, _ = mpc.control(eta, theta, envelope)
        times.append(time.perf_counter() - start)
        state = eta + [theta] + envelope
        remaining = mpc.control_interval
        while remaining > 1e-12:
            dt = min(mpc.internal_step, remaining)
            state = mpc.step(state, rate, dt)
            remaining -= dt
        eta, theta, envelope = state[:2], state[2], state[3:5]
    return times


def _legendre_values_and_derivatives(x, order):
    values = [1.0]
    derivatives = [0.0]
    if order:
        values.append(x)
        derivatives.append(1.0)
    for degree in range(1, order):
        values.append(((2 * degree + 1) * x * values[-1] - degree * values[-2])
                      / (degree + 1))
        derivatives.append(((2 * degree + 1) * (values[-2] + x * derivatives[-1])
                            - degree * derivatives[-2]) / (degree + 1))
    normalization = [math.sqrt((2 * degree + 1) / 2.0)
                     for degree in range(order + 1)]
    return ([normalization[i] * values[i] for i in range(order + 1)],
            [normalization[i] * derivatives[i] for i in range(order + 1)])


def _legendre_product_basis(x, exponents, center, scale, limit):
    scaled = [_clip((x[i] - center[i]) / scale[i], limit) for i in range(len(x))]
    orders = [max(row[i] for row in exponents) for i in range(len(x))]
    data = [_legendre_values_and_derivatives(scaled[i], orders[i])
            for i in range(len(x))]
    basis = []
    derivatives = [[] for _ in x]
    for exponent in exponents:
        factors = [data[j][0][degree] for j, degree in enumerate(exponent)]
        basis.append(math.prod(factors))
        for j, degree in enumerate(exponent):
            others = math.prod(factors[k] for k in range(len(factors)) if k != j)
            inside = abs(scaled[j]) < limit
            derivatives[j].append(
                data[j][1][degree] * others / scale[j] if inside else 0.0)
    return basis, derivatives


def _legendre_product_values(x, exponents, center, scale, limit):
    scaled = [_clip((x[i] - center[i]) / scale[i], limit) for i in range(len(x))]
    orders = [max(row[i] for row in exponents) for i in range(len(x))]
    values = [_legendre_values_and_derivatives(scaled[i], orders[i])[0]
              for i in range(len(x))]
    return [math.prod(values[j][degree] for j, degree in enumerate(exponent))
            for exponent in exponents]


def _fourier_basis(theta, harmonics):
    values = []
    derivatives = []
    for harmonic in harmonics:
        if harmonic == 0:
            values.append(1.0)
            derivatives.append(0.0)
        elif harmonic > 0:
            values.append(math.cos(harmonic * theta))
            derivatives.append(-harmonic * math.sin(harmonic * theta))
        else:
            frequency = -harmonic
            values.append(math.sin(frequency * theta))
            derivatives.append(frequency * math.cos(frequency * theta))
    return values, derivatives


def _integer_matrix(value):
    if value and not isinstance(value[0], list):
        return [[int(entry)] for entry in value]
    return [[int(entry) for entry in row] for row in value]


def _vector(value):
    if not isinstance(value, list):
        return [float(value)]
    if value and isinstance(value[0], list):
        if len(value) == 1:
            return [float(entry) for entry in value[0]]
        if all(len(row) == 1 for row in value):
            return [float(row[0]) for row in value]
    return [float(entry) for entry in value]


def _columns(matrix):
    if not matrix:
        return []
    if not isinstance(matrix[0], list):
        return [[float(value)] for value in matrix]
    return [[float(matrix[i][j]) for i in range(len(matrix))]
            for j in range(len(matrix[0]))]


def _clip(value, bound):
    return max(-bound, min(bound, value))


def _project_ball(vector, radius):
    norm = math.sqrt(_norm_squared(vector))
    if norm <= radius or norm == 0.0:
        return vector[:]
    return [radius * value / norm for value in vector]


def _add(x, dx, scale):
    return [x[i] + scale * dx[i] for i in range(len(x))]


def _norm_squared(x):
    return sum(value * value for value in x)


def _dot(x, y):
    return sum(x[i] * y[i] for i in range(len(x)))


def _matrix_vector(matrix, vector):
    return [_dot(row, vector) for row in matrix]


def _matrix_multiply(left, right):
    return [[sum(left[i][k] * right[k][j] for k in range(len(right)))
             for j in range(len(right[0]))] for i in range(len(left))]


def _matrix_add(left, right, scale=1.0):
    return [[left[i][j] + scale * right[i][j] for j in range(len(left[0]))]
            for i in range(len(left))]


def _matrix_scale(matrix, scale):
    return [[scale * value for value in row] for row in matrix]


def _identity(size):
    return [[float(i == j) for j in range(size)] for i in range(size)]


def _inverse_bfgs_update(matrix, step, gradient_change, curvature):
    size = len(step)
    rho = 1.0 / curvature
    left = [[float(i == j) - rho * step[i] * gradient_change[j]
             for j in range(size)] for i in range(size)]
    right = [[float(i == j) - rho * gradient_change[i] * step[j]
              for j in range(size)] for i in range(size)]
    update = _matrix_multiply(_matrix_multiply(left, matrix), right)
    return [[update[i][j] + rho * step[i] * step[j] for j in range(size)]
            for i in range(size)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate or benchmark the FSI2 envelope MPC.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--benchmark-updates", type=int, default=8)
    arguments = parser.parse_args()
    with arguments.model.open() as model_file:
        model_data = json.load(model_file)
    errors = validate_export(model_data)
    print("max export errors: dynamics={:.3e}, output={:.3e}, jacobian={:.3e}".format(
        *errors))
    solve_times = benchmark_controller(model_data, arguments.benchmark_updates)
    warm_times = solve_times[1:] or solve_times
    print("MPC solve time: first={:.3f}s, median warm={:.3f}s, max={:.3f}s".format(
        solve_times[0], sorted(warm_times)[len(warm_times) // 2],
        max(solve_times)))
