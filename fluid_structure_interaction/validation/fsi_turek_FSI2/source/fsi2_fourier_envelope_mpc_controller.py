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


FORMAT_VERSION = 4
PARAMETER_COORDINATES = ["amplitude", "omega_rad_s"]
OBSOLETE_QUADRATURE_KEYS = {
    "envelope_bound",
    "envelope_rate_bound",
    "envelope_weight",
    "rate_weight",
    "carrier_frequency_hz",
}


def validate_artifact_schema(data):
    """Reject legacy quadrature exports before constructing online objects."""
    format_version = int(data.get("format_version", 0))
    if format_version < FORMAT_VERSION:
        raise ValueError(
            f"Amplitude-frequency MPC requires format_version >= {FORMAT_VERSION}; "
            f"received {format_version}. The artifact is likely a legacy quadrature export."
        )
    coordinates = data.get("parameter_coordinates")
    if coordinates != PARAMETER_COORDINATES:
        raise ValueError(
            "Amplitude-frequency MPC requires parameter_coordinates "
            f"{PARAMETER_COORDINATES}; received {coordinates!r}."
        )
    obsolete = sorted(OBSOLETE_QUADRATURE_KEYS.intersection(data))
    if obsolete:
        raise ValueError(
            "Amplitude-frequency artifact contains obsolete quadrature settings: "
            + ", ".join(obsolete)
        )

    required_vectors = [
        "parameter_reference",
        "parameter_lower",
        "parameter_upper",
        "parameter_rate_bound",
        "parameter_weights",
        "rate_weights",
    ]
    for name in required_vectors:
        values = _vector(data[name])
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"{name} must contain two finite values.")

    lower = _vector(data["parameter_lower"])
    upper = _vector(data["parameter_upper"])
    reference = _vector(data["parameter_reference"])
    rate_bound = _vector(data["parameter_rate_bound"])
    if any(lower[i] >= upper[i] for i in range(2)):
        raise ValueError("Each parameter lower bound must be below its upper bound.")
    if any(not lower[i] <= reference[i] <= upper[i] for i in range(2)):
        raise ValueError("parameter_reference must lie inside the parameter bounds.")
    if any(value <= 0.0 for value in rate_bound):
        raise ValueError("parameter_rate_bound entries must be positive.")

    for name in [
            "control_interval", "internal_step", "move_block_duration",
            "prediction_horizon", "terminal_weight"]:
        value = float(data[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value.")


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

    def evaluate(self, eta, theta, parameters):
        eta_basis = _legendre_product_values(
            eta, self.eta_exponents, self.eta_center, self.eta_scale,
            self.feature_limit)
        delta_basis = _legendre_product_values(
            parameters, self.delta_exponents, self.delta_center, self.delta_scale,
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

    def evaluate_with_jacobian(self, eta, theta, parameters):
        eta_basis, eta_derivatives = _legendre_product_basis(
            eta, self.eta_exponents, self.eta_center, self.eta_scale,
            self.feature_limit)
        delta_basis, delta_derivatives = _legendre_product_basis(
            parameters, self.delta_exponents, self.delta_center, self.delta_scale,
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
            d_delta = [0.0] * len(parameters)
            for h, delta_blocks in enumerate(output_blocks):
                harmonic_value = 0.0
                harmonic_eta = [0.0] * len(eta)
                harmonic_delta = [0.0] * len(parameters)
                for d, eta_coefficients in enumerate(delta_blocks):
                    eta_value = _dot(eta_coefficients, eta_basis)
                    harmonic_value += delta_basis[d] * eta_value
                    for j in range(len(eta)):
                        harmonic_eta[j] += delta_basis[d] * _dot(
                            eta_coefficients, eta_derivatives[j])
                    for j in range(len(parameters)):
                        harmonic_delta[j] += delta_derivatives[j][d] * eta_value
                value += fourier[h] * harmonic_value
                d_theta += fourier_derivative[h] * harmonic_value
                for j in range(len(eta)):
                    d_eta[j] += fourier[h] * harmonic_eta[j]
                for j in range(len(parameters)):
                    d_delta[j] += fourier[h] * harmonic_delta[j]
            values.append(value)
            jac_eta.append(d_eta)
            jac_theta.append(d_theta)
            jac_delta.append(d_delta)
        return values, jac_eta, jac_theta, jac_delta


class ParameterizedOutputMap:
    """Map (eta, phase, amplitude, frequency) to normalized probe outputs."""

    def __init__(self, data):
        self.mapping = FourierEnvelopeRom({
            "dynamics_coefficients": data["output_coefficients"],
            "eta_exponents": data["output_eta_exponents"],
            "delta_exponents": data["output_delta_exponents"],
            "harmonic_indices": data["output_harmonic_indices"],
            "eta_center": data["output_eta_center"],
            "eta_scale": data["output_eta_scale"],
            "delta_center": data["output_delta_center"],
            "delta_scale": data["output_delta_scale"],
            "feature_limit": data["output_feature_limit"],
        })
        self.target = _vector(data["output_target"])
        self.weights = _vector(data["output_weights"])
        self.names = data["output_names"]
        if not (len(self.mapping.coefficients) == len(self.target) == len(self.weights)):
            raise ValueError("Output coefficients, target, and weights are incompatible.")

    def evaluate(self, eta, theta, parameters):
        return self.mapping.evaluate(eta, theta, parameters)

    def cost(self, state):
        output = self.evaluate(state[:2], state[2], state[3:5])
        return sum(self.weights[i] * (output[i] - self.target[i]) ** 2
                   for i in range(len(output)))

    def cost_gradient(self, state):
        output, jac_eta, jac_theta, jac_delta = self.mapping.evaluate_with_jacobian(
            state[:2], state[2], state[3:5])
        jacobian = [jac_eta[i] + [jac_theta[i]] + jac_delta[i]
                    for i in range(len(output))]
        error = [output[i] - self.target[i] for i in range(len(output))]
        cost = sum(self.weights[i] * error[i] ** 2 for i in range(len(output)))
        gradient = [2.0 * sum(
            self.weights[i] * error[i] * jacobian[i][j]
            for i in range(len(output))) for j in range(5)]
        return cost, gradient


class AmplitudeFrequencyMpc:
    """Warm-started (amplitude, frequency)-rate MPC with RK4 sensitivities."""

    def __init__(self, rom, output, data):
        self.rom = rom
        self.output = output
        self.control_interval = float(data["control_interval"])
        self.internal_step = float(data["internal_step"])
        self.block_duration = float(data["move_block_duration"])
        self.horizon = float(data["prediction_horizon"])
        self.parameter_reference = _vector(data["parameter_reference"])
        self.parameter_lower = _vector(data["parameter_lower"])
        self.parameter_upper = _vector(data["parameter_upper"])
        self.rate_bound = _vector(data["parameter_rate_bound"])
        self.parameter_weights = _vector(data["parameter_weights"])
        self.rate_weights = _vector(data["rate_weights"])
        self.parameter_scale = [self.parameter_upper[i] - self.parameter_lower[i]
                                for i in range(2)]
        self.terminal_weight = float(data["terminal_weight"])
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
        self.additional_terminal_objective = None

    def set_additional_terminal_objective(self, objective):
        """Add a differentiable terminal objective without changing MPC mechanics."""
        self.additional_terminal_objective = objective

    def control(self, eta, theta, parameters):
        decision = self._project(self.guess, parameters)
        best, gradient = self._objective_gradient(eta, theta, parameters, decision)
        inverse_hessian = _identity(len(decision))
        iterations = 0
        for iterations in range(1, self.optimizer_iterations + 1):
            projected = self._project(
                [decision[i] - gradient[i] for i in range(len(decision))], parameters)
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
                     for i in range(len(decision))], parameters)
                value = self._objective(eta, theta, parameters, trial)
                displacement = [trial[i] - decision[i] for i in range(len(decision))]
                slope = _dot(gradient, displacement)
                if slope < 0.0 and value <= best + 1e-4 * slope:
                    new_cost, new_gradient = self._objective_gradient(
                        eta, theta, parameters, trial)
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
            state[4], rate[0], rate[1]]

    def _rhs_with_jacobian(self, state, rate):
        eta = state[:2]
        theta = state[2]
        parameters = state[3:5]
        value, jac_eta, jac_theta, jac_delta = self.rom.evaluate_with_jacobian(
            eta, theta, parameters)
        rhs = value + [parameters[1], rate[0], rate[1]]
        jac_state = [[0.0] * 5 for _ in range(5)]
        jac_rate = [[0.0] * 2 for _ in range(5)]
        for i in range(2):
            jac_state[i][:2] = jac_eta[i]
            jac_state[i][2] = jac_theta[i]
            jac_state[i][3:5] = jac_delta[i]
        jac_state[2][4] = 1.0
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

    def _objective(self, eta, theta, parameters, decision):
        state = eta[:] + [theta] + parameters[:]
        cost = 0.0
        for block in range(self.blocks):
            rate = decision[2 * block:2 * block + 2]
            for _ in range(self.steps_per_block):
                cost += self.internal_step * self._stage_cost(state, rate)
                state = self._rk4_state_step(state, rate, self.internal_step)
        cost += self.terminal_weight * self.output.cost(state)
        if self.additional_terminal_objective is not None:
            cost += self.additional_terminal_objective.cost(state)
        return cost

    def _objective_gradient(self, eta, theta, parameters, decision):
        state = eta[:] + [theta] + parameters[:]
        sensitivity = [[0.0] * len(decision) for _ in range(5)]
        gradient = [0.0] * len(decision)
        cost = 0.0
        for block in range(self.blocks):
            rate = decision[2 * block:2 * block + 2]
            for _ in range(self.steps_per_block):
                output_cost, output_gradient = self.output.cost_gradient(state)
                cost += self.internal_step * (
                    output_cost
                    + sum(self.parameter_weights[j] * (
                        (state[3 + j] - self.parameter_reference[j])
                        / self.parameter_scale[j]) ** 2 for j in range(2))
                    + sum(self.rate_weights[j] * (rate[j] / self.rate_bound[j]) ** 2
                          for j in range(2)))
                state_gradient = output_gradient[:]
                for j in range(2):
                    state_gradient[3 + j] += 2.0 * self.parameter_weights[j] * (
                        state[3 + j] - self.parameter_reference[j]) \
                        / self.parameter_scale[j] ** 2
                for j in range(len(decision)):
                    gradient[j] += self.internal_step * sum(
                        state_gradient[i] * sensitivity[i][j] for i in range(5))
                for j in range(2):
                    gradient[2 * block + j] += self.internal_step * (
                        2.0 * self.rate_weights[j] * rate[j] / self.rate_bound[j] ** 2)

                state, jac_state, jac_rate = self._rk4_step(
                    state, rate, self.internal_step)
                sensitivity = _matrix_multiply(jac_state, sensitivity)
                for i in range(5):
                    for j in range(2):
                        sensitivity[i][2 * block + j] += jac_rate[i][j]

        terminal_cost, terminal_gradient = self.output.cost_gradient(state)
        cost += self.terminal_weight * terminal_cost
        for j in range(len(decision)):
            gradient[j] += self.terminal_weight * sum(
                terminal_gradient[i] * sensitivity[i][j] for i in range(5))
        if self.additional_terminal_objective is not None:
            auxiliary_cost, auxiliary_gradient = \
                self.additional_terminal_objective.cost_gradient(state)
            cost += auxiliary_cost
            for j in range(len(decision)):
                gradient[j] += sum(
                    auxiliary_gradient[i] * sensitivity[i][j]
                    for i in range(5))
        return cost, gradient

    def _stage_cost(self, state, rate):
        return (self.output.cost(state)
                + sum(self.parameter_weights[j] * (
                    (state[3 + j] - self.parameter_reference[j])
                    / self.parameter_scale[j]) ** 2 for j in range(2))
                + sum(self.rate_weights[j] * (rate[j] / self.rate_bound[j]) ** 2
                      for j in range(2)))

    def _project(self, decision, initial_parameters):
        projected = []
        parameters = initial_parameters[:]
        for block in range(self.blocks):
            rate = [_clip_to_interval(decision[2 * block + i],
                                      -self.rate_bound[i], self.rate_bound[i])
                    for i in range(2)]
            endpoint = [_clip_to_interval(
                parameters[i] + self.block_duration * rate[i],
                self.parameter_lower[i], self.parameter_upper[i])
                for i in range(2)]
            rate = [(endpoint[i] - parameters[i]) / self.block_duration
                    for i in range(2)]
            projected.extend(rate)
            parameters = endpoint
        return projected


class FourierEnvelopeMpcController:
    """Kratos bridge: probe history -> eta -> (A, Omega) MPC -> actuator."""

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
        validate_artifact_schema(data)
        if data.get("controller_type") != "fourier_envelope_mpc":
            raise ValueError("The selected ROM is not a Fourier-envelope controller artifact.")

        self.model = model
        self.rom = FourierEnvelopeRom(data)
        self.output = ParameterizedOutputMap(data)
        self.mpc = AmplitudeFrequencyMpc(self.rom, self.output, data)
        self.observable_names = data["observable_names"]
        self.observable_scale = _vector(data["observable_scale"])
        self.delay_basis = data["delay_basis"]
        self.sample_interval = float(data["sample_interval"])
        self.shift_steps = int(data["shift_steps"])
        self.delay_count = int(data["delay_count"])
        history_length = (self.delay_count - 1) * self.shift_steps + 1
        self.history = deque(maxlen=history_length)

        self.phase_offset = float(data.get("carrier_phase", 0.0))
        self.activation_time = settings["mpc_activation_time"].GetDouble()
        self.initial_kick_value = settings["mpc_initial_kick_value"].GetDouble()
        self.initial_kick_end_time = settings["mpc_initial_kick_end_time"].GetDouble()
        self.parameters = self.mpc.parameter_reference[:]
        self.parameter_rate = [0.0, 0.0]
        self.parameter_time = self.activation_time
        self.phase = self.parameters[1] * self.activation_time + self.phase_offset
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
            "time", "eta_1", "eta_2", "theta", "A",
            "Omega", "frequency_hz", "A_dot", "Omega_dot",
            "frequency_dot_hz_s", "control_u",
            "objective", "solve_time_seconds", "optimizer_iterations",
            "output_error"
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

        self._advance_parameters(current_time)
        if len(self.history) < self.history.maxlen:
            self.current_control = self._physical_control()
            return self.current_control

        if current_time + 1e-10 >= self.next_control_time:
            eta = self._reduced_state()
            start = time.perf_counter()
            rate, objective, iterations = self.mpc.control(
                eta, self.phase, self.parameters)
            elapsed = time.perf_counter() - start
            self.parameter_rate = rate
            while self.next_control_time <= current_time + 1e-10:
                self.next_control_time += self.mpc.control_interval

            state = eta + [self.phase] + self.parameters
            output_error = math.sqrt(self.output.cost(state))
            self.writer.writerow([
                f"{current_time:.12g}", *[f"{value:.12g}" for value in eta],
                f"{self.phase:.12g}", f"{self.parameters[0]:.12g}",
                f"{self.parameters[1]:.12g}",
                f"{self.parameters[1] / (2.0 * math.pi):.12g}",
                f"{self.parameter_rate[0]:.12g}",
                f"{self.parameter_rate[1]:.12g}",
                f"{self.parameter_rate[1] / (2.0 * math.pi):.12g}",
                f"{self._physical_control():.12g}", f"{objective:.12g}",
                f"{elapsed:.12g}", iterations, f"{output_error:.12g}",
            ])
            self.output_file.flush()

        self.current_control = self._physical_control()
        return self.current_control

    def Finalize(self):
        if self.output_file is not None:
            self.output_file.close()
            self.output_file = None

    def _advance_parameters(self, current_time):
        if current_time <= self.parameter_time:
            return
        dt = current_time - self.parameter_time
        self.phase += (self.parameters[1] * dt
                       + 0.5 * self.parameter_rate[1] * dt * dt)
        self.parameters = [_clip_to_interval(
            self.parameters[i] + dt * self.parameter_rate[i],
            self.mpc.parameter_lower[i], self.mpc.parameter_upper[i])
            for i in range(2)]
        self.parameter_time = current_time

    def _physical_control(self):
        return self.parameters[0] * math.cos(self.phase)

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
    validate_artifact_schema(data)
    rom = FourierEnvelopeRom(data)
    output = ParameterizedOutputMap(data)
    reference = data["validation"]
    eta_columns = _columns(reference["eta"])
    delta_columns = _columns(reference["delta"])
    theta = _vector(reference["theta"])
    dynamics_columns = _columns(reference["dynamics"])
    output_columns = _columns(reference["output"])
    max_dynamics_error = 0.0
    max_output_error = 0.0
    max_jacobian_error = 0.0
    for eta, parameters, angle, expected_dynamics, expected_output in zip(
            eta_columns, delta_columns, theta, dynamics_columns, output_columns):
        value, jac_eta, jac_theta, jac_delta = rom.evaluate_with_jacobian(
            eta, angle, parameters)
        max_dynamics_error = max(max_dynamics_error, max(
            abs(value[i] - expected_dynamics[i]) for i in range(len(value))))
        predicted_output = output.evaluate(eta, angle, parameters)
        max_output_error = max(max_output_error, max(
            abs(predicted_output[i] - expected_output[i])
            for i in range(len(predicted_output))))
        step = 1e-6
        variables = eta + [angle] + parameters
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


def validate_control_contract(data):
    """Exercise the amplitude-frequency state, rate, phase, and output contracts."""
    validate_artifact_schema(data)
    rom = FourierEnvelopeRom(data)
    output = ParameterizedOutputMap(data)
    mpc = AmplitudeFrequencyMpc(rom, output, data)

    parameters = [
        0.5 * (mpc.parameter_lower[i] + mpc.parameter_upper[i])
        for i in range(2)
    ]
    rate = [0.37 * mpc.rate_bound[0], -0.61 * mpc.rate_bound[1]]
    eta = [0.0, 0.0]
    theta = 0.37
    dt = 0.137
    next_state = mpc.step(eta + [theta] + parameters, rate, dt)
    expected_theta = theta + parameters[1] * dt + 0.5 * rate[1] * dt * dt
    phase_error = abs(next_state[2] - expected_theta)

    expected_parameters = [parameters[i] + rate[i] * dt for i in range(2)]
    parameter_error = max(
        abs(next_state[3 + i] - expected_parameters[i]) for i in range(2)
    )
    control = next_state[3] * math.cos(next_state[2])
    identity_error = abs(control - expected_parameters[0] * math.cos(expected_theta))

    projected = mpc._project([10.0, -10.0] * mpc.blocks, parameters)
    projected_parameters = parameters[:]
    max_rate_fraction = 0.0
    for block in range(mpc.blocks):
        block_rate = projected[2 * block:2 * block + 2]
        max_rate_fraction = max(max_rate_fraction, max(
            abs(block_rate[i]) / mpc.rate_bound[i] for i in range(2)
        ))
        projected_parameters = [
            projected_parameters[i] + mpc.block_duration * block_rate[i]
            for i in range(2)
        ]
        if any(projected_parameters[i] < mpc.parameter_lower[i] - 1e-12
               or projected_parameters[i] > mpc.parameter_upper[i] + 1e-12
               for i in range(2)):
            raise ValueError("Projected MPC decision violates parameter bounds.")

    if phase_error > 1e-11:
        raise ValueError(f"Changing-frequency phase propagation error is {phase_error:.3e}.")
    if parameter_error > 1e-12:
        raise ValueError(f"Parameter propagation error is {parameter_error:.3e}.")
    if identity_error > 1e-12:
        raise ValueError(f"Physical-control identity error is {identity_error:.3e}.")
    if max_rate_fraction > 1.0 + 1e-12:
        raise ValueError("Projected MPC decision violates parameter-rate bounds.")

    # Ensure the parameterized output map is callable over the same state.
    output.evaluate(next_state[:2], next_state[2], next_state[3:5])
    rom.evaluate(next_state[:2], next_state[2], next_state[3:5])
    return {
        "phase_error": phase_error,
        "parameter_error": parameter_error,
        "control_identity_error": identity_error,
        "max_projected_rate_fraction": max_rate_fraction,
    }


def benchmark_controller(data, updates=8):
    validate_artifact_schema(data)
    rom = FourierEnvelopeRom(data)
    output = ParameterizedOutputMap(data)
    mpc = AmplitudeFrequencyMpc(rom, output, data)
    eta = [0.5 * value for value in _vector(data["eta_scale"])]
    theta = 0.0
    parameters = mpc.parameter_reference[:]
    times = []
    for _ in range(updates):
        start = time.perf_counter()
        rate, _, _ = mpc.control(eta, theta, parameters)
        times.append(time.perf_counter() - start)
        state = eta + [theta] + parameters
        remaining = mpc.control_interval
        while remaining > 1e-12:
            dt = min(mpc.internal_step, remaining)
            state = mpc.step(state, rate, dt)
            remaining -= dt
        eta, theta, parameters = state[:2], state[2], state[3:5]
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


def _clip_to_interval(value, lower, upper):
    return max(lower, min(upper, value))


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
    if max(errors) > 1e-7:
        raise RuntimeError(
            "Export validation exceeded tolerance: "
            + ", ".join(f"{value:.3e}" for value in errors)
        )
    contract = validate_control_contract(model_data)
    print("max export errors: dynamics={:.3e}, output={:.3e}, jacobian={:.3e}".format(
        *errors))
    print(
        "control contract: format={}, coordinates={}, phase_error={:.3e}, "
        "u_identity_error={:.3e}, max_rate_fraction={:.6f}".format(
            model_data["format_version"], model_data["parameter_coordinates"],
            contract["phase_error"], contract["control_identity_error"],
            contract["max_projected_rate_fraction"],
        )
    )
    print(
        "MPC settings: control_interval={:.6g}s, internal_step={:.6g}s, "
        "block={:.6g}s, horizon={:.6g}s, A=[{:.6g},{:.6g}], f=[{:.6g},{:.6g}]Hz, "
        "rates=[{:.6g},{:.6g}]".format(
            float(model_data["control_interval"]), float(model_data["internal_step"]),
            float(model_data["move_block_duration"]),
            float(model_data["prediction_horizon"]),
            float(model_data["parameter_lower"][0]),
            float(model_data["parameter_upper"][0]),
            float(model_data["parameter_lower"][1]) / (2.0 * math.pi),
            float(model_data["parameter_upper"][1]) / (2.0 * math.pi),
            float(model_data["parameter_rate_bound"][0]),
            float(model_data["parameter_rate_bound"][1]) / (2.0 * math.pi),
        )
    )
    solve_times = benchmark_controller(model_data, arguments.benchmark_updates)
    warm_times = solve_times[1:] or solve_times
    print("MPC solve time: first={:.3f}s, median warm={:.3f}s, max={:.3f}s".format(
        solve_times[0], sorted(warm_times)[len(warm_times) // 2],
        max(solve_times)))
