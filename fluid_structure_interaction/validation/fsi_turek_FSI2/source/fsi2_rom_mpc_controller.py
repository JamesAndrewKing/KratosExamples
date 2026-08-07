"""Pure-Python reduced MPC controller for the FSI2 Rabault actuator pair."""

import csv
import json
import math
import time
from collections import deque
from pathlib import Path

try:
    import KratosMultiphysics
except ImportError:  # The reduced model can be tested without Kratos.
    KratosMultiphysics = None


class LegendreRom:
    """Evaluate the exported two-dimensional Fourier-Legendre ROM."""

    def __init__(self, data):
        self.coefficients = data["coefficients"]
        self.eta_exponents = data["eta_exponents"]
        self.u_exponents = [int(value) for value in _vector(data["u_exponents"])]
        self.eta_center = _vector(data["eta_center"])
        self.eta_scale = _vector(data["eta_scale"])
        self.u_center = _scalar(data["u_center"])
        self.u_scale = _scalar(data["u_scale"])
        self.feature_limit = float(data["feature_limit"])

    def evaluate(self, eta, control):
        value, _, _ = self.evaluate_with_jacobian(eta, control)
        return value

    def evaluate_with_jacobian(self, eta, control):
        x = [
            _clip((eta[i] - self.eta_center[i]) / self.eta_scale[i], self.feature_limit)
            for i in range(len(eta))
        ]
        q = _clip((control - self.u_center) / self.u_scale, self.feature_limit)
        eta_data = [_legendre_values_and_derivatives(
            x[i], max(row[i] for row in self.eta_exponents)) for i in range(len(x))]
        u_values, u_derivatives = _legendre_values_and_derivatives(q, max(self.u_exponents))
        eta_basis = []
        eta_derivatives = []
        for exponent in self.eta_exponents:
            factors = [eta_data[j][0][degree] for j, degree in enumerate(exponent)]
            eta_basis.append(math.prod(factors))
            derivative = []
            for j, degree in enumerate(exponent):
                others = math.prod(factors[k] for k in range(len(factors)) if k != j)
                inside = abs(x[j]) < self.feature_limit
                derivative.append(
                    eta_data[j][1][degree] * others / self.eta_scale[j] if inside else 0.0)
            eta_derivatives.append(derivative)
        features = [u_values[degree] * value
                    for degree in self.u_exponents for value in eta_basis]
        eta_feature_derivatives = [
            [u_values[degree] * derivative[j] for degree in self.u_exponents
             for derivative in eta_derivatives]
            for j in range(len(eta))
        ]
        inside_u = abs(q) < self.feature_limit
        u_feature_derivatives = [
            (u_derivatives[degree] / self.u_scale if inside_u else 0.0) * value
            for degree in self.u_exponents for value in eta_basis
        ]
        value = [_dot(row, features) for row in self.coefficients]
        jac_eta = [[_dot(row, eta_feature_derivatives[j]) for j in range(len(eta))]
                   for row in self.coefficients]
        jac_u = [_dot(row, u_feature_derivatives) for row in self.coefficients]
        return value, jac_eta, jac_u

    def step(self, eta, control, dt):
        k1 = self.evaluate(eta, control)
        k2 = self.evaluate(_add(eta, k1, 0.5 * dt), control)
        k3 = self.evaluate(_add(eta, k2, 0.5 * dt), control)
        k4 = self.evaluate(_add(eta, k3, dt), control)
        return [eta[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6
                for i in range(len(eta))]

    def step_with_jacobian(self, eta, control, dt):
        n = len(eta)
        identity = [[float(i == j) for j in range(n)] for i in range(n)]
        k1, j1, b1 = self.evaluate_with_jacobian(eta, control)

        x2 = _add(eta, k1, 0.5 * dt)
        dx2 = _matrix_add(identity, j1, 0.5 * dt)
        du2 = [0.5 * dt * value for value in b1]
        k2, jf2, bu2 = self.evaluate_with_jacobian(x2, control)
        j2 = _matrix_multiply(jf2, dx2)
        b2 = _vector_add(_matrix_vector(jf2, du2), bu2)

        x3 = _add(eta, k2, 0.5 * dt)
        dx3 = _matrix_add(identity, j2, 0.5 * dt)
        du3 = [0.5 * dt * value for value in b2]
        k3, jf3, bu3 = self.evaluate_with_jacobian(x3, control)
        j3 = _matrix_multiply(jf3, dx3)
        b3 = _vector_add(_matrix_vector(jf3, du3), bu3)

        x4 = _add(eta, k3, dt)
        dx4 = _matrix_add(identity, j3, dt)
        du4 = [dt * value for value in b3]
        k4, jf4, bu4 = self.evaluate_with_jacobian(x4, control)
        j4 = _matrix_multiply(jf4, dx4)
        b4 = _vector_add(_matrix_vector(jf4, du4), bu4)

        next_eta = [eta[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6
                    for i in range(n)]
        jac_eta = [[identity[i][j] + dt * (
            j1[i][j] + 2 * j2[i][j] + 2 * j3[i][j] + j4[i][j]) / 6
                    for j in range(n)] for i in range(n)]
        jac_u = [dt * (b1[i] + 2 * b2[i] + 2 * b3[i] + b4[i]) / 6
                 for i in range(n)]
        return next_eta, jac_eta, jac_u


class ReducedMpc:
    """Bounded projected-gradient MPC with exact discrete sensitivities."""

    def __init__(self, rom, target, control_bound, control_interval,
                 prediction_horizon, move_blocks=20, optimizer_iterations=8):
        self.rom = rom
        self.target = target
        self.bound = control_bound
        self.dt = control_interval
        self.steps = max(1, round(prediction_horizon / control_interval))
        self.blocks = min(move_blocks, self.steps)
        self.optimizer_iterations = optimizer_iterations
        self.guess = [0.0] * self.blocks

    def control(self, eta, previous_control):
        decision = self.guess[:]
        best, gradient = self._objective_gradient(eta, decision, previous_control)
        inverse_hessian = _identity(self.blocks)
        for _ in range(self.optimizer_iterations):
            projected_gradient = gradient[:]
            for i in range(self.blocks):
                if ((decision[i] <= -self.bound + 1e-10 and gradient[i] > 0)
                        or (decision[i] >= self.bound - 1e-10 and gradient[i] < 0)):
                    projected_gradient[i] = 0.0
            if max(abs(value) for value in projected_gradient) < 1e-7:
                break
            direction = [-value for value in _matrix_vector(
                inverse_hessian, projected_gradient)]
            if _dot(gradient, direction) >= 0:
                direction = [-value for value in projected_gradient]
                inverse_hessian = _identity(self.blocks)

            step = 1.0
            accepted = False
            for _ in range(12):
                trial = [_clip(decision[i] + step * direction[i], self.bound)
                         for i in range(self.blocks)]
                value = self._objective(eta, trial, previous_control)
                displacement = [trial[i] - decision[i] for i in range(self.blocks)]
                if value <= best + 1e-4 * _dot(gradient, displacement):
                    new_cost, new_gradient = self._objective_gradient(
                        eta, trial, previous_control)
                    change = [new_gradient[i] - gradient[i] for i in range(self.blocks)]
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
        if self.blocks == self.steps:
            self.guess = decision[1:] + [decision[-1]]
        else:
            self.guess = decision
        return decision[0], best

    def _objective(self, eta0, decision, previous_control):
        eta = eta0[:]
        cost = 0.0
        old_control = previous_control
        for step in range(self.steps):
            block = min(self.blocks - 1, step * self.blocks // self.steps)
            control = decision[block]
            error = [eta[i] - self.target[i] for i in range(len(eta))]
            cost += self.dt * (
                _norm_squared(error)
                + 1e-4 * (control / self.bound) ** 2
                + 1e-4 * ((control - old_control) / self.bound) ** 2
            )
            eta = self.rom.step(eta, control, self.dt)
            old_control = control
        error = [eta[i] - self.target[i] for i in range(len(eta))]
        return cost + 50.0 * _norm_squared(error)

    def _objective_gradient(self, eta0, decision, previous_control):
        eta = eta0[:]
        sensitivity = [[0.0] * self.blocks for _ in eta]
        gradient = [0.0] * self.blocks
        cost = 0.0
        old_control = previous_control
        old_block = None
        for step in range(self.steps):
            block = min(self.blocks - 1, step * self.blocks // self.steps)
            control = decision[block]
            error = [eta[i] - self.target[i] for i in range(len(eta))]
            cost += self.dt * (
                _norm_squared(error)
                + 1e-4 * (control / self.bound) ** 2
                + 1e-4 * ((control - old_control) / self.bound) ** 2
            )
            for j in range(self.blocks):
                gradient[j] += 2 * self.dt * sum(
                    error[i] * sensitivity[i][j] for i in range(len(eta)))
            gradient[block] += 2e-4 * self.dt * control / self.bound ** 2
            rate = control - old_control
            gradient[block] += 2e-4 * self.dt * rate / self.bound ** 2
            if old_block is not None:
                gradient[old_block] -= 2e-4 * self.dt * rate / self.bound ** 2

            eta, jac_eta, jac_u = self.rom.step_with_jacobian(eta, control, self.dt)
            sensitivity = [
                [sum(jac_eta[i][k] * sensitivity[k][j] for k in range(len(eta)))
                 + jac_u[i] * float(j == block) for j in range(self.blocks)]
                for i in range(len(eta))
            ]
            old_control = control
            old_block = block

        error = [eta[i] - self.target[i] for i in range(len(eta))]
        cost += 50.0 * _norm_squared(error)
        for j in range(self.blocks):
            gradient[j] += 100.0 * sum(
                error[i] * sensitivity[i][j] for i in range(len(eta)))
        return cost, gradient


class RomMpcController:
    """Kratos-facing controller: measurements -> delays -> eta -> MPC -> u."""

    SAMPLE_POINTS = {
        "x_0_30": (0.30, 0.20),
        "x_0_40": (0.40, 0.20),
        "x_0_50": (0.50, 0.20),
        "tip": (0.60, 0.20),
    }

    def __init__(self, model, settings):
        if KratosMultiphysics is None:
            raise RuntimeError("RomMpcController must run inside Kratos.")
        with Path(settings["rom_file_name"].GetString()).open() as input_file:
            data = json.load(input_file)

        self.model = model
        self.rom = LegendreRom(data)
        self.observable_names = data["observable_names"]
        self.observable_scale = _vector(data["observable_scale"])
        self.delay_basis = data["delay_basis"]
        self.sample_interval = float(data["sample_interval"])
        self.shift_steps = int(data["shift_steps"])
        self.delay_count = int(data["delay_count"])
        history_length = (self.delay_count - 1) * self.shift_steps + 1
        self.history = deque(maxlen=history_length)

        control_interval = settings["mpc_control_interval"].GetDouble()
        prediction_horizon = settings["mpc_prediction_horizon"].GetDouble()
        control_bound = settings["mpc_control_bound"].GetDouble()
        self.mpc = ReducedMpc(
            self.rom,
            _vector(data["eta_equilibrium"]),
            control_bound,
            control_interval,
            prediction_horizon,
            settings["mpc_move_blocks"].GetInt(),
            settings["mpc_optimizer_iterations"].GetInt(),
        )
        self.control_interval = control_interval
        self.activation_time = settings["mpc_activation_time"].GetDouble()
        self.max_increment = settings["mpc_max_control_increment"].GetDouble()
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
            "time", "eta_1", "eta_2", "control_u", "objective",
            "solve_time_seconds", "reduced_radius"
        ])
        self._append_observation()
        self.next_sample_time = self.sample_interval

    def ComputeControl(self, current_time):
        if self.last_compute_time is not None and abs(current_time - self.last_compute_time) < 1e-12:
            return self.current_control
        self.last_compute_time = current_time

        while current_time + 1e-10 >= self.next_sample_time:
            self._append_observation()
            self.next_sample_time += self.sample_interval

        if (current_time + 1e-10 < self.next_control_time
                or len(self.history) < self.history.maxlen):
            return self.current_control

        eta = self._reduced_state()
        start = time.perf_counter()
        requested, objective = self.mpc.control(eta, self.current_control)
        elapsed = time.perf_counter() - start
        change = _clip(requested - self.current_control, self.max_increment)
        self.current_control += change
        self.next_control_time += self.control_interval

        target = self.mpc.target
        radius = math.sqrt(sum((eta[i] - target[i]) ** 2 for i in range(len(eta))))
        self.writer.writerow([
            f"{current_time:.12g}", *[f"{value:.12g}" for value in eta],
            f"{self.current_control:.12g}", f"{objective:.12g}",
            f"{elapsed:.12g}", f"{radius:.12g}"
        ])
        self.output_file.flush()
        return self.current_control

    def Finalize(self):
        if self.output_file is not None:
            self.output_file.close()
            self.output_file = None

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
        return [sum(self.delay_basis[i][j] * delayed[i] for i in range(len(delayed)))
                for j in range(len(self.delay_basis[0]))]

    def _find_measurement_nodes(self):
        structure = self.model["Structure"]
        nodes = {}
        for name, (x, y) in self.SAMPLE_POINTS.items():
            nodes[name] = min(
                structure.Nodes,
                key=lambda node: (node.X0 - x) ** 2 + (node.Y0 - y) ** 2,
            )
        expected = set()
        for name in self.observable_names:
            stem = name.removeprefix("measurement_")
            stem = stem.removesuffix("_DISPLACEMENT_X").removesuffix("_DISPLACEMENT_Y")
            expected.add(stem)
        if expected != set(nodes):
            raise RuntimeError(f"ROM observables {sorted(expected)} do not match FSI2 sample points.")
        return nodes


def _legendre_values(x, order):
    return _legendre_values_and_derivatives(x, order)[0]


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


def _vector(value):
    if not isinstance(value, list):
        return [float(value)]
    if value and isinstance(value[0], list):
        if len(value) == 1:
            return [float(entry) for entry in value[0]]
        return [float(row[0]) for row in value]
    return [float(entry) for entry in value]


def _scalar(value):
    return _vector(value)[0]


def _clip(value, bound):
    return max(-bound, min(bound, value))


def _add(x, dx, scale):
    return [x[i] + scale * dx[i] for i in range(len(x))]


def _norm_squared(x):
    return sum(value * value for value in x)


def _dot(x, y):
    return sum(x[i] * y[i] for i in range(len(x)))


def _vector_add(x, y):
    return [x[i] + y[i] for i in range(len(x))]


def _matrix_vector(matrix, vector):
    return [_dot(row, vector) for row in matrix]


def _matrix_multiply(left, right):
    return [[sum(left[i][k] * right[k][j] for k in range(len(right)))
             for j in range(len(right[0]))] for i in range(len(left))]


def _matrix_add(left, right, scale):
    return [[left[i][j] + scale * right[i][j] for j in range(len(left[0]))]
            for i in range(len(left))]


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
