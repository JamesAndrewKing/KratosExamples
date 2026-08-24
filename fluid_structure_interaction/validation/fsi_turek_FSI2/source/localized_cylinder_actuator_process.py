import csv
import math
from bisect import bisect_left, bisect_right
from pathlib import Path

import KratosMultiphysics

from fsi2_fourier_envelope_mpc_controller import FourierEnvelopeMpcController
from fsi2_local_handoff_lqr_controller import LocalHandoffLqrController


def Factory(settings, model):
    if not isinstance(settings, KratosMultiphysics.Parameters):
        raise TypeError("expected input shall be a Parameters object")
    return LocalizedCylinderActuatorProcess(model, settings["Parameters"])


class SinusoidalController:

    def __init__(self, settings):
        self.amplitude = settings["amplitude"].GetDouble()
        self.frequency = settings["frequency"].GetDouble()
        self.phase = settings["phase"].GetDouble()
    def ComputeControl(self, time):
        return self.amplitude * math.cos(
            2.0 * math.pi * self.frequency * time + self.phase
        )


class CsvSignalController:

    def __init__(self, settings):
        self.times = []
        self.values = []
        file_name = settings["csv_file_name"].GetString()
        time_column = settings["csv_time_column"].GetString()
        value_column = settings["csv_value_column"].GetString()
        self.interpolation = settings["csv_interpolation"].GetString()

        with Path(file_name).open(newline="") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                self.times.append(float(row[time_column]))
                self.values.append(float(row[value_column]))

        if len(self.times) < 2:
            raise RuntimeError("CSV actuator signals require at least two samples.")
        if self.interpolation not in ("linear", "zoh"):
            raise ValueError('csv_interpolation must be "linear" or "zoh".')
        if any(next_time <= time for time, next_time in zip(self.times, self.times[1:])):
            raise RuntimeError("CSV actuator signal times must be strictly increasing.")

    def ComputeControl(self, time):
        if time <= self.times[0]:
            return self.values[0]
        if time >= self.times[-1]:
            return self.values[-1]

        if self.interpolation == "zoh":
            return self.values[bisect_right(self.times, time) - 1]

        i = bisect_left(self.times, time)
        previous_time = self.times[i - 1]
        next_time = self.times[i]
        ratio = (time - previous_time) / (next_time - previous_time)
        return self.values[i - 1] + ratio * (self.values[i] - self.values[i - 1])


class LocalizedCylinderActuatorProcess(KratosMultiphysics.Process):

    def __init__(self, model, settings):
        KratosMultiphysics.Process.__init__(self)

        default_settings = KratosMultiphysics.Parameters("""{
            "model_part_name" : "FluidModelPart.NoSlip2D_Cylinder",
            "output_file_name" : "actuator_timeseries.csv",
            "cylinder_center" : [0.2, 0.2, 0.0],
            "actuators" : [{
                "name" : "top_blowing_suction",
                "type" : "localized",
                "center_angle_degrees" : 90.0,
                "width_degrees" : 30.0,
                "theta1_degrees" : 60.0,
                "theta2_degrees" : 70.0,
                "direction" : "normal",
                "controller_type" : "sinusoidal",
                "amplitude" : 0.02,
                "frequency" : 1.0,
                "phase" : 0.0,
                "csv_file_name" : "",
                "csv_time_column" : "time",
                "csv_value_column" : "value",
                "csv_interpolation" : "linear",
                "rom_file_name" : "",
                "rom_log_file_name" : "rom_mpc_timeseries.csv",
                "mpc_activation_time" : 15.0,
                "mpc_initial_kick_value" : 0.0,
                "mpc_initial_kick_end_time" : 0.0,
                "local_controller_file_name" : "",
                "local_controller_log_file_name" : "local_lqr_timeseries.csv",
                "local_controller_activation_time" : 3.0,
                "interval" : [0.0, "End"]
            }]
        }""")
        settings.ValidateAndAssignDefaults(default_settings)

        self.model = model
        self.model_part = model[settings["model_part_name"].GetString()]
        self.output_file_name = settings["output_file_name"].GetString()
        self.center = [
            settings["cylinder_center"][0].GetDouble(),
            settings["cylinder_center"][1].GetDouble(),
            settings["cylinder_center"][2].GetDouble()
        ]
        self.actuator_settings = settings["actuators"]
        self.actuators = []
        self.output_file = None
        self.csv_writer = None

    def ExecuteInitialize(self):
        for i in range(self.actuator_settings.size()):
            self.actuators.extend(self._CreateActuators(self.actuator_settings[i]))

        output_path = Path(self.output_file_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_file = output_path.open("w", newline="")
        self.csv_writer = csv.writer(self.output_file)
        self.csv_writer.writerow([
            "time",
            "actuator_name",
            "control_value",
            "weighted_mean_velocity_x",
            "weighted_mean_velocity_y",
            "weighted_mean_velocity_z",
            "number_of_nodes"
        ])

    def ExecuteInitializeSolutionStep(self):
        time = self.model_part.ProcessInfo[KratosMultiphysics.TIME]

        for actuator in self.actuators:
            control_value = 0.0
            if self._IsInInterval(time, actuator["interval"]):
                control_value = (
                    actuator["control_multiplier"]
                    * actuator["controller"].ComputeControl(time)
                )

            weighted_velocity = [0.0, 0.0, 0.0]
            profile_sum = 0.0
            for node, weight, direction in actuator["nodes"]:
                velocity = [control_value * weight * direction[i] for i in range(3)]
                node.SetSolutionStepValue(KratosMultiphysics.VELOCITY_X, velocity[0])
                node.SetSolutionStepValue(KratosMultiphysics.VELOCITY_Y, velocity[1])
                node.SetSolutionStepValue(KratosMultiphysics.VELOCITY_Z, velocity[2])
                weighted_velocity[0] += velocity[0]
                weighted_velocity[1] += velocity[1]
                weighted_velocity[2] += velocity[2]
                profile_sum += weight

            if profile_sum > 0.0:
                weighted_velocity = [value / profile_sum for value in weighted_velocity]

            self.csv_writer.writerow([
                f"{time:.12g}",
                actuator["name"],
                f"{control_value:.12g}",
                f"{weighted_velocity[0]:.12g}",
                f"{weighted_velocity[1]:.12g}",
                f"{weighted_velocity[2]:.12g}",
                len(actuator["nodes"])
            ])
        self.output_file.flush()

    def ExecuteFinalize(self):
        finalized = set()
        for actuator in self.actuators:
            controller = actuator["controller"]
            if id(controller) not in finalized and hasattr(controller, "Finalize"):
                controller.Finalize()
                finalized.add(id(controller))
        if self.output_file is not None:
            self.output_file.close()
            self.output_file = None

    def _CreateActuators(self, settings):
        actuator_type = settings["type"].GetString()
        if actuator_type == "localized":
            return [self._CreateLocalizedActuator(settings)]
        if actuator_type == "rabault_pair":
            return self._CreateRabaultPair(settings)
        raise ValueError(f'Unsupported actuator type "{actuator_type}".')

    def _CreateLocalizedActuator(self, settings):
        center_angle = math.radians(settings["center_angle_degrees"].GetDouble())
        half_width = 0.5 * math.radians(settings["width_degrees"].GetDouble())
        if half_width <= 0.0:
            raise ValueError("Actuator width_degrees must be positive.")

        direction_type = settings["direction"].GetString()
        interval = self._ReadInterval(settings["interval"])
        controller = self._CreateController(settings)

        actuator_nodes = []
        for node in self.model_part.Nodes:
            dx = node.X0 - self.center[0]
            dy = node.Y0 - self.center[1]
            angle = math.atan2(dy, dx)
            angle_distance = self._PeriodicAngleDistance(angle, center_angle)
            if abs(angle_distance) <= half_width:
                weight = 0.5 * (1.0 + math.cos(math.pi * angle_distance / half_width))
                direction = self._CalculateDirection(dx, dy, direction_type)
                actuator_nodes.append((node, weight, direction))

        if not actuator_nodes:
            raise RuntimeError(
                f'Actuator "{settings["name"].GetString()}" selected no nodes.'
            )

        return {
            "name": settings["name"].GetString(),
            "interval": interval,
            "controller": controller,
            "control_multiplier": 1.0,
            "nodes": actuator_nodes
        }

    def _CreateRabaultPair(self, settings):
        theta1 = math.radians(settings["theta1_degrees"].GetDouble())
        theta2 = math.radians(settings["theta2_degrees"].GetDouble())
        width = theta2 - theta1
        if width <= 0.0:
            raise ValueError("Rabault pair requires theta2_degrees > theta1_degrees.")

        theta0 = 0.5 * (theta1 + theta2)
        interval = self._ReadInterval(settings["interval"])
        controller = self._CreateController(settings)
        base_name = settings["name"].GetString()

        upper_nodes = self._CollectRabaultNodes(
            theta0,
            width,
            direction=[1.0, 1.0, 0.0],
            profile_argument=lambda theta: theta - theta0
        )
        lower_nodes = self._CollectRabaultNodes(
            -theta0,
            width,
            direction=[1.0, -1.0, 0.0],
            profile_argument=lambda theta: theta + theta0
        )

        if not upper_nodes:
            raise RuntimeError(f'Rabault actuator "{base_name}_upper" selected no nodes.')
        if not lower_nodes:
            raise RuntimeError(f'Rabault actuator "{base_name}_lower" selected no nodes.')

        return [{
            "name": f"{base_name}_upper",
            "interval": interval,
            "controller": controller,
            "control_multiplier": 1.0,
            "nodes": upper_nodes
        }, {
            "name": f"{base_name}_lower",
            "interval": interval,
            "controller": controller,
            "control_multiplier": -1.0,
            "nodes": lower_nodes
        }]

    def _CollectRabaultNodes(self, center_angle, width, direction, profile_argument):
        half_width = 0.5 * width
        actuator_nodes = []
        for node in self.model_part.Nodes:
            angle = math.atan2(node.Y0 - self.center[1], node.X0 - self.center[0])
            angle_distance = self._PeriodicAngleDistance(angle, center_angle)
            if abs(angle_distance) <= half_width:
                argument = self._PeriodicAngleDistance(profile_argument(angle), 0.0)
                weight = math.cos(math.pi * argument / width)
                actuator_nodes.append((node, weight, direction))
        return actuator_nodes

    def _CreateController(self, settings):
        controller_type = settings["controller_type"].GetString()
        if controller_type == "sinusoidal":
            return SinusoidalController(settings)
        if controller_type == "csv":
            return CsvSignalController(settings)
        if controller_type == "fourier_envelope_mpc":
            return FourierEnvelopeMpcController(self.model, settings)
        if controller_type == "local_handoff_lqr":
            return LocalHandoffLqrController(self.model, settings)
        raise ValueError(f'Unsupported actuator controller_type "{controller_type}".')

    def _CalculateDirection(self, dx, dy, direction_type):
        radius = math.hypot(dx, dy)
        if radius <= 0.0:
            raise RuntimeError("Cannot calculate actuator direction at cylinder center.")

        normal = [dx / radius, dy / radius, 0.0]
        if direction_type == "normal":
            return normal
        if direction_type == "tangential":
            return [-normal[1], normal[0], 0.0]
        raise ValueError(f'Unsupported actuator direction "{direction_type}".')

    @staticmethod
    def _PeriodicAngleDistance(angle, center_angle):
        return math.atan2(
            math.sin(angle - center_angle),
            math.cos(angle - center_angle)
        )

    @staticmethod
    def _ReadInterval(interval_settings):
        start = interval_settings[0].GetDouble()
        if interval_settings[1].IsString():
            end = float("inf")
        else:
            end = interval_settings[1].GetDouble()
        return start, end

    @staticmethod
    def _IsInInterval(time, interval):
        return interval[0] <= time <= interval[1]
