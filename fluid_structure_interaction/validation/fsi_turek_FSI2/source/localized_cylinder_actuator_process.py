import csv
import math
from pathlib import Path

import KratosMultiphysics


def Factory(settings, model):
    if not isinstance(settings, KratosMultiphysics.Parameters):
        raise TypeError("expected input shall be a Parameters object")
    return LocalizedCylinderActuatorProcess(model, settings["Parameters"])


class SinusoidalController:

    def __init__(self, settings):
        self.amplitude = settings["amplitude"].GetDouble()
        self.frequency = settings["frequency"].GetDouble()
        self.phase = settings["phase"].GetDouble()
        self.offset = settings["offset"].GetDouble()

    def ComputeControl(self, time, feedback):
        return self.offset + self.amplitude * math.sin(
            2.0 * math.pi * self.frequency * time + self.phase
        )


class CsvSignalController:

    def __init__(self, settings):
        self.times = []
        self.values = []
        file_name = settings["csv_file_name"].GetString()
        time_column = settings["csv_time_column"].GetString()
        value_column = settings["csv_value_column"].GetString()

        with Path(file_name).open(newline="") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                self.times.append(float(row[time_column]))
                self.values.append(float(row[value_column]))

        if len(self.times) < 2:
            raise RuntimeError("CSV actuator signals require at least two samples.")

    def ComputeControl(self, time, feedback):
        if time <= self.times[0]:
            return self.values[0]
        if time >= self.times[-1]:
            return self.values[-1]

        for i in range(1, len(self.times)):
            if time <= self.times[i]:
                previous_time = self.times[i - 1]
                next_time = self.times[i]
                ratio = (time - previous_time) / (next_time - previous_time)
                return self.values[i - 1] + ratio * (self.values[i] - self.values[i - 1])

        return self.values[-1]


class BeamTipFeedbackController:

    def __init__(self, model, settings):
        self.model = model
        self.model_part_name = settings["feedback_model_part_name"].GetString()
        self.target_x = settings["feedback_point"][0].GetDouble()
        self.target_y = settings["feedback_point"][1].GetDouble()
        self.kp = settings["proportional_gain"].GetDouble()
        self.kd = settings["derivative_gain"].GetDouble()
        self.control_sign = settings["control_sign"].GetDouble()
        self.max_abs_control = settings["max_abs_control"].GetDouble()
        self.feedback_delay = settings["feedback_delay"].GetDouble()
        self.history = []
        self.node = None

    def ComputeControl(self, time, feedback):
        if self.node is None:
            self.node = self._FindNearestNode()

        displacement = self.node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)
        velocity = self.node.GetSolutionStepValue(KratosMultiphysics.VELOCITY)
        self.history.append((time, displacement[1], velocity[1]))
        delayed_displacement_y, delayed_velocity_y = self._GetDelayedFeedback(time)
        control = self.control_sign * (self.kp * delayed_displacement_y + self.kd * delayed_velocity_y)

        if self.max_abs_control > 0.0:
            control = max(-self.max_abs_control, min(self.max_abs_control, control))

        feedback["beam_tip_displacement_y"] = displacement[1]
        feedback["beam_tip_velocity_y"] = velocity[1]
        feedback["delayed_beam_tip_displacement_y"] = delayed_displacement_y
        feedback["delayed_beam_tip_velocity_y"] = delayed_velocity_y
        return control

    def _GetDelayedFeedback(self, time):
        target_time = time - self.feedback_delay
        if target_time <= self.history[0][0]:
            return self.history[0][1], self.history[0][2]

        previous_time, previous_displacement_y, previous_velocity_y = self.history[0]
        for next_time, next_displacement_y, next_velocity_y in self.history[1:]:
            if target_time <= next_time:
                ratio = (target_time - previous_time) / (next_time - previous_time)
                displacement_y = previous_displacement_y + ratio * (
                    next_displacement_y - previous_displacement_y
                )
                velocity_y = previous_velocity_y + ratio * (
                    next_velocity_y - previous_velocity_y
                )
                return displacement_y, velocity_y
            previous_time = next_time
            previous_displacement_y = next_displacement_y
            previous_velocity_y = next_velocity_y

        return self.history[-1][1], self.history[-1][2]

    def _FindNearestNode(self):
        model_part = self.model[self.model_part_name]
        nearest_node = None
        nearest_distance_squared = float("inf")

        for node in model_part.Nodes:
            distance_squared = (
                (node.X0 - self.target_x) ** 2
                + (node.Y0 - self.target_y) ** 2
            )
            if distance_squared < nearest_distance_squared:
                nearest_node = node
                nearest_distance_squared = distance_squared

        if nearest_node is None:
            raise RuntimeError(
                f'Feedback controller found no nodes in "{self.model_part_name}".'
            )
        return nearest_node


class BeamTipPhaseController:

    def __init__(self, model, settings):
        self.model = model
        self.model_part_name = settings["feedback_model_part_name"].GetString()
        self.target_x = settings["feedback_point"][0].GetDouble()
        self.target_y = settings["feedback_point"][1].GetDouble()
        self.frequency = settings["oscillator_frequency"].GetDouble()
        self.phase_shift = settings["oscillator_phase_shift"].GetDouble()
        self.gain = settings["oscillator_gain"].GetDouble()
        self.control_sign = settings["control_sign"].GetDouble()
        self.max_abs_control = settings["max_abs_control"].GetDouble()
        self.node = None

        if self.frequency <= 0.0:
            raise ValueError("oscillator_frequency must be positive.")

    def ComputeControl(self, time, feedback):
        if self.node is None:
            self.node = self._FindNearestNode()

        displacement = self.node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)
        velocity = self.node.GetSolutionStepValue(KratosMultiphysics.VELOCITY)
        omega = 2.0 * math.pi * self.frequency
        normalized_velocity_y = velocity[1] / omega
        phase_signal = (
            displacement[1] * math.cos(self.phase_shift)
            + normalized_velocity_y * math.sin(self.phase_shift)
        )
        control = self.control_sign * self.gain * phase_signal

        if self.max_abs_control > 0.0:
            control = max(-self.max_abs_control, min(self.max_abs_control, control))

        feedback["beam_tip_displacement_y"] = displacement[1]
        feedback["beam_tip_velocity_y"] = velocity[1]
        feedback["phase_signal"] = phase_signal
        return control

    def _FindNearestNode(self):
        model_part = self.model[self.model_part_name]
        nearest_node = None
        nearest_distance_squared = float("inf")

        for node in model_part.Nodes:
            distance_squared = (
                (node.X0 - self.target_x) ** 2
                + (node.Y0 - self.target_y) ** 2
            )
            if distance_squared < nearest_distance_squared:
                nearest_node = node
                nearest_distance_squared = distance_squared

        if nearest_node is None:
            raise RuntimeError(
                f'Phase controller found no nodes in "{self.model_part_name}".'
            )
        return nearest_node


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
                "offset" : 0.0,
                "csv_file_name" : "",
                "csv_time_column" : "time",
                "csv_value_column" : "value",
                "feedback_model_part_name" : "Structure",
                "feedback_point" : [0.6, 0.2, 0.0],
                "proportional_gain" : 0.0,
                "derivative_gain" : 0.0,
                "control_sign" : -1.0,
                "max_abs_control" : 0.0,
                "feedback_delay" : 0.0,
                "oscillator_frequency" : 3.8,
                "oscillator_phase_shift" : 0.0,
                "oscillator_gain" : 200.0,
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
        feedback = {}

        for actuator in self.actuators:
            control_value = 0.0
            if self._IsInInterval(time, actuator["interval"]):
                control_value = (
                    actuator["control_multiplier"]
                    * actuator["controller"].ComputeControl(time, feedback)
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
        if controller_type == "beam_tip_feedback":
            return BeamTipFeedbackController(self.model, settings)
        if controller_type == "beam_tip_phase":
            return BeamTipPhaseController(self.model, settings)
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
