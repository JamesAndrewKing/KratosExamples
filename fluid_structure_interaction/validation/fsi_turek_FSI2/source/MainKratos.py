import sys
import time
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime

import KratosMultiphysics


def ReadFloatEnvironmentVariable(name, default_value):
    value = os.environ.get(name)
    if value is None:
        return default_value
    return float(value)


def ReadBooleanEnvironmentVariable(name, default_value):
    value = os.environ.get(name)
    if value is None:
        return default_value
    return value.lower() not in ("0", "false", "no", "off")


def CreateRunOutputDirectory():
    requested_output_directory = os.environ.get("KRATOS_FSI_RUN_OUTPUT_DIRECTORY")
    if requested_output_directory:
        output_directory = Path(requested_output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        return output_directory

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    label = os.environ.get("KRATOS_FSI_RUN_LABEL")
    if label:
        label = "".join(character if character.isalnum() or character in "-_" else "_" for character in label)
        run_name = f"{run_name}_{label}"
    output_directory = Path("run_outputs") / run_name
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def AddParaViewOutput(project_parameters, output_directory):
    output_interval = ReadFloatEnvironmentVariable("KRATOS_FSI_OUTPUT_INTERVAL", 0.01)
    project_parameters["output_processes"]["vtk_output"] = [{
        "python_module": "vtk_output_process",
        "kratos_module": "KratosMultiphysics",
        "process_name": "VtkOutputProcess",
        "help": "This process writes postprocessing files for ParaView",
        "Parameters": {
            "model_part_name": "Structure",
            "output_control_type": "time",
            "output_interval": output_interval,
            "file_format": "binary",
            "output_precision": 7,
            "output_sub_model_parts": False,
            "output_path": str(output_directory / "vtk_output_structure"),
            "save_output_files_in_folder": True,
            "nodal_solution_step_data_variables": ["DISPLACEMENT", "REACTION", "VELOCITY", "ACCELERATION"],
            "nodal_data_value_variables": [],
            "element_data_value_variables": [],
            "condition_data_value_variables": [],
            "gauss_point_variables_extrapolated_to_nodes": ["VON_MISES_STRESS"]
        }
    }, {
        "python_module": "vtk_output_process",
        "kratos_module": "KratosMultiphysics",
        "process_name": "VtkOutputProcess",
        "help": "This process writes postprocessing files for ParaView",
        "Parameters": {
            "model_part_name": "FluidModelPart.fluid_computational_model_part",
            "output_control_type": "time",
            "output_interval": output_interval,
            "file_format": "binary",
            "output_precision": 7,
            "output_sub_model_parts": False,
            "output_path": str(output_directory / "vtk_output_fluid"),
            "save_output_files_in_folder": True,
            "nodal_solution_step_data_variables": ["VELOCITY", "PRESSURE", "MESH_DISPLACEMENT"],
            "nodal_data_value_variables": [],
            "element_data_value_variables": [],
            "condition_data_value_variables": []
        }
    }]


def AddCylinderActuatorProcess(project_parameters, output_directory):
    controller_type = os.environ.get("KRATOS_FSI_CONTROLLER_TYPE", "sinusoidal")
    controller_settings = {"controller_type": controller_type}
    if controller_type == "sinusoidal":
        controller_settings.update({
            "amplitude": ReadFloatEnvironmentVariable(
                "KRATOS_FSI_ACTUATOR_AMPLITUDE", 0.0),
            "frequency": ReadFloatEnvironmentVariable(
                "KRATOS_FSI_ACTUATOR_FREQUENCY", 0.0),
            "phase": ReadFloatEnvironmentVariable("KRATOS_FSI_ACTUATOR_PHASE", 0.0),
        })
    elif controller_type == "csv":
        controller_settings.update({
            "csv_file_name": os.environ.get("KRATOS_FSI_ACTUATOR_CSV_FILE", ""),
            "csv_time_column": os.environ.get(
                "KRATOS_FSI_ACTUATOR_CSV_TIME_COLUMN", "time"),
            "csv_value_column": os.environ.get(
                "KRATOS_FSI_ACTUATOR_CSV_VALUE_COLUMN", "value"),
            "csv_interpolation": os.environ.get(
                "KRATOS_FSI_ACTUATOR_CSV_INTERPOLATION", "linear"),
        })
    elif controller_type == "fourier_envelope_mpc":
        controller_settings.update({
            "rom_file_name": os.environ.get("KRATOS_FSI_ROM_FILE", ""),
            "rom_log_file_name": str(output_directory / "rom_mpc_timeseries.csv"),
            "mpc_activation_time": ReadFloatEnvironmentVariable(
                "KRATOS_FSI_MPC_ACTIVATION_TIME", 20.0),
            "mpc_initial_kick_value": ReadFloatEnvironmentVariable(
                "KRATOS_FSI_MPC_INITIAL_KICK_VALUE", 0.0),
            "mpc_initial_kick_end_time": ReadFloatEnvironmentVariable(
                "KRATOS_FSI_MPC_INITIAL_KICK_END_TIME", 0.0),
        })
    elif controller_type == "local_handoff_lqr":
        controller_settings.update({
            "local_controller_file_name": os.environ.get(
                "KRATOS_FSI_LOCAL_CONTROLLER_FILE", ""),
            "local_controller_log_file_name": str(
                output_directory / "local_lqr_timeseries.csv"),
            "local_controller_activation_time": ReadFloatEnvironmentVariable(
                "KRATOS_FSI_LOCAL_CONTROLLER_ACTIVATION_TIME", 3.0),
        })
    elif controller_type == "mpc_local_handoff":
        controller_settings.update({
            "handoff_controller_file_name": os.environ.get(
                "KRATOS_FSI_HANDOFF_CONTROLLER_FILE", ""),
            "handoff_controller_log_file_name": str(
                output_directory / "mpc_handoff_timeseries.csv"),
            "mpc_activation_time": ReadFloatEnvironmentVariable(
                "KRATOS_FSI_MPC_ACTIVATION_TIME", 8.0),
            "mpc_initial_kick_value": ReadFloatEnvironmentVariable(
                "KRATOS_FSI_MPC_INITIAL_KICK_VALUE", 0.0),
            "mpc_initial_kick_end_time": ReadFloatEnvironmentVariable(
                "KRATOS_FSI_MPC_INITIAL_KICK_END_TIME", 0.0),
        })
    else:
        raise ValueError(f'Unsupported KRATOS_FSI_CONTROLLER_TYPE "{controller_type}".')

    actuator_settings = {
        "name": "rabault_pair",
        "type": "rabault_pair",
        "theta1_degrees": 60.0,
        "theta2_degrees": 70.0,
        "width_degrees": 10.0,
        "interval": [0.0, "End"],
        **controller_settings,
    }
    actuator_process = {
        "python_module": "localized_cylinder_actuator_process",
        "Parameters": {
            "model_part_name": "FluidModelPart.NoSlip2D_Cylinder",
            "output_file_name": str(output_directory / "actuator_timeseries.csv"),
            "cylinder_center": [0.2, 0.2, 0.0],
            "actuators": [actuator_settings]
        }
    }

    process_list = project_parameters["processes"]["fluid_boundary_conditions_process_list"]
    process_list.append(actuator_process)


def WriteParaViewCollections(output_directory):
    pvpython = (
        shutil.which("pvpython")
        or "/Applications/ParaView-6.0.0-RC1.app/Contents/bin/pvpython"
    )

    if not Path(pvpython).exists():
        KratosMultiphysics.Logger.PrintWarning(
            "ParaViewOutput",
            "pvpython was not found. Legacy VTK files were written, but VTU/PVD conversion was skipped."
        )
        return

    conversion_script = Path(__file__).resolve().parent / "convert_vtk_output_to_paraview.py"
    subprocess.run([pvpython, str(conversion_script), str(output_directory)], check=True)


class BeamDisplacementCsvWriter:

    def __init__(self, model, output_path, output_interval=0.01):
        self.model = model
        self.output_interval = output_interval
        self.output_path = Path(output_path)
        self.sample_points = [
            ("x_0_30", 0.30, 0.20),
            ("x_0_40", 0.40, 0.20),
            ("x_0_50", 0.50, 0.20),
            ("tip", 0.60, 0.20)
        ]
        self.sample_nodes = []
        self.next_output_time = 0.0
        self.output_file = None

    def Initialize(self):
        structure = self.model["Structure"]
        self.sample_nodes = [
            (name, self._FindNearestNode(structure, x, y))
            for name, x, y in self.sample_points
        ]

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_path.open("w")
        self._WriteHeader()
        self.WriteCurrentStep(force=True)

    def Finalize(self):
        if self.output_file is not None:
            self.output_file.close()
            self.output_file = None

    def WriteCurrentStep(self, force=False):
        structure = self.model["Structure"]
        current_time = structure.ProcessInfo[KratosMultiphysics.TIME]

        if not force and current_time + 1e-12 < self.next_output_time:
            return

        row = [f"{current_time:.12g}"]
        for _, node in self.sample_nodes:
            displacement = node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT)
            row.extend([
                f"{displacement[0]:.12g}",
                f"{displacement[1]:.12g}",
                f"{displacement[2]:.12g}"
            ])

        self.output_file.write(",".join(row) + "\n")
        self.output_file.flush()
        self.next_output_time = current_time + self.output_interval

    def _FindNearestNode(self, model_part, x, y):
        nearest_node = None
        nearest_distance_squared = float("inf")

        for node in model_part.Nodes:
            distance_squared = (node.X0 - x) ** 2 + (node.Y0 - y) ** 2
            if distance_squared < nearest_distance_squared:
                nearest_node = node
                nearest_distance_squared = distance_squared

        return nearest_node

    def _WriteHeader(self):
        metadata = ["time"]
        header = ["time"]
        for name, node in self.sample_nodes:
            metadata.extend([
                f"{name}_node_id={node.Id}",
                f"{name}_x0={node.X0:.12g}",
                f"{name}_y0={node.Y0:.12g}"
            ])
            header.extend([
                f"{name}_DISPLACEMENT_X",
                f"{name}_DISPLACEMENT_Y",
                f"{name}_DISPLACEMENT_Z"
            ])

        self.output_file.write(",".join(metadata) + "\n")
        self.output_file.write(",".join(header) + "\n")


def CreateAnalysisStageWithFlushInstance(cls, global_model, parameters, output_directory):
    class AnalysisStageWithFlush(cls):

        def __init__(self, model, project_parameters, flush_frequency=10.0):
            super().__init__(model, project_parameters)
            self.flush_frequency = flush_frequency
            self.last_flush = time.time()
            self.beam_displacement_writer = BeamDisplacementCsvWriter(
                model,
                output_directory / "beam_displacement_timeseries.csv"
            )
            sys.stdout.flush()

        def Initialize(self):
            super().Initialize()
            self.beam_displacement_writer.Initialize()
            sys.stdout.flush()

        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()
            self.beam_displacement_writer.WriteCurrentStep()

            if self.parallel_type == "OpenMP":
                now = time.time()
                if now - self.last_flush > self.flush_frequency:
                    sys.stdout.flush()
                    self.last_flush = now

        def Finalize(self):
            self.beam_displacement_writer.Finalize()
            super().Finalize()

    return AnalysisStageWithFlush(global_model, parameters)


if __name__ == "__main__":

    with open("ProjectParameters.json", 'r') as parameter_file:
        parameter_data = json.load(parameter_file)

    output_directory = CreateRunOutputDirectory()
    end_time = os.environ.get("KRATOS_FSI_END_TIME")
    if end_time is not None:
        parameter_data["problem_data"]["end_time"] = float(end_time)

    write_paraview = ReadBooleanEnvironmentVariable("KRATOS_FSI_WRITE_PARAVIEW", True)
    if write_paraview:
        AddParaViewOutput(parameter_data, output_directory)
    AddCylinderActuatorProcess(parameter_data, output_directory)
    with (output_directory / "ProjectParameters.effective.json").open("w") as parameter_file:
        json.dump(parameter_data, parameter_file, indent=4)
    parameters = KratosMultiphysics.Parameters(json.dumps(parameter_data))

    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name = ''.join(
        x.title() for x in analysis_stage_class_name.split('_'))

    analysis_stage_module = importlib.import_module(analysis_stage_module_name)
    analysis_stage_class = getattr(
        analysis_stage_module, analysis_stage_class_name)

    global_model = KratosMultiphysics.Model()
    simulation = CreateAnalysisStageWithFlushInstance(
        analysis_stage_class, global_model, parameters, output_directory)
    simulation.Run()

    if write_paraview:
        WriteParaViewCollections(output_directory)

    print(f"Run outputs written to: {output_directory.resolve()}")
