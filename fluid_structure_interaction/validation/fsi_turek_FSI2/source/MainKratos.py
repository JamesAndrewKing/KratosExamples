import sys
import time
import importlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime

import KratosMultiphysics


def CreateRunOutputDirectory():
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_directory = Path("run_outputs") / run_name
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def AddParaViewOutput(project_parameters, output_directory):
    project_parameters["output_processes"]["vtk_output"] = [{
        "python_module": "vtk_output_process",
        "kratos_module": "KratosMultiphysics",
        "process_name": "VtkOutputProcess",
        "help": "This process writes postprocessing files for ParaView",
        "Parameters": {
            "model_part_name": "Structure",
            "output_control_type": "time",
            "output_interval": 0.01,
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
            "output_interval": 0.01,
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
    actuator_process = {
        "python_module": "localized_cylinder_actuator_process",
        "Parameters": {
            "model_part_name": "FluidModelPart.NoSlip2D_Cylinder",
            "output_file_name": str(output_directory / "actuator_timeseries.csv"),
            "cylinder_center": [0.2, 0.2, 0.0],
            "actuators": [{
                "name": "rabault_pair",
                "type": "rabault_pair",
                "theta1_degrees": 60.0,
                "theta2_degrees": 70.0,
                "width_degrees": 10.0,
                "controller_type": "sinusoidal",
                "amplitude": 0.02,
                "frequency": 1.0,
                "phase": 0.0,
                "offset": 0.0,
                "interval": [0.0, "End"]
            }]
        }
    }

    process_list = project_parameters["processes"]["fluid_boundary_conditions_process_list"]
    process_list.append(actuator_process)


def WriteParaViewCollections(output_pairs):
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

    conversion_script = r"""
from pathlib import Path
import html
import re
import shutil
import sys

from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid


def ParseOutputPair(item):
    source_name, rest = item.split("=", 1)
    target_name, deformation_variable_name = rest.split(":", 1)
    return source_name, target_name, deformation_variable_name


pairs = [ParseOutputPair(item) for item in sys.argv[1:]]

for source_name, target_name, deformation_variable_name in pairs:
    source_path = Path(source_name)
    target_path = Path(target_name)

    if not source_path.exists():
        continue

    vtk_files = sorted(source_path.glob("*.vtk"))
    if not vtk_files:
        continue

    if target_path.exists():
        shutil.rmtree(target_path)
    target_path.mkdir(parents=True)

    data_sets = []
    for vtk_file in vtk_files:
        time_match = re.findall(r"_(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\.vtk$", vtk_file.name)
        output_time = float(time_match[-1]) if time_match else float(len(data_sets))
        vtu_name = vtk_file.stem + ".vtu"

        reader = vtkUnstructuredGridReader()
        reader.SetFileName(str(vtk_file))
        reader.Update()

        output = vtkUnstructuredGrid()
        output.DeepCopy(reader.GetOutput())
        deformation = output.GetPointData().GetArray(deformation_variable_name)
        if deformation is not None:
            points = vtkPoints()
            points.SetNumberOfPoints(output.GetNumberOfPoints())
            for i in range(output.GetNumberOfPoints()):
                point = output.GetPoint(i)
                displacement = deformation.GetTuple(i)
                points.SetPoint(
                    i,
                    point[0] + displacement[0],
                    point[1] + displacement[1],
                    point[2] + displacement[2]
                )
            output.SetPoints(points)

        writer = vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(target_path / vtu_name))
        writer.SetInputData(output)
        writer.SetDataModeToBinary()
        writer.Write()

        data_sets.append((output_time, vtu_name))

    rows = [
        f'    <DataSet timestep="{time:g}" group="" part="0" file="{html.escape(file_name)}"/>'
        for time, file_name in data_sets
    ]
    pvd_contents = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        '  <Collection>\n'
        + "\n".join(rows)
        + '\n  </Collection>\n'
        '</VTKFile>\n'
    )
    (target_path / "paraview_series.pvd").write_text(pvd_contents)
"""

    arguments = [
        f"{source}={target}:{deformation_variable}"
        for source, target, deformation_variable in output_pairs
    ]
    subprocess.run([pvpython, "-c", conversion_script, *arguments], check=True)


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

    WriteParaViewCollections([
        (
            output_directory / "vtk_output_fluid",
            output_directory / "paraview_fluid",
            "MESH_DISPLACEMENT"
        ),
        (
            output_directory / "vtk_output_structure",
            output_directory / "paraview_structure",
            "DISPLACEMENT"
        )
    ])

    print(f"Run outputs written to: {output_directory.resolve()}")
