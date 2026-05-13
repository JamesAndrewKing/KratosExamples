import sys
import time
import importlib
import json
from pathlib import Path
import shutil
import subprocess

import KratosMultiphysics


def AddParaViewOutput(project_parameters):
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
            "output_path": "vtk_output_structure",
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
            "output_path": "vtk_output_fluid",
            "save_output_files_in_folder": True,
            "nodal_solution_step_data_variables": ["VELOCITY", "PRESSURE", "MESH_DISPLACEMENT"],
            "nodal_data_value_variables": [],
            "element_data_value_variables": [],
            "condition_data_value_variables": []
        }
    }]


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

pairs = [tuple(item.split("=", 1)) for item in sys.argv[1:]]

for source_name, target_name in pairs:
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

        writer = vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(target_path / vtu_name))
        writer.SetInputData(reader.GetOutput())
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

    arguments = [f"{source}={target}" for source, target in output_pairs]
    subprocess.run([pvpython, "-c", conversion_script, *arguments], check=True)

def CreateAnalysisStageWithFlushInstance(cls, global_model, parameters):
    class AnalysisStageWithFlush(cls):

        def __init__(self, model, project_parameters, flush_frequency=10.0):
            super().__init__(model, project_parameters)
            self.flush_frequency = flush_frequency
            self.last_flush = time.time()
            sys.stdout.flush()

        def Initialize(self):
            super().Initialize()
            sys.stdout.flush()

        def FinalizeSolutionStep(self):
            super().FinalizeSolutionStep()

            if self.parallel_type == "OpenMP":
                now = time.time()
                if now - self.last_flush > self.flush_frequency:
                    sys.stdout.flush()
                    self.last_flush = now

    return AnalysisStageWithFlush(global_model, parameters)


if __name__ == "__main__":

    with open("ProjectParameters.json", 'r') as parameter_file:
        parameter_data = json.load(parameter_file)

    AddParaViewOutput(parameter_data)
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
        analysis_stage_class, global_model, parameters)
    simulation.Run()

    WriteParaViewCollections([
        ("vtk_output_fluid", "paraview_fluid"),
        ("vtk_output_structure", "paraview_structure")
    ])
