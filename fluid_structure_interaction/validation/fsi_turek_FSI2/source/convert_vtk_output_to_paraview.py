#!/usr/bin/env python3
"""Convert Kratos legacy VTK output folders to deformed VTU/PVD ParaView series.

Run this script with ParaView's pvpython, for example:

    /Applications/ParaView-6.0.0-RC1.app/Contents/bin/pvpython \
        convert_vtk_output_to_paraview.py run_outputs/my_run
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
from pathlib import Path

from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter


DEFAULT_OUTPUTS = {
    "fluid": ("vtk_output_fluid", "paraview_fluid", "MESH_DISPLACEMENT"),
    "structure": ("vtk_output_structure", "paraview_structure", "DISPLACEMENT"),
}


def read_time_from_name(path: Path, fallback: int) -> float:
    match = re.findall(r"_(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\.vtk$", path.name)
    return float(match[-1]) if match else float(fallback)


def deformed_copy(vtk_file: Path, deformation_variable_name: str) -> vtkUnstructuredGrid:
    reader = vtkUnstructuredGridReader()
    reader.SetFileName(str(vtk_file))
    reader.Update()

    output = vtkUnstructuredGrid()
    output.DeepCopy(reader.GetOutput())

    deformation = output.GetPointData().GetArray(deformation_variable_name)
    if deformation is None:
        return output

    points = vtkPoints()
    points.SetNumberOfPoints(output.GetNumberOfPoints())
    for i in range(output.GetNumberOfPoints()):
        point = output.GetPoint(i)
        displacement = deformation.GetTuple(i)
        points.SetPoint(
            i,
            point[0] + displacement[0],
            point[1] + displacement[1],
            point[2] + displacement[2],
        )
    output.SetPoints(points)
    return output


def write_pvd(target_path: Path, data_sets: list[tuple[float, str]]) -> None:
    rows = [
        f'    <DataSet timestep="{time:g}" group="" part="0" file="{html.escape(file_name)}"/>'
        for time, file_name in data_sets
    ]
    pvd_contents = (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
        "  <Collection>\n"
        + "\n".join(rows)
        + "\n  </Collection>\n"
        "</VTKFile>\n"
    )
    (target_path / "paraview_series.pvd").write_text(pvd_contents)


def convert_folder(
    source_path: Path,
    target_path: Path,
    deformation_variable_name: str,
    overwrite: bool,
) -> int:
    vtk_files = list(source_path.glob("*.vtk"))
    vtk_files.sort(key=lambda path: read_time_from_name(path, 0))

    if not vtk_files:
        print(f"Skipping {source_path}: no .vtk files found")
        return 0

    if target_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{target_path} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(target_path)
    target_path.mkdir(parents=True)

    data_sets = []
    for index, vtk_file in enumerate(vtk_files, start=1):
        output_time = read_time_from_name(vtk_file, index - 1)
        vtu_name = vtk_file.stem + ".vtu"

        output = deformed_copy(vtk_file, deformation_variable_name)

        writer = vtkXMLUnstructuredGridWriter()
        writer.SetFileName(str(target_path / vtu_name))
        writer.SetInputData(output)
        writer.SetDataModeToBinary()
        writer.Write()

        data_sets.append((output_time, vtu_name))
        if index == 1 or index == len(vtk_files) or index % 100 == 0:
            print(f"{target_path.name}: {index}/{len(vtk_files)}")

    write_pvd(target_path, data_sets)
    return len(data_sets)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Kratos vtk_output_fluid/vtk_output_structure folders to "
            "deformed VTU/PVD files for ParaView."
        )
    )
    parser.add_argument(
        "run_directory",
        type=Path,
        help="Run folder containing vtk_output_fluid and/or vtk_output_structure.",
    )
    parser.add_argument(
        "--only",
        choices=("both", "fluid", "structure"),
        default="both",
        help="Which output to convert. Default: both.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Replace existing paraview_* folders. Default: enabled.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail instead of replacing existing paraview_* folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    run_directory = args.run_directory.resolve()

    selected = ["fluid", "structure"] if args.only == "both" else [args.only]
    for name in selected:
        source_name, target_name, deformation_variable = DEFAULT_OUTPUTS[name]
        count = convert_folder(
            run_directory / source_name,
            run_directory / target_name,
            deformation_variable,
            args.overwrite,
        )
        if count:
            print(
                f"Wrote {count} {name} frames to "
                f"{run_directory / target_name / 'paraview_series.pvd'}"
            )


if __name__ == "__main__":
    main()
