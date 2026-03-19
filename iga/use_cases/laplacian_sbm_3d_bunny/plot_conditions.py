from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import KratosMultiphysics as KM
import KratosMultiphysics.IgaApplication  # noqa: F401
from KratosMultiphysics.ConvectionDiffusionApplication.convection_diffusion_analysis import (
    ConvectionDiffusionAnalysis,
)


AXIS_LABELS = {
    0: "Faces parallel to YZ",
    1: "Faces parallel to XZ",
    2: "Faces parallel to XY",
}

AXIS_COLORS = {
    0: "#4a4a4a",
    1: "#8a8a8a",
    2: "#c4c4c4",
}


def _get_project_parameters_file():
    if len(sys.argv) > 2:
        raise RuntimeError(
            'Use "python3 plot_conditions.py" or "python3 plot_conditions.py CustomProjectParameters.json".'
        )
    return Path(sys.argv[1]) if len(sys.argv) == 2 else Path("ProjectParameters.json")


def _load_simulation():
    parameter_file = _get_project_parameters_file()
    with parameter_file.open("r", encoding="utf-8") as input_file:
        parameters = KM.Parameters(input_file.read())

    model = KM.Model()
    simulation = ConvectionDiffusionAnalysis(model, parameters)
    simulation.Initialize()
    return simulation, model


def _collect_surrogate_outer_faces(model_part):
    grouped_faces = {0: [], 1: [], 2: []}

    for condition in model_part.Conditions:
        geometry = condition.GetGeometry()
        face_points = [[node.X, node.Y, node.Z] for node in geometry]
        coordinates = np.asarray(face_points)

        # The surrogate faces are axis-aligned patches on the background grid.
        # The face orientation is therefore identified more robustly by the
        # coordinate with the smallest spread than by using geometry normals.
        coordinate_spreads = np.ptp(coordinates, axis=0)
        axis_normal_to_face = int(np.argmin(coordinate_spreads))
        grouped_faces[axis_normal_to_face].append(face_points)

    return grouped_faces


def _triangulate_faces(faces):
    triangles = []

    for face in faces:
        if len(face) == 3:
            triangles.append(face)
            continue

        pivot = face[0]
        for i in range(1, len(face) - 1):
            triangles.append([pivot, face[i], face[i + 1]])

    return triangles


def _set_equal_axes(ax, points):
    coords = np.asarray(points)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _plot_surrogate_outer(grouped_faces):
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    all_points = []
    legend_handles = []
    all_triangles = []
    all_colors = []

    for axis_id in (0, 1, 2):
        faces = _triangulate_faces(grouped_faces[axis_id])
        if not faces:
            continue

        all_triangles.extend(faces)
        all_colors.extend([AXIS_COLORS[axis_id]] * len(faces))
        legend_handles.append(
            Patch(facecolor=AXIS_COLORS[axis_id], edgecolor="#222222", label=AXIS_LABELS[axis_id])
        )

        for face in faces:
            all_points.extend(face)

    if not all_points:
        raise RuntimeError("No surrogate outer faces were found to plot.")

    collection = Poly3DCollection(
        all_triangles,
        facecolors=all_colors,
        edgecolors="none",
        linewidths=0.0,
        alpha=1.0,
        zsort="average",
    )
    ax.add_collection3d(collection)

    _set_equal_axes(ax, all_points)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Surrogate Boundary Outer")
    ax.legend(handles=legend_handles, loc="upper right")
    plt.tight_layout()
    plt.show()


def main():
    simulation, model = _load_simulation()

    try:
        surrogate_outer = model["IgaModelPart.surrogate_outer"]
        grouped_faces = _collect_surrogate_outer_faces(surrogate_outer)
        _plot_surrogate_outer(grouped_faces)
    finally:
        simulation.Finalize()


if __name__ == "__main__":
    main()
