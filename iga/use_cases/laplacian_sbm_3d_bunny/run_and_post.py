from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import KratosMultiphysics as KM
import KratosMultiphysics.IgaApplication  # noqa: F401
from KratosMultiphysics.ConvectionDiffusionApplication.convection_diffusion_analysis import (
    ConvectionDiffusionAnalysis,
)


GENERATED_OUTPUT_FILES = (
    "txt_files/Projection_Coordinates.txt",
    "surrogate_condition_nodes_coordinates.txt",
    "skin_condition_nodes_coordinates.txt",
    "knot_spans_available_before.txt",
    "knot_spans_available_after.txt",
)


def analytical_temperature(x, y, z):
    return np.sin(x) * np.sinh(y) * np.cos(z)


def _case_directory():
    return Path(__file__).resolve().parent


def _cleanup_generated_files():
    case_directory = _case_directory()
    for relative_path in GENERATED_OUTPUT_FILES:
        file_path = case_directory / relative_path
        if file_path.exists():
            file_path.unlink()


def _load_project_parameters():
    parameter_file = _case_directory() / "ProjectParameters.json"
    return KM.Parameters(parameter_file.read_text(encoding="utf-8"))


def _extract_solution_at_integration_points(model):
    model_part = model["IgaModelPart.ConvectionDiffusionDomain"]

    coordinates = []
    temperatures = []
    weights = []

    for element in model_part.Elements:
        geometry = element.GetGeometry()
        shape_functions = geometry.ShapeFunctionsValues()
        center = geometry.Center()
        weight = element.GetValue(KM.INTEGRATION_WEIGHT)

        if weight <= 0.0:
            continue

        temperature = 0.0
        for node_index, node in enumerate(geometry):
            nodal_temperature = node.GetSolutionStepValue(KM.TEMPERATURE, 0)
            temperature += shape_functions[0, node_index] * nodal_temperature

        coordinates.append((center.X, center.Y, center.Z))
        temperatures.append(temperature)
        weights.append(weight)

    return np.asarray(coordinates), np.asarray(temperatures), np.asarray(weights)


def _compute_error_metrics(coordinates, computed_temperature, weights):
    analytical_temperature_values = analytical_temperature(
        coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    )
    absolute_error = np.abs(computed_temperature - analytical_temperature_values)

    l2_error = np.sqrt(np.sum(weights * absolute_error**2))
    analytical_l2_norm = np.sqrt(np.sum(weights * analytical_temperature_values**2))
    normalized_l2_error = l2_error / analytical_l2_norm

    return absolute_error, l2_error, normalized_l2_error


def _plot_3d_error_scatter(coordinates, absolute_error):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        c=absolute_error,
        cmap="jet",
        s=10,
    )

    ax.set_title("Absolute Temperature Error")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    colorbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    colorbar.set_label("Absolute Error")

    plt.tight_layout()
    plt.show()


def main():
    _cleanup_generated_files()

    parameters = _load_project_parameters()
    model = KM.Model()
    simulation = ConvectionDiffusionAnalysis(model, parameters)
    simulation.Run()

    coordinates, computed_temperature, weights = _extract_solution_at_integration_points(model)
    absolute_error, l2_error, normalized_l2_error = _compute_error_metrics(
        coordinates, computed_temperature, weights
    )

    current_time = simulation._GetSolver().GetComputingModelPart().ProcessInfo[KM.TIME]
    print(f"Current time after simulation: {current_time}")
    print(f"L2 norm of temperature error: {l2_error}")
    print(f"Normalized L2 error: {normalized_l2_error}")

    _plot_3d_error_scatter(coordinates, absolute_error)


if __name__ == "__main__":
    main()
