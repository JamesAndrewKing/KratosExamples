from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import KratosMultiphysics as KM
import KratosMultiphysics.IgaApplication  # noqa: F401
from KratosMultiphysics.ConvectionDiffusionApplication.convection_diffusion_analysis import (
    ConvectionDiffusionAnalysis,
)


INSERTION_COUNTS = [15, 20, 25, 30, 35, 40, 50]
POLYNOMIAL_ORDER = 2
DOMAIN_LENGTH = 2.0


def analytical_temperature(x, y, z):
    return np.sin(x) * np.sinh(y) * np.cos(z)


def _case_directory():
    return Path(__file__).resolve().parent


def _load_project_parameters_template():
    parameter_file = _case_directory() / "ProjectParameters.json"
    return parameter_file.read_text(encoding="utf-8")


def _build_parameters(parameters_template, insertion_count):
    parameters = KM.Parameters(parameters_template)

    for modeler_index in range(parameters["modelers"].size()):
        modeler = parameters["modelers"][modeler_index]
        if modeler["modeler_name"].GetString() != "NurbsGeometryModelerSbm":
            continue

        modeler_parameters = modeler["Parameters"]
        modeler_parameters["number_of_knot_spans"] = KM.Parameters(
            f"[{insertion_count}, {insertion_count}, {insertion_count}]"
        )
        modeler_parameters["polynomial_order"] = KM.Parameters(
            f"[{POLYNOMIAL_ORDER}, {POLYNOMIAL_ORDER}, {POLYNOMIAL_ORDER}]"
        )
        break

    return parameters


def _compute_normalized_l2_error(model):
    model_part = model["IgaModelPart.ConvectionDiffusionDomain"]

    error_norm_squared = 0.0
    exact_norm_squared = 0.0

    for element in model_part.Elements:
        geometry = element.GetGeometry()
        shape_functions = geometry.ShapeFunctionsValues()
        center = geometry.Center()
        weight = element.GetValue(KM.INTEGRATION_WEIGHT)

        if weight <= 0.0:
            continue

        computed_temperature = 0.0
        for node_index, node in enumerate(geometry):
            nodal_temperature = node.GetSolutionStepValue(KM.TEMPERATURE, 0)
            computed_temperature += shape_functions[0, node_index] * nodal_temperature

        exact_temperature = analytical_temperature(center.X, center.Y, center.Z)
        error_norm_squared += weight * (computed_temperature - exact_temperature) ** 2
        exact_norm_squared += weight * exact_temperature**2

    return np.sqrt(error_norm_squared / exact_norm_squared)


def _plot_convergence(h_values, normalized_l2_errors):
    plt.figure(figsize=(8, 6))
    plt.loglog(h_values, normalized_l2_errors, "s-", markersize=4, linewidth=2, label="Normalized L2 error")

    reference_order = POLYNOMIAL_ORDER + 1
    reference_curve = normalized_l2_errors[0] * (h_values / h_values[0]) ** reference_order
    plt.loglog(
        h_values,
        reference_curve,
        "--",
        linewidth=1.5,
        label=f"Reference slope h^{reference_order}",
    )

    plt.grid(True, which="both", linestyle="--")
    plt.xlabel("h")
    plt.ylabel("Normalized L2 error")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def main():
    parameters_template = _load_project_parameters_template()

    h_values = []
    normalized_l2_errors = []

    for insertion_count in INSERTION_COUNTS:
        print(f"insertions: {insertion_count}")

        parameters = _build_parameters(parameters_template, insertion_count)
        model = KM.Model()
        simulation = ConvectionDiffusionAnalysis(model, parameters)
        simulation.Run()

        normalized_l2_error = _compute_normalized_l2_error(model)
        h_value = DOMAIN_LENGTH / insertion_count

        h_values.append(h_value)
        normalized_l2_errors.append(normalized_l2_error)

        print(f"Normalized L2 error: {normalized_l2_error}")
        simulation.Clear()

    print(f"\nh = {h_values}")
    print(f"\nNormalized L2 error = {normalized_l2_errors}")

    if len(h_values) > 1:
        slopes = []
        for index in range(len(h_values) - 1):
            slope = np.log(normalized_l2_errors[index + 1] / normalized_l2_errors[index]) / np.log(
                h_values[index + 1] / h_values[index]
            )
            slopes.append(slope)
            print(f"slope {index + 1}: {slope}")

        print(f"average slope: {np.mean(slopes)}")

    _plot_convergence(np.asarray(h_values), np.asarray(normalized_l2_errors))


if __name__ == "__main__":
    main()
