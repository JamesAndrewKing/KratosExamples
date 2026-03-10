import sys
import time
import importlib
import numpy as np
import matplotlib.pyplot as plt

import KratosMultiphysics


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
        parameters = KratosMultiphysics.Parameters(parameter_file.read())

    analysis_stage_module_name = parameters["analysis_stage"].GetString()
    analysis_stage_class_name = analysis_stage_module_name.split('.')[-1]
    analysis_stage_class_name = ''.join(x.title() for x in analysis_stage_class_name.split('_'))

    analysis_stage_module = importlib.import_module(analysis_stage_module_name)
    analysis_stage_class = getattr(analysis_stage_module, analysis_stage_class_name)

    # Create model and run simulation
    global_model = KratosMultiphysics.Model()
    simulation = CreateAnalysisStageWithFlushInstance(analysis_stage_class, global_model, parameters)
    simulation.Run()

    # Extract Cp
    modelpart = global_model["FluidModelPart.Body2D_Body"]

    upper_x = []
    upper_cp = []
    lower_x = []
    lower_cp = []

    for node in modelpart.Nodes:
        x = node.X0
        y = node.Y0
        cp = node.GetValue(KratosMultiphysics.PRESSURE_COEFFICIENT)

        if y >= 0.0:
            upper_x.append(x)
            upper_cp.append(cp)
        else:
            lower_x.append(x)
            lower_cp.append(cp)

    # convert to numpy
    upper_x = np.array(upper_x)
    upper_cp = np.array(upper_cp)
    lower_x = np.array(lower_x)
    lower_cp = np.array(lower_cp)

    # sort along chord
    upper_order = np.argsort(upper_x)
    lower_order = np.argsort(lower_x)

    upper_x = upper_x[upper_order]
    upper_cp = upper_cp[upper_order]

    lower_x = lower_x[lower_order]
    lower_cp = lower_cp[lower_order]

    # Plot
    fig, ax = plt.subplots()
    fig.set_figwidth(8)
    fig.set_figheight(6)

    ax.plot(upper_x, -upper_cp, "o", label="Upper surface")
    ax.plot(lower_x, -lower_cp, "o", label="Lower surface")

    ax.set_xlabel("x/c")
    ax.set_ylabel("-Cp")
    ax.set_title("NACA0012 Potential Flow")
    ax.grid()

    # aerodynamic convention
    ax.invert_yaxis()

    ax.legend()
    plt.tight_layout()
    plt.savefig("Cp_distribution.png", dpi=400)
    plt.show()