# Use Case IGA - 3D Laplacian on the Stanford Bunny with SBM

Short description

This example solves a stationary 3D Laplacian problem with the Surrogate Boundary Method (SBM) in the IGA Application. The embedded boundary is the Stanford bunny imported from `bunny.mdpa`, while the analysis is carried out on a structured background NURBS volume.

The manufactured solution used for validation is:

`T(x, y, z) = sin(x) * sinh(y) * cos(z)`

The same field is prescribed on the surrogate boundary, and the corresponding volumetric term is assigned through `HEAT_FLUX`.

Files
- `ProjectParameters.json`
- `materials.json`
- `bunny.mdpa`
- `run_and_post.py`
- `convergence.py`
- `plot_conditions.py`

Geometry
- `import_mdpa_modeler` loads the Stanford bunny skin into `initial_skin_model_part_out`.
- `NurbsGeometryModelerSbm` creates the background volume on `[0, 2]^3` with order `[2, 2, 2]` and a structured knot-span grid.
- `IgaModelerSbm` creates:
  - `LaplacianElement` on `IgaModelPart.ConvectionDiffusionDomain`
  - `SbmLaplacianConditionDirichlet` on `IgaModelPart.SBM_Support_outer`

Solver
- Stationary convection-diffusion analysis
- 3D Laplacian formulation
- SBM Dirichlet condition with `PENALTY_FACTOR = -1`
- Linear solver: `skyline_lu_factorization`

Run

```bash
python3 run_and_post.py
python3 convergence.py
python3 plot_conditions.py
```

`run_and_post.py` runs the analysis, prints the absolute and normalized L2 errors, and plots the error distribution at the element integration points.

`convergence.py` performs a mesh-refinement study by changing the number of knot spans of the background NURBS volume.

`plot_conditions.py` initializes the model and plots the outer surrogate boundary in 3D using three gray tones according to the face orientation.
