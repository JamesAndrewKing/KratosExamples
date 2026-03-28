# Scordelis-Lo Roof - Single_Patch - Linear Analysis

**Author:** Aakash Ravichandran

**Kratos version:** 10.4

**Source files:** [Scordelis-Lo Roof - Single_Patch - Linear Analysis](https://github.com/KratosMultiphysics/Examples/tree/master/iga/validation/scordelis_lo_roof_single_patch_linear/source)

## Problem definition

This example presents the validation of Scordelis-Lo Roof with Shell3pElement in geometric linear analysis [1]. 

![Reference Model](data/Reference_Model.png)

*Structural System [1]*

Load = 90 per unit

The roof is modeled using single IGA patch with the Shell3pElement. The CAD model is constructed with single span b-splines of curve degree 2 in both axis of the plate. Additional refinement is applied in Kratos, by increasing the curve degree to 4 and inserting 6 knots in each direction of the plate. 

## Results

The displacement at point A is obtained as -0.3068 units, which is in agreement with the reference value of -0.3024 units.

## References

1. Josef M. Kiendl, *Isogeometric Analysis and Shape Optimal Design of Shell Structures*, PhD Dissertation, pp. 51–52. [Link](https://mediatum.ub.tum.de/doc/1002634/464162.pdf)