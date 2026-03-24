# Basic AO System Example

This example builds a simple adaptive-optics style control chain:

1. A Shack-Hartmann image is split into tiled subapertures.
2. A centroid kernel computes local centroid displacements for each tile.
3. A scale-offset kernel applies centroid gain and subtracts a calibration offset.
4. A flatten kernel reshapes the centroid cube into a slope vector.
5. An affine transform applies a reconstruction matrix and bias vector.
6. A leaky-integrator kernel produces the deformable-mirror command.

The example script verifies every stage numerically on every frame by reading
each intermediate shared-memory stream after the final controller output is
available.

Run it with:

```bash
python examples/basic_ao_system/run_example.py
```