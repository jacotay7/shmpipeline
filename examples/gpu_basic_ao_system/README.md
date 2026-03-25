# GPU Basic AO System Example

This example builds the same adaptive-optics style control chain as the CPU AO
example, but keeps every stream in GPU shared memory:

1. A Shack-Hartmann image is split into tiled subapertures.
2. A GPU centroid kernel computes local centroid displacements.
3. A GPU scale-offset kernel applies centroid gain and subtracts calibration.
4. A GPU flatten kernel reshapes the centroid cube into a slope vector.
5. A GPU affine transform applies a reconstruction matrix and bias vector.
6. A GPU leaky-integrator kernel produces the deformable-mirror command.

The example script verifies every stage numerically on every frame by reading
the GPU streams back to the host after the final controller output is ready.

Run it with:

```bash
python examples/gpu_basic_ao_system/run_example.py
```