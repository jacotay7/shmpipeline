# Observatory AO System Example

This example models a high-order single-conjugate adaptive-optics real-time
control chain sized to feel like something you might see at an 8-10 meter
class observatory.

Representative dimensions:

- 256x256 Shack-Hartmann detector image
- 32x32 subapertures with 8x8 pixels per tile
- 2048 measured slope values after flattening
- 1024 deformable-mirror commands after reconstruction

Pipeline stages:

1. Ingest a pre-calibrated Shack-Hartmann image.
2. Compute per-subaperture Shack-Hartmann centroids.
3. Flatten the slope cube into a 2048-element vector.
4. Apply a synthetic control matrix and a bias term that already folds in
   static reference slopes.
5. Run a leaky-integrator controller.
6. Clip the integrated actuator demand to per-actuator stroke limits.

The example is still intentionally simplified:

- it uses a square fully illuminated subaperture grid instead of a masked pupil
- the reconstructor and bias vectors are synthetic, not telescope-specific
- the detector image is generated as an already calibrated synthetic WFS frame
- on Linux it uses `fork` worker startup to keep the larger demo runnable on
  tighter-memory machines
- it runs on CPU kernels for portability, although the same topology can be
  adapted to GPU kernels

Run it with:

```bash
python examples/observatory_ao_system/run_example.py
```

The script synthesizes moving Shack-Hartmann spot images, loads the static
reconstruction and actuator-limit streams, verifies every stage numerically,
and reports achieved frame rate plus how often actuator saturation occurred.

For sustained reconstructor timing on the same observatory workload at a
target 500 Hz input rate, run:

```bash
python examples/observatory_ao_system/benchmark_affine_reconstructor.py --rate-hz 500 --warmup 1 --duration 4
```

That benchmark precomputes a bank of the same 256x256 synthetic WFS frames,
drives the full observatory pipeline at a stable cadence, and reports the
worker-runtime `avg_exec_us` values for each stage so you can compare the
affine reconstructor directly against the rest of the loop.

The CPU affine kernel now caps OpenBLAS to one thread per worker by default to
reduce latency jitter in process-per-kernel pipelines. If you want to override
that for a specific pipeline, set `parameters.blas_threads` on the
`cpu.affine_transform` stage.

You can also inspect the config directly with the CLI or GUI:

```bash
shmpipeline describe examples/observatory_ao_system/pipeline.yaml
shmpipeline-gui examples/observatory_ao_system/pipeline.yaml
```