# Observatory AO System

This is the most representative pipeline in the repository: a high-order single-conjugate adaptive-optics style control chain sized to feel like a real observatory workload.

## Representative dimensions

- 256x256 Shack-Hartmann detector image
- 32x32 subapertures with 8x8 pixels per tile
- 2048 measured slope values after flattening
- 1024 deformable-mirror commands after reconstruction

## Pipeline stages

1. ingest a pre-calibrated synthetic Shack-Hartmann image
2. compute per-subaperture centroids
3. flatten the slope cube into a vector
4. apply a synthetic reconstructor and folded bias term
5. run a leaky-integrator controller
6. clip actuator demand to stroke limits

## Files

- `examples/observatory_ao_system/pipeline.yaml`
- `examples/observatory_ao_system/run_example.py`
- `examples/observatory_ao_system/benchmark_affine_reconstructor.py`

## Run the example

```bash
python examples/observatory_ao_system/run_example.py
```

## Run the benchmark

```bash
python examples/observatory_ao_system/benchmark_affine_reconstructor.py --rate-hz 500 --warmup 1 --duration 4
```

## What it demonstrates

- a larger realistic graph with several high-value stages
- end-to-end verification against local reference calculations
- runtime metrics that make it possible to compare stage cost directly
- a benchmark path for the affine reconstructor under sustained input rate

## Implementation notes

- the example is CPU-based for portability
- the detector model is synthetic rather than telescope-specific
- the affine stage defaults to one OpenBLAS thread per worker to reduce latency jitter

## Useful inspection commands

```bash
shmpipeline describe examples/observatory_ao_system/pipeline.yaml
shmpipeline-gui examples/observatory_ao_system/pipeline.yaml
```

## When to use it

Use this example when you want to evaluate the package as a system rather than just as a kernel library. It is the best release smoke test for documentation, runtime state, and observability together.
