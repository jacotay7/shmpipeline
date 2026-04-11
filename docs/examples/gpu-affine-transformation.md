# GPU Affine Transformation

This example mirrors the CPU affine example but keeps all streams in GPU shared memory.

## Computation

```text
output_vector = transform_matrix @ input_vector + offset_vector
```

## Files

- `examples/gpu_affine_transformation/pipeline.yaml`
- `examples/gpu_affine_transformation/run_example.py`

## Run it

```bash
python examples/gpu_affine_transformation/run_example.py
```

## What it demonstrates

- GPU-backed shared-memory streams
- CUDA-backed tensor exchange between processes
- parity between CPU and GPU kernel kinds for the same topology
- numeric verification against a host-side reference

## Requirements

- installed `gpu` extra or equivalent PyTorch dependency
- compatible CUDA runtime and device

## When to use it

Use this example after the CPU affine example when you want to confirm that the GPU runtime path is working correctly on your machine.
