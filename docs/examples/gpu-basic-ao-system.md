# GPU Basic AO System

This example mirrors the CPU adaptive-optics style chain but keeps every stream in GPU shared memory.

## Pipeline stages

1. GPU centroid extraction
2. GPU gain and offset correction
3. GPU flattening into a slope vector
4. GPU affine reconstruction
5. GPU leaky-integrator control

## Files

- `examples/gpu_basic_ao_system/pipeline.yaml`
- `examples/gpu_basic_ao_system/run_example.py`

## Run it

```bash
python examples/gpu_basic_ao_system/run_example.py
```

## What it demonstrates

- parity between the CPU and GPU AO-style topologies
- reading results back to the host for verification after the final controller output
- a larger multi-stage GPU graph than the affine and custom-operation examples

## When to use it

Use this example when you want to validate GPU execution across several kernel stages rather than only a single transform.
