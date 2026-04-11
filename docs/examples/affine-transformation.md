# Affine Transformation

This is the smallest complete pipeline in the repository: one worker stage that computes

```text
output_vector = transform_matrix @ input_vector + offset_vector
```

## Why this example matters

Use this example when you want the shortest path to understanding:

- stream definitions
- one-stage kernel wiring
- loading static parameter buffers through shared memory
- validating outputs numerically

## Files

- `examples/affine_transformation/pipeline.yaml`
- `examples/affine_transformation/run_example.py`

## Run it

```bash
python examples/affine_transformation/run_example.py
```

## What it demonstrates

- a CPU shared-memory pipeline
- one trigger input stream
- auxiliary streams for matrix and bias terms
- deterministic end-to-end verification against a local reference computation

## Useful follow-up commands

```bash
shmpipeline validate examples/affine_transformation/pipeline.yaml
shmpipeline describe examples/affine_transformation/pipeline.yaml
shmpipeline run examples/affine_transformation/pipeline.yaml --duration 1.0
```

## When to study this first

Start here if you are new to the package or if you are verifying a fresh install.
