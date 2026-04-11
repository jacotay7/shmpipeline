# GPU Custom Operations

This example is the GPU counterpart to the CPU custom-operations example.

## Expression

```text
(input - dark) / flat
```

## Files

- `examples/gpu_custom_operations/pipeline.yaml`
- `examples/gpu_custom_operations/run_example.py`

## Run it

```bash
python examples/gpu_custom_operations/run_example.py
```

## What it demonstrates

- GPU `custom_operation` support
- the same restricted expression language used by the CPU version
- fused arithmetic on GPU shared-memory tensors

## When to study it

Use this example when you want to keep a simple preprocessing or calibration expression on the GPU without splitting it into several workers.
