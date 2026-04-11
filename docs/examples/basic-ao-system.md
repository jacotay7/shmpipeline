# Basic AO System

This example builds a compact adaptive-optics style control chain on CPU streams.

## Pipeline stages

1. Shack-Hartmann centroid extraction
2. gain and offset correction
3. flattening into a slope vector
4. affine reconstruction
5. leaky-integrator control

## Files

- `examples/basic_ao_system/pipeline.yaml`
- `examples/basic_ao_system/run_example.py`
- `examples/basic_ao_system/run_benchmark.py`
- `examples/basic_ao_system/benchmark_120x120.yaml`

## Run the end-to-end example

```bash
python examples/basic_ao_system/run_example.py
```

## Run the benchmark-oriented workload

```bash
python examples/basic_ao_system/run_benchmark.py
```

## What it demonstrates

- a multi-stage pipeline with meaningful intermediate products
- centroid, scale-offset, flatten, affine, and control kernels working together
- stage-by-stage numerical validation
- a path from simple toy pipelines toward larger AO-style systems

## Why this example matters

This is the best midpoint between the minimal affine example and the larger observatory-scale example. It is large enough to show graph structure clearly without the size and runtime cost of the observatory workload.
