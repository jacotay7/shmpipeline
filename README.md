# shmpipeline

`shmpipeline` builds process-based compute pipelines on top of the
named shared-memory streams provided by `pyshmem`.

The repository is scaffolded around three priorities:

- a small and explicit user-facing API
- a robust validation and testing story
- a clear separation between config loading, pipeline supervision, and
	kernel implementations

## Current Scope

The initial scaffold provides:

- YAML-driven pipeline configuration
- a pipeline manager with a simple state machine
- spawned worker processes for each configured kernel
- best-effort worker distribution across CPU slots where the OS allows it
- built-in CPU proof-of-concept kernels
- built-in elementwise arithmetic kernels
- fused custom arithmetic expressions with `cpu.custom_operation`
- per-kernel modules grouped under CPU and GPU kernel folders
- process supervision and worker error propagation
- unit and integration tests for the scaffolded behavior

## Kernel Layout

Built-in kernels are organized by backend:

- `src/shmpipeline/kernels/cpu/`
- `src/shmpipeline/kernels/gpu/`

Each concrete kernel lives in its own module and inherits from either the CPU
or GPU base class.

## Affine Example

The first concrete example lives in
`examples/affine_transformation/`.

It defines a pipeline that computes $y = A x + b$ where:

- `input_vector` is the streamed input
- `transform_matrix` and `offset_vector` are loaded separately into shared
	memory
- `output_vector` receives the transformed result

## AO Example

The repository also includes a basic adaptive-optics style example in
`examples/basic_ao_system/`.

That example verifies a multi-stage CPU pipeline with:

- Shack-Hartmann centroid extraction
- gain and offset correction
- flattening into slope vectors
- affine reconstruction
- leaky-integrator control

GPU kernels are not implemented yet, but the configuration model and base
classes already reserve storage-specific validation for them.

## Custom Operations

Simple arithmetic pipelines can be fused into a single CPU worker with
`cpu.custom_operation`.

Example:

```yaml
shared_memory:
	- name: ao_wfs_image
		shape: [120, 120]
		dtype: float32
		storage: cpu

	- name: ao_dark_image
		shape: [120, 120]
		dtype: float32
		storage: cpu

	- name: ao_flat_image
		shape: [120, 120]
		dtype: float32
		storage: cpu

	- name: ao_clean_image
		shape: [120, 120]
		dtype: float32
		storage: cpu

kernels:
	- name: image_processing
		kind: cpu.custom_operation
		operation: (input - dark) / flat
		input: ao_wfs_image
		output: ao_clean_image
		auxiliary:
			dark: ao_dark_image
			flat: ao_flat_image
```

Supported syntax is intentionally small and safe:

- elementwise `+`, `-`, `*`, `/`
- unary `+` and `-`
- matrix multiplication with `@`
- intrinsic functions `abs(x)`, `minimum(a, b)`, `maximum(a, b)`, `min(a, b)`, `max(a, b)`, and `clip(x, low, high)`
- numeric constants
- names bound through `input` and `auxiliary`

For common two-input arithmetic there are also built-in kernels:

- `cpu.elementwise_add`
- `cpu.elementwise_subtract`
- `cpu.elementwise_multiply`
- `cpu.elementwise_divide`

## Configuration Example

```yaml
shared_memory:
	- name: input_frame
		shape: [4]
		dtype: float32
		storage: cpu

	- name: scaled_frame
		shape: [4]
		dtype: float32
		storage: cpu

kernels:
	- name: scale_stage
		kind: cpu.scale
		input: input_frame
		output: scaled_frame
		parameters:
			factor: 2.0
```

## Quick Start

```python
import numpy as np

from shmpipeline import PipelineConfig, PipelineManager

config = PipelineConfig.from_yaml("pipeline.yaml")
manager = PipelineManager(config)
manager.build()
manager.start()

manager.get_stream("input_frame").write(np.array([1, 2, 3, 4], dtype=np.float32))
result = manager.get_stream("scaled_frame").read_new(timeout=2.0)

manager.stop()
manager.shutdown()
```

## Development

Install the package and test dependencies:

```bash
pip install -e .[test]
```

Run the test suite:

```bash
pytest
```