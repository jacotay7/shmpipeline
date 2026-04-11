# shmpipeline

`shmpipeline` builds process-based compute pipelines on top of the
named shared-memory streams provided by `pyshmem`.

The package is built around three priorities:

- a small and explicit user-facing API
- a robust validation and testing story
- a clear separation between config loading, pipeline supervision, and
	kernel implementations

## Current Capabilities

The current release provides:

- YAML-driven pipeline configuration
- a pipeline manager with a simple state machine
- spawned worker processes for each configured kernel
- best-effort worker distribution across CPU slots where the OS allows it
- built-in CPU kernels
- GPU kernel parity for the built-in kernel family
- built-in elementwise arithmetic kernels
- fused custom arithmetic expressions with `cpu.custom_operation`
- per-kernel modules grouped under CPU and GPU kernel folders
- process supervision and worker error propagation
- pipeline graph introspection and runtime snapshots
- a CLI for `validate`, `describe`, and `run` workflows
- backend synthetic inputs for test and demo pipelines
- unit and integration tests for the supported behavior

## Kernel Layout

Built-in kernels are organized by backend:

- `src/shmpipeline/kernels/cpu/`
- `src/shmpipeline/kernels/gpu/`

Each concrete kernel lives in its own module and inherits from either the CPU
or GPU base class.

## Affine Example

The first concrete example lives in
`examples/affine_transformation/`.

There is also a GPU-backed variant in
`examples/gpu_affine_transformation/`.

It defines a pipeline that computes $y = A x + b$ where:

- `input_vector` is the streamed input
- `transform_matrix` and `offset_vector` are loaded separately into shared
	memory
- `output_vector` receives the transformed result

## AO Example

The repository also includes a basic adaptive-optics style example in
`examples/basic_ao_system/`.

There is also a GPU-backed variant in
`examples/gpu_basic_ao_system/`.

For a more representative high-order single-conjugate AO control loop, see
`examples/observatory_ao_system/`.

That example verifies a multi-stage CPU pipeline with:

- Shack-Hartmann centroid extraction
- gain and offset correction
- flattening into slope vectors
- affine reconstruction
- leaky-integrator control

The observatory-scale example expands that into a more RTC-like chain with:

- a pre-calibrated 256x256 Shack-Hartmann image with 32x32 subapertures
- reference slopes folded into the affine bias term for a compact process graph
- a 2048-slope to 1024-actuator synthetic reconstructor
- actuator saturation after the control integrator

To benchmark the observatory reconstructor under sustained 500 Hz WFS input,
run:

```bash
python examples/observatory_ao_system/benchmark_affine_reconstructor.py --rate-hz 500 --warmup 1 --duration 4
```

GPU kernels now mirror the built-in CPU kernels using CUDA-backed PyTorch
tensors underneath `pyshmem` GPU streams.

## Custom Operations

Simple arithmetic pipelines can be fused into a single CPU worker with
`cpu.custom_operation`.

The same expression support is available in the GPU example under
`examples/gpu_custom_operations/`.

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

## Installation Profiles

Choose the smallest install that matches how you plan to use the package.

- Base runtime and CLI from PyPI: `pip install shmpipeline`
- GPU support from PyPI: `pip install "shmpipeline[gpu]"`
- Desktop GUI from PyPI: `pip install "shmpipeline[gui]"`
- Local source install: `pip install -e .`
- Combined local development setup: `pip install -e ".[gpu,gui,test,docs]"`

## CLI

The package includes a headless CLI entry point.

Validate a config without creating shared memory:

```bash
shmpipeline validate pipeline.yaml
```

Describe the derived pipeline graph:

```bash
shmpipeline describe pipeline.yaml
shmpipeline describe pipeline.yaml --json
```

Run a pipeline until interrupted or for a bounded duration:

```bash
shmpipeline run pipeline.yaml
shmpipeline run pipeline.yaml --duration 5.0 --json-status
```

## Graph Introspection

`shmpipeline` can derive a graph model from the config before any workers are
spawned.

That graph surface reports:

- source streams that must be driven externally
- sink streams that terminate the pipeline
- orphaned shared-memory definitions
- upstream and downstream kernel dependencies
- ambiguous multiple-producer wiring as a validation error

Programmatic example:

```python
from shmpipeline import PipelineConfig, PipelineGraph

config = PipelineConfig.from_yaml("pipeline.yaml")
graph = PipelineGraph.from_config(config)

print(graph.source_streams())
print(graph.describe())
```

## Synthetic Inputs

The manager can start background synthetic writers for any built input stream.

This is useful for:

- benchmarking a pipeline without an external producer
- demoing GUI viewers and runtime behavior
- driving deterministic regression tests

Example:

```python
from shmpipeline import PipelineConfig, PipelineManager, SyntheticInputConfig

config = PipelineConfig.from_yaml("pipeline.yaml")
manager = PipelineManager(config)
manager.build()
manager.start()

manager.start_synthetic_input(
	SyntheticInputConfig(
		stream_name="input_frame",
		pattern="random",
		seed=7,
		rate_hz=500.0,
	)
)

snapshot = manager.runtime_snapshot()
print(snapshot["synthetic_sources"])

manager.stop_synthetic_input("input_frame")
manager.shutdown()
```

Available patterns:

- `constant`
- `random`
- `ramp`
- `sine`
- `impulse`
- `checkerboard`

## GUI

The package includes a desktop GUI for editing and running pipelines.

It supports:

- loading and saving YAML configs
- add, edit, and remove operations for shared memory and kernels
- config validation without building the pipeline
- a graph summary tab derived from the current document
- pipeline state-machine controls (`build`, `start`, `pause`, `resume`, `stop`, `shutdown`)
- live worker status and runtime metrics for configured kernels
- light and dark themes, with light as the default
- start and stop controls for synthetic test inputs on built streams
- live shared-memory viewers polling at about 30 Hz

Install the GUI dependencies from PyPI:

```bash
pip install "shmpipeline[gui]"
```

Launch it with:

```bash
shmpipeline-gui
```

The GUI can open live viewers for configured shared-memory streams and switch
between light and dark themes from the `View` menu.

## Runtime Health And Viewers

Runtime snapshots now report worker health in addition to raw timing metrics.

Per-worker status includes:

- `health`: one of `starting`, `waiting-input`, `active`, `idle`, `paused`, `failed`, or `stopped`
- `idle_s`: seconds since the worker last made output progress
- `last_metric_age_s`: age of the most recent worker metrics update
- rolling execution timing and throughput fields such as `avg_exec_us`, `jitter_us_rms`, and `throughput_hz`

Viewer windows run in separate spawned Python processes. Their status text now
distinguishes between:

- stream rate derived from shared-memory metadata
- viewer refresh rate derived from the local GUI timer

For GPU streams:

- CPU-only readers need `cpu_mirror: true`
- GPU viewers can fall back to direct CUDA attachment when no CPU mirror is present

## Troubleshooting

- Recent GPU lifecycle fixes removed the earlier Linux `resource_tracker`
	traceback noise caused by shared-memory unlink bookkeeping during repeated
	build/start/stop/shutdown cycles.
- If a GPU viewer should be readable from CPU code, create that stream with
	`cpu_mirror: true`.
- If a worker shows `waiting-input`, the process is alive but has not yet seen
	its first trigger input.
- If a worker shows `idle`, it has processed frames before but has not made
	recent output progress.

## Custom Kernel Extensions

Third-party kernels can now be integrated programmatically by extending the
default registry and passing the result into `PipelineManager`.

Example:

```python
import numpy as np

from shmpipeline import (
		Kernel,
		PipelineConfig,
		PipelineManager,
		get_default_registry,
)


class BiasCpuKernel(Kernel):
    kind = "example.bias"
    storage = "cpu"

    @classmethod
    def validate_config(cls, config, shared_memory):
        super().validate_config(config, shared_memory)
        if "bias" not in config.parameters:
            raise ValueError("example.bias requires a 'bias' parameter")

    def compute_into(self, trigger_input, output, auxiliary_inputs):
        output[...] = np.asarray(trigger_input) + float(
            self.context.config.parameters["bias"]
        )


config = PipelineConfig.from_yaml("pipeline.yaml")
registry = get_default_registry().extended(BiasCpuKernel)
manager = PipelineManager(config, registry=registry)
```

This extension path is currently programmatic. CLI and GUI plugin discovery is
still a manual integration story rather than an automatic entry-point system.

## Development

For source development, install the package with the test extra:

```bash
pip install -e .[test]
```

Build the documentation locally:

```bash
pip install -e .[docs]
sphinx-build -W -b html docs docs/_build/html
```

Run the test suite:

```bash
pytest
```