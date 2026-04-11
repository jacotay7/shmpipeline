# Configuration Model

A pipeline configuration has two top-level sections:

- `shared_memory`: named stream definitions
- `kernels`: processing stages wired between those streams

## Shared memory definitions

Each shared-memory record defines a named stream:

```yaml
- name: input_frame
  shape: [120, 120]
  dtype: float32
  storage: cpu
```

Supported fields:

- `name`: unique stream name
- `shape`: one or more positive dimensions
- `dtype`: NumPy-compatible dtype string
- `storage`: `cpu` or `gpu`
- `gpu_device`: required for GPU streams
- `cpu_mirror`: optional for GPU streams that need CPU-side readers such as viewers or external tooling

## Kernel definitions

Each kernel consumes one trigger input stream and writes one output stream:

```yaml
- name: scale_stage
  kind: cpu.scale
  input: input_frame
  output: scaled_frame
  parameters:
    factor: 2.0
```

Supported fields:

- `name`: unique kernel name
- `kind`: registered kernel kind such as `cpu.scale` or `gpu.affine_transform`
- `input`: trigger input stream name
- `output`: output stream name
- `auxiliary`: optional extra stream bindings
- `operation`: expression string for `custom_operation` kernels
- `parameters`: kind-specific parameter mapping
- `read_timeout`: trigger read timeout in seconds
- `pause_sleep`: worker pause polling interval in seconds

## Auxiliary inputs

Auxiliary inputs can be declared as a mapping when aliases matter:

```yaml
auxiliary:
  matrix: affine_transform_matrix
  bias: affine_offset_vector
```

This is especially useful for `custom_operation` kernels and affine-style stages that read several non-trigger streams.

## Graph validation rules

The loader and graph layer validate several important invariants before workers start:

- shared-memory names must be unique
- kernel names must be unique
- every referenced stream must exist
- a kernel cannot reuse the same stream as both input and output
- a kernel cannot reuse the same auxiliary binding multiple times
- a stream cannot have more than one producer kernel

## CPU and GPU pipelines

The same pipeline topology can often be expressed for either CPU or GPU streams by switching stream storage and kernel kinds together.

CPU example:

```yaml
storage: cpu
kind: cpu.scale
```

GPU example:

```yaml
storage: gpu
gpu_device: cuda:0
kind: gpu.scale
```

## Custom operations

`cpu.custom_operation` and `gpu.custom_operation` accept a restricted expression language for fused elementwise workflows.

Supported syntax:

- elementwise `+`, `-`, `*`, `/`
- unary `+` and `-`
- matrix multiplication with `@`
- `abs`, `minimum`, `maximum`, `min`, `max`, and `clip`
- numeric constants
- names bound through `input` and `auxiliary`

## Where to look next

- See [CLI guide](../guides/cli.md) for validation and run flows.
- See [runtime guide](../guides/runtime.md) for health states and snapshots.
- See [API reference](../reference/index.md) for the configuration classes.
