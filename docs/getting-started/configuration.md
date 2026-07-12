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
- `notify`: optional boolean override for pyshmem waitable notifications;
  when omitted, shmpipeline enables notifications for streams consumed by
  workers or sinks where supported
- `mode`: stream lifecycle policy: `create_or_attach` (default), `create`,
  `attach`, or `replace`

`create_or_attach` reuses a compatible stream and replaces an incompatible
one. `attach` never creates or replaces a stream, `create` fails when the name
already exists, and `replace` explicitly recreates an existing stream.
Streams attached from outside the manager are closed but not unlinked during
shutdown unless `shutdown(unlink_external=True)` is requested.

## Kernel definitions

Most kernels consume one trigger input stream and write one output stream:

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
- `inputs`: ordered trigger stream names for a multi-input kernel; mutually
  exclusive with `input`
- `trigger_policy`: `any_new` (single-input default) or `all_new` (multi-input
  default)
- `output`: output stream name
- `auxiliary`: optional extra stream bindings
- `operation`: expression string for `custom_operation` kernels
- `parameters`: kind-specific parameter mapping
- `read_timeout`: trigger read timeout in seconds
- `pause_sleep`: worker pause polling interval in seconds

## Synchronized multi-input kernels

Multi-input kernels can wait until every dynamic input has published a new
value. `cpu.concatenate` and `gpu.concatenate` require this mode:

```yaml
- name: synchronized_join
  kind: cpu.concatenate
  inputs: [wfs0_slopes, wfs1_slopes, wfs2_slopes]
  trigger_policy: all_new
  output: combined_slopes
  parameters:
    axis: 0
```

The worker tracks a consumed publication count per trigger, waits until all of
them advance, acquires all input/output locks in sorted order, and rechecks the
counts before computing. This guarantees that every concatenation uses a new
value from every input. Publication counts do not by themselves prove that the
values share an external hardware frame ID if upstream stages dropped different
frames; applications needing that stronger guarantee use the `frame_id` token
described next.

## Frame-id tokens and matching_frame_id

pyshmem publishes an optional uint64 `frame_id` token atomically with each
write. shmpipeline uses it to establish cross-branch frame identity — a
coordinated source (`synthetic.frame_set`) stamps the same token on every
camera of one hardware generation, and a fan-in only combines inputs carrying
the same token:

```yaml
- name: synchronized_join
  kind: cpu.concatenate
  inputs: [wfs0_slopes, wfs1_slopes, wfs2_slopes]
  trigger_policy: all_new
  output: combined_slopes
  synchronization:
    mode: matching_frame_id     # combine only equal frame_id tokens
    max_skew_generations: 16    # give up once branches skew this far
    max_wait_s: 0.010           # ...or after this long (alias: timeout)
    on_skew: drop_older         # advance past lagging generations
  parameters:
    axis: 0
```

Under `matching_frame_id` the worker reads all trigger tokens in the same lock
scope that snapshots the payloads, and:

- computes only when every trigger carries the same token, propagating that
  token to the output;
- when branches skew, applies `on_skew: drop_older` — it advances past the
  lagging branches and waits for their next publication, bounded by
  `max_skew_generations` and `max_wait_s` so a dead branch cannot stall the
  loop forever;
- reports `frame_sync_skew_events`, `frame_sync_skipped_generations`, and
  `frame_sync_timeouts` in the worker metrics.

Any single-input kernel can carry the token forward with `propagate_frame_id:
true`, which copies its trigger's token to every output in the same locked
publication. Token handling is entirely opt-in — a kernel with neither
`matching_frame_id` nor `propagate_frame_id` never touches the token metadata,
so the default hot path is unchanged.

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
- a kernel cannot repeat a stream in `inputs`
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
