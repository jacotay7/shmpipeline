# Build Plan

## Goals

- Build a pipeline manager on top of `pyshmem` instead of reimplementing
  shared-memory transport.
- Accept a YAML configuration file that fully describes shared-memory
  resources and compute kernels.
- Run each kernel in its own Python process with clear process lifecycle,
  state transitions, and surfaced failures.
- Separate CPU and GPU kernels cleanly while keeping one coherent config and
  manager API.
- Make testing a first-class concern from the first scaffold.

## Architecture

### 1. Configuration Layer

- Parse YAML into immutable dataclass models.
- Validate shared-memory definitions, duplicate names, kernel bindings,
  and required parameters before any process is spawned.
- Keep storage details explicit: `cpu` vs `gpu`, optional GPU device, dtype,
  and shape.

### 2. Kernel Layer

- Define base kernel abstractions with storage-specific validation hooks.
- Keep kernels pure with respect to compute: they receive arrays, produce
  arrays, and do not own process supervision.
- Split built-in kernels by storage backend and keep each concrete kernel in
  its own module.
- Use `numba` for CPU kernels where the operation benefits from it.

### 3. Runtime Layer

- Spawn one process per kernel.
- Reopen `pyshmem` streams in child processes by name.
- Use a simple loop: wait for new data on the primary input, read remaining
  inputs, compute, then write outputs.
- Surface worker lifecycle events and exceptions back to the manager through a
  multiprocessing queue.

### 4. Manager Layer

- The manager owns build/start/pause/resume/stop/shutdown transitions.
- The manager creates shared-memory resources during `build()`.
- The manager supervises worker processes, reports state, and propagates
  failures as structured errors.
- The manager assigns workers across available CPU slots on platforms that
  support affinity and falls back cleanly where they do not.
- `stop()` halts workers but preserves the built pipeline.
- `shutdown()` tears down workers and shared-memory resources.

## Testing Strategy

- Unit tests for config parsing and validation.
- Unit tests for manager state transitions.
- Integration tests for spawned worker pipelines against real `pyshmem`
  streams.
- Failure-path tests to ensure worker exceptions are visible to the caller.
- Strict pytest markers and `spawn` multiprocessing to stay aligned with the
  `pyshmem` process model.

## Proof Of Concept Scope

The initial proof of concept includes:

- `cpu.copy`
- `cpu.scale`
- `cpu.add_constant`
- `cpu.affine_transform`
- `cpu.raise_error` for supervision tests

## Next Phases

1. Add GPU kernel implementations and GPU integration tests.
2. Add richer scheduling and CPU affinity placement.
3. Add a higher-level command-line interface for launching pipelines from YAML.
4. Add pipeline graph introspection and metrics.