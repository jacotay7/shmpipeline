# shmpipeline — Developer Reference

## Purpose

shmpipeline is a high-performance shared-memory compute pipeline framework. It lets users compose directed acyclic graphs (DAGs) of compute kernels connected by pyshmem streams, running each kernel in a dedicated OS process with CPU affinity, and provides tools (CLI, GUI, REST API) to run and monitor these pipelines. The design target is lowest possible latency for real-time data processing (e.g., adaptive-optics control systems).

## Architecture Overview

```
PipelineConfig (YAML or dict)
    ↓
PipelineManager
    ├── creates pyshmem streams (PipelineManager.build)
    ├── spawns worker processes (PipelineManager.start)
    │     each process runs: run_kernel_process() in runtime.py
    ├── manages source threads (_SourceController)
    ├── manages sink threads (_SinkController)
    └── monitors via event pipe (poll_events)
```

### Components

| Component | File | Role |
|-----------|------|------|
| `PipelineManager` | `manager.py` | Top-level orchestrator: build, start, stop, restart, shutdown |
| `PipelineConfig` | `config.py` | Immutable validated config loaded from YAML or dict |
| `PipelineGraph` | `graph.py` | DAG view, validation, serialization |
| `KernelRegistry` | `registry.py` | Plugin registry for kernel/source/sink kinds (lazy loading) |
| `Kernel` / `KernelContext` | `kernel.py` | Base class for compute kernels |
| `Source` / `SourceContext` | `source.py` | Base class for data source plugins |
| `Sink` / `SinkContext` | `sink.py` | Base class for data sink plugins |
| `run_kernel_process` | `runtime.py` | Worker process entry point |
| `SyntheticInputConfig` | `synthetic.py` | Test/demo data generators |
| `shm_cleanup` | `shm_cleanup.py` | Stream teardown via `pyshmem.unlink_quiet` |
| `scheduling.py` | | CPU affinity placement policies |
| `cli.py` | | `shmpipeline` CLI (validate, describe, run, serve, kinds, sources, sinks) |
| `control/` | | FastAPI REST + SSE control plane |
| `gui/` | | PySide6 GUI (main app + control viewer) |
| `kernels/cpu/` | | Built-in CPU kernels |
| `kernels/gpu/` | | Built-in GPU kernels (requires torch) |

## Pipeline Lifecycle

```python
manager = PipelineManager(config)  # INITIALIZED
manager.build()                    # → BUILT  (streams created)
manager.start()                    # → RUNNING (workers spawned, sources/sinks started)
manager.pause()                    # → PAUSED
manager.resume()                   # → RUNNING
manager.stop()                     # → BUILT  (workers stopped, streams remain)
manager.restart()                  # restart failed workers without full stop/start
manager.shutdown()                 # → STOPPED (streams closed/unlinked)
```

Valid state transitions are enforced; illegal transitions raise `StateTransitionError`.

## Config Format (YAML)

```yaml
shared_memory:
  - name: input_stream
    shape: [100]
    dtype: float32
    storage: cpu            # or gpu
    # gpu_device: cuda:0    # required when storage: gpu
    # cpu_mirror: true      # optional for gpu streams

kernels:
  - name: my_kernel
    kind: cpu.scale         # or gpu.scale
    input: input_stream
    output: output_stream   # single output
    # outputs: [out_a, out_b]   # multi-output form (mutually exclusive with output)
    # auxiliary: {alias: stream_name}  # or list form
    parameters: {factor: 2.0}
    read_timeout: 1.0       # seconds
    pause_sleep: 0.01       # seconds
    poll_interval: 1.0e-5   # fallback wait polling interval (default 10 µs)

sources:                    # optional external data providers
  - name: my_source
    kind: my_plugin.source
    stream: input_stream
    parameters: {}
    poll_interval: 0.01
    # read_timeout: 0.5     # optional: fail the source if read() blocks longer

sinks:                      # optional data consumers
  - name: my_sink
    kind: my_plugin.sink
    stream: output_stream
    parameters: {}
    read_timeout: 1.0
    # consume_timeout: 0.5  # optional: fail the sink if consume() blocks longer
```

A kernel uses **either** `output:` (single stream) **or** `outputs:` (an
ordered list for multi-output kernels) — never both. `output` always equals the
first of `outputs`.

## Writing a Custom Kernel

```python
from shmpipeline import Kernel, KernelContext

class MyKernel(Kernel):
    kind = "mypackage.my_kernel"   # unique kind string
    storage = "cpu"                # "cpu" or "gpu"
    auxiliary_arity = 0            # number of auxiliary streams (None = any)

    def __init__(self, context: KernelContext) -> None:
        super().__init__(context)
        self.factor = context.config.parameters.get("factor", 1.0)

    def compute_into(self, trigger_input, output, auxiliary_inputs):
        # Write result directly into the shared-memory output view — no allocation.
        # CPU and GPU kernels receive the pyshmem output view for this frame.
        output[...] = trigger_input * self.factor
```

**Note on output buffers**: the runtime passes the shared pyshmem output view
to both CPU and GPU kernels. `GpuKernel.output_buffer` remains available to
kernels that need a private staging/intermediate tensor, but the runtime no
longer copies every result through it.

### Multi-output kernels

A kernel may declare more than one output stream via `outputs: [a, b, ...]` and
set the class attribute `output_arity` (default `1`; use the matching int, or
`None` for any number). Override `compute_into_multiple` instead of
`compute_into`:

```python
class SplitKernel(Kernel):
    kind = "mypackage.split"
    storage = "cpu"
    output_arity = 2

    def compute_into_multiple(self, trigger_input, outputs, auxiliary_inputs):
        # `outputs` aligns positionally with KernelConfig.all_outputs.
        outputs[0][...] = trigger_input * 2.0
        outputs[1][...] = trigger_input + 1.0
```

The base `Kernel.compute_into_multiple` forwards `outputs[0]` to
`compute_into`, so single-output kernels need no changes. For CPU kernels each
output view is the zero-copy locked pyshmem shared-memory buffer for either
backend. GPU kernels may retain private intermediate buffers when an operation
cannot target its caller-provided output.

Register via pyproject.toml entry point:
```toml
[project.entry-points."shmpipeline.kernels"]
"mypackage.my_kernel" = "mypackage.kernels:MyKernel"
```

Or programmatically:
```python
registry = get_default_registry().extended(MyKernel)
manager = PipelineManager(config, registry=registry)
```

## Writing a Custom Source / Sink

```python
from shmpipeline import Source, SourceContext

class MySource(Source):
    kind = "mypackage.my_source"
    storage = "cpu"

    def open(self): ...          # called once before thread starts
    def read(self): return array # return payload or None to skip
    def close(self): ...         # called on thread exit
```

```python
from shmpipeline import Sink, SinkContext

class MySink(Sink):
    kind = "mypackage.my_sink"
    storage = "cpu"

    def open(self): ...
    def consume(self, value): ...   # called for each new payload
    def close(self): ...
```

Register via:
```toml
[project.entry-points."shmpipeline.sources"]
"mypackage.my_source" = "mypackage.endpoints:MySource"

[project.entry-points."shmpipeline.sinks"]
"mypackage.my_sink" = "mypackage.endpoints:MySink"
```

## Built-in Kernel Kinds

| Kind | Description |
|------|-------------|
| `cpu.copy` / `gpu.copy` | Identity copy |
| `cpu.concatenate` / `gpu.concatenate` | Synchronized multi-input concatenation (`trigger_policy: all_new`) |
| `cpu.scale` / `gpu.scale` | Multiply by scalar |
| `cpu.add_constant` / `gpu.add_constant` | Add scalar |
| `cpu.scale_offset` / `gpu.scale_offset` | `out = gain*x - offset` (1 auxiliary) |
| `cpu.elementwise_add` / `gpu.elementwise_add` | Element-wise add (1 auxiliary) |
| `cpu.elementwise_subtract` / … | Element-wise subtract |
| `cpu.elementwise_multiply` / … | Element-wise multiply |
| `cpu.elementwise_divide` / … | Element-wise divide |
| `cpu.flatten` / `gpu.flatten` | Flatten N-D array to 1-D |
| `cpu.affine_transform` / `gpu.affine_transform` | `y = Ax + b` (2 auxiliaries: matrix, offset) |
| `cpu.leaky_integrator` / `gpu.leaky_integrator` | `u_k = leak*u_{k-1} + gain*e_k` |
| `cpu.centroid` / `gpu.centroid` | Shack-Hartmann centroid (alias for `shack_hartmann_centroid`) |
| `cpu.shack_hartmann_centroid` / `gpu.shack_hartmann_centroid` | Tiled Shack-Hartmann centroid |
| `cpu.spot_centroid` / `gpu.spot_centroid` | Single-spot centroid |
| `cpu.tip_tilt_controller` / `gpu.tip_tilt_controller` | Fused spot centroid + leaky integrator + affine rotation (tip/tilt loop) |
| `gpu.tomographic_controller` | Fused batched WFS calibration + centroid + reconstruction + integration + command clip (tomographic AO loop) |
| `cpu.reduce` | Reduce input to a scalar: `operation` = sum/mean/max/min |
| `cpu.custom_operation` / `gpu.custom_operation` | Numba/torch expression evaluated at runtime |
| `cpu.raise_error` / `gpu.raise_error` | Test kernel that raises intentionally |

CPU kernels use Numba JIT (`@njit(cache=True)`) in hot paths when available; fall back to NumPy otherwise.

### cpu.reduce parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `operation` | str | `"mean"` | One of `"sum"`, `"mean"`, `"max"`, `"min"` |

The input stream can have any shape; the output stream must be a scalar (single-element). Both streams must share the same dtype.

## Worker Runtime

Each worker process (`run_kernel_process` in `runtime.py`):
1. Opens pyshmem streams for its input, auxiliaries, and output.
2. Waits for the trigger stream's level-triggered pyshmem count to advance
   (futex notification on supported streams, polling fallback otherwise).
3. Acquires sorted locks on all streams and keeps them held through borrowed
   CPU input use and output publication.
4. Calls `kernel.compute_into(trigger, output_view, auxiliaries)`.
5. Publishes outputs through pyshmem's exception-safe `write_view_locked()`
   transactions.
6. Emits rolling metrics (exec time, jitter, throughput) via a `multiprocessing.Pipe`.

**Performance note**: CPU and GPU kernels use a zero-copy path that writes
directly into the locked pyshmem buffer. Publication, mirror updates, CUDA
synchronization, and abort handling belong to pyshmem's public transaction API.

**GPU note**: safe GPU input reads remain snapshots because a file lock alone
does not establish CUDA stream synchronization. When a CUDA handle is
attached, pyshmem snapshots the device tensor even if a host mirror exists.

**Lazy kernel loading**: Built-in kernels (both CPU and GPU) are loaded lazily on first use. Each worker process only imports the single kernel kind it runs, keeping startup time proportional to what's needed rather than importing everything.

**poll_interval**: The trigger wait fallback is configurable per kernel via
`KernelConfig.poll_interval` (default 10 µs). On Linux notification-enabled
streams, normal waits park in the kernel; the interval remains useful for
non-futex platforms and dead-writer checks.

## PipelineManager Parameters

```python
PipelineManager(
    config,
    *,
    spawn_method="spawn",         # multiprocessing start method
    placement_policy=None,        # CPU affinity policy
    registry=None,                # custom KernelRegistry
    worker_start_timeout=10.0,    # seconds to wait for workers to report started
)
```

`worker_start_timeout` controls how long `start()` and `restart()` wait for each worker process to emit a `worker_started` event. Increase this (e.g., to 30.0) when running on loaded systems where Numba JIT compilation may take longer.

## restart() Method

`PipelineManager.restart()` replaces only the failed or dead worker processes without stopping the entire pipeline:

```python
# After a worker failure:
manager.state  # → FAILED

manager.restart()  # replaces failed workers, preserves streams and metrics

manager.state  # → RUNNING (if all failures resolved)
```

- Only failed or dead workers are restarted; healthy workers continue uninterrupted.
- Shared-memory streams are preserved (no `stop()`/`start()` cycle required).
- After restart, `manager._failures` is cleared for the restarted workers.
- Accepts an optional `timeout` parameter (defaults to `worker_start_timeout`).

## add_kernel() — Hot-reload

`PipelineManager.add_kernel(kernel, *, shared_memory=(), timeout=None)` spawns a
new worker for an additional kernel on a **running** (or paused) pipeline,
creating any new streams it needs, without stopping existing workers:

```python
manager.add_kernel(
    {"name": "post", "kind": "cpu.add_constant",
     "input": "out", "output": "final", "parameters": {"constant": 10.0}},
    shared_memory=[{"name": "final", "shape": [4], "dtype": "float32"}],
)
```

- Requires state `RUNNING` or `PAUSED`.
- The candidate config is fully re-validated (references, name uniqueness, graph
  producer rules, kernel arity/storage) before anything is committed.
- Only genuinely new streams are created; existing names are referenced as-is.
- On a spawn failure the partial change is rolled back (worker, new streams, and
  config) so the live pipeline is left intact.

## benchmark() Mode

`PipelineManager.benchmark(*, duration_s, source=None, output_stream=None,
warmup_s=0.5, poll_interval=1e-4)` drives a **running** pipeline and measures
throughput and terminal inter-arrival spacing at a terminal output stream:

```python
report = manager.benchmark(
    duration_s=5.0,
    source=SyntheticInputConfig(stream_name="input", pattern="random", rate_hz=1000.0),
)
# report -> {throughput_hz, frames, inter_arrival_ms: {...}, latency_ms: {...}, workers: {...}}
```

- Optionally starts a synthetic source for the run and stops it afterwards.
- `output_stream` defaults to the graph's single terminal output (specify it
  explicitly when there is more than one).
- Raises if a worker failed during the run.

## Thread Safety

`poll_events()` and `status()` are thread-safe — concurrent calls from multiple threads (e.g., a GUI polling thread and a REST endpoint) are serialised internally via a `threading.Lock`.

## CPU Affinity

Default placement policy is `RoundRobinPlacementPolicy`: kernel `i` is pinned to CPU `i % cpu_count`. Override with:

```python
from shmpipeline.scheduling import NoAffinityPlacementPolicy
manager = PipelineManager(config, placement_policy=NoAffinityPlacementPolicy())
```

## Event System

Workers emit events via `multiprocessing.Pipe`. The manager drains these on `poll_events()` (called automatically by `status()`, `start()`, `stop()`). Event types: `worker_started`, `worker_stopped`, `worker_failed`, `worker_metrics`.

## CLI

```bash
shmpipeline validate pipeline.yaml
shmpipeline describe pipeline.yaml [--json]
shmpipeline run pipeline.yaml [--duration 10] [--json-status]
shmpipeline serve pipeline.yaml [--host 0.0.0.0] [--port 8765] [--token SECRET]

# Inspect registered plugins:
shmpipeline kinds     # list all registered kernel kinds
shmpipeline sources   # list all registered source kinds
shmpipeline sinks     # list all registered sink kinds
```

## SyntheticPatternGenerator Warnings

`SyntheticPatternGenerator` emits a `UserWarning` when a pattern that produces floating-point values (``"random"``, ``"sine"``, ``"ramp"``) is used with an integer dtype. The values will be silently truncated. Use ``"constant"`` or ``"impulse"`` for integer streams, or switch to a float dtype.

## Control Plane (optional extras)

Install with `pip install shmpipeline[control]`. Exposes a FastAPI REST API and Server-Sent Events endpoint. The GUI (`shmpipeline-gui`) and `shmpipeline-control-gui` require `pip install shmpipeline[gui]`.

### Authorization scopes

`create_control_app(service, *, token=None, tokens=None)` supports three
hierarchical scopes:

- `read` — status, snapshot, graph, info, events, document reads.
- `control` — start/stop/pause/resume and synthetic I/O (implies `read`).
- `admin` — build, shutdown, and document writes (implies `control`).

Pass a single `token=` for full (`admin`) access — the original behaviour — or
`tokens={"read": ..., "control": ..., "admin": ...}` for per-scope credentials.
A request presenting a token gets that scope plus all lower ones; a missing or
unknown bearer token yields `401`, an under-privileged one yields `403`. With
neither `token` nor `tokens`, authentication is disabled.

### SSE auto-reconnect

`RemoteManagerClient.stream_events(...)` wraps `iter_events` with exponential
backoff, transparently reconnecting after a control-server restart or transient
network drop and resuming from the last seen event id via `Last-Event-ID`. The
GUI session exposes it as `RemotePipelineSession.stream_events`.

## Project Structure

```
src/shmpipeline/
  __init__.py           # lazy public exports
  cli.py                # CLI entry point
  config.py             # PipelineConfig, KernelConfig, SharedMemoryConfig, etc.
  document.py           # editable YAML document helpers (for GUI/API)
  errors.py             # exception hierarchy
  graph.py              # PipelineGraph (DAG view + validation)
  kernel.py             # Kernel ABC + KernelContext
  logging_utils.py      # ColorFormatter + get_logger
  manager.py            # PipelineManager
  registry.py           # KernelRegistry + default registry (all kernels lazy)
  runtime.py            # run_kernel_process (worker entry point)
  scheduling.py         # WorkerPlacementPolicy
  shm_cleanup.py        # quiet stream teardown via pyshmem.unlink_quiet
  sink.py               # Sink ABC + SinkContext
  source.py             # Source ABC + SourceContext
  state.py              # PipelineState enum
  synthetic.py          # SyntheticInputConfig + SyntheticSourceController
  control/              # FastAPI control-plane service
  gui/                  # PySide6 GUI application
  kernels/
    cpu/                # CPU kernel implementations
      _common.py        # Numba JIT helpers (shared across CPU kernels)
      base.py           # CpuKernel base class
      reduce.py         # ReduceCpuKernel (sum/mean/max/min)
      ...
    gpu/
      base.py           # GpuKernel base class (allocates output_buffer on GPU)
      _common.py        # torch helpers
      ...
tests/
  conftest.py              # shm_prefix fixture (auto-cleanup)
  test_config.py           # config models, multi-output, YAML line numbers
  test_graph.py
  test_document.py         # editable-document helpers
  test_runtime.py          # in-process unit tests of worker-runtime helpers
  test_shm_cleanup.py      # pyshmem.unlink_quiet teardown
  test_kernels.py          # kernel compute, cpu.reduce, multi-output ABC
  test_manager.py          # Integration tests — spawn workers; restart/benchmark/add_kernel
  test_registry.py         # registry, picklability, lazy loading
  test_synthetic.py
  test_cli.py
  test_control.py          # control plane, auth scopes, SSE reconnect
  test_observatory_example.py
  test_gui_*.py            # requires PySide6
```

## Running Tests

```bash
# Core tests (no GUI)
python -m pytest tests/ --ignore=tests/test_gui_app.py --ignore=tests/test_gui_model.py -q

# Single test module
python -m pytest tests/test_manager.py -q

# Integration tests can be slow (spawn worker processes)
# Run test_manager.py in isolation for reliable results
```

**Test isolation**: the heavy spawn-based integration tests are marked
`@pytest.mark.slow` and CI runs them in a **separate `pytest` invocation**
(`pytest -m "not slow"` for the main run, then `pytest -m slow`). Running them
in their own process keeps accumulated C-level numba/CUDA JIT state from
exhausting memory partway through the session. Locally, do the same on
memory-constrained machines:

```bash
python -m pytest -m "not slow" -q     # fast + light integration tests
python -m pytest -m slow -q           # heavy spawn-based integration tests
```

**Test note**: `test_manager_runs_basic_ao_pipeline_and_verifies_all_stages` and other integration tests that spawn many worker processes may fail with `MemoryError` when run as part of the full suite on memory-constrained machines (many `spawn`-based worker subprocesses exhaust page table or mmap slot limits). Run in isolation or increase system `vm.max_map_count` if this occurs.

## Definition of Done — coverage & lint

Every change must leave the repository green on both gates that CI enforces
(`.github/workflows/ci.yml`):

1. **Lint** — `ruff check .` and `ruff format --check .` must pass. Run
   `ruff format .` before committing.
2. **Coverage** — `pytest --cov=shmpipeline` must stay at or above the
   **80%** floor (`fail_under = 80` in `[tool.coverage.report]`). New code must
   ship with tests in the appropriate `tests/test_*.py` module (do not create
   ad-hoc "misc" test files).

The coverage gate excludes code that headless CI cannot meaningfully exercise —
GPU kernels (`kernels/gpu/*`, no CUDA on CI) and the PySide6 GUI
(`gui/*`) — configured via `omit` in `[tool.coverage.run]`. Everything else
(core library + control plane) counts toward the 80% floor. The full suite
(including GUI tests) still runs in the matrix `test` job; the dedicated
`coverage` job is what enforces the threshold.

## Package Info

- Package name on PyPI: `shmpipeline` (v1.0.4)
- License: GPL-3.0-only
- Required deps: `numba>=0.60`, `numpy>=1.26,<3`, `pyshmem>=1.1.1,<2`, `PyYAML>=6.0`
- Optional extras: `gpu` (torch), `control` (fastapi/uvicorn/httpx), `gui` (PySide6/pyqtgraph)
- Python: 3.9–3.13
- GitHub: `https://github.com/jacotay7/shmpipeline`

## pyshmem integration boundary

shmpipeline uses pyshmem's public stream API for mappings, locks, snapshots,
level-triggered waits, writable transactions, GPU IPC, mirrors, and cleanup.
No runtime code should access pyshmem private attributes. Changes to the
public transaction or wait contracts must be coordinated with the integration
tests in `tests/test_runtime.py` and the pyshmem API tests.
