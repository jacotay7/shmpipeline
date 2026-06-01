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
    output: output_stream
    # auxiliary: {alias: stream_name}  # or list form
    parameters: {factor: 2.0}
    read_timeout: 1.0       # seconds
    pause_sleep: 0.01       # seconds
    poll_interval: 1.0e-5   # trigger polling interval in seconds (default 10 µs)

sources:                    # optional external data providers
  - name: my_source
    kind: my_plugin.source
    stream: input_stream
    parameters: {}
    poll_interval: 0.01

sinks:                      # optional data consumers
  - name: my_sink
    kind: my_plugin.sink
    stream: output_stream
    parameters: {}
    read_timeout: 1.0
```

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
        # For CPU kernels, output IS the locked shared-memory buffer.
        # For GPU kernels (GpuKernel subclass), output is self.output_buffer.
        output[...] = trigger_input * self.factor
```

**Note on output_buffer**: GPU kernels (`GpuKernel` subclasses) receive a private CUDA tensor (`self.output_buffer`) as the `output` argument, which is then copied to the shared CUDA IPC tensor. CPU kernels write directly into the shared-memory view and do **not** have `self.output_buffer`. Custom CPU kernels that need intermediate buffers should allocate them in `__init__` explicitly.

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
| `cpu.spot_centroid` | Single-spot centroid |
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
2. Waits for the trigger stream's write count to advance (polling at `kernel_config.poll_interval`).
3. Acquires sorted locks on all streams to read a consistent snapshot.
4. Calls `kernel.compute_into(trigger, output_view, auxiliaries)`.
5. Publishes the output (directly into the locked output_view for CPU, via `output_buffer` copy for GPU).
6. Emits rolling metrics (exec time, jitter, throughput) via a `multiprocessing.Pipe`.

**Performance note**: CPU kernels use a zero-copy fast path that writes directly into the shared-memory buffer without an intermediate `output_buffer` copy. This requires accessing pyshmem private API (`_mark_write_started`, `_finish_write`, `_array`). This coupling is intentional and documented.

**GPU note**: GPU kernels always go through a `kernel.output_buffer` (CUDA tensor) → copy to shared CUDA IPC tensor path. Safe-reads are used even inside locks because CUDA IPC memory can be stale across process boundaries.

**Lazy kernel loading**: Built-in kernels (both CPU and GPU) are loaded lazily on first use. Each worker process only imports the single kernel kind it runs, keeping startup time proportional to what's needed rather than importing everything.

**poll_interval**: The trigger polling interval is configurable per kernel via `KernelConfig.poll_interval` (default 10 µs). Increase for throughput-oriented pipelines (e.g., 1 ms) to reduce CPU burn; keep at default for lowest-latency use cases.

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
  test_config.py
  test_graph.py
  test_improvements.py     # tests for IMPROVEMENTS.md features
  test_kernels.py
  test_manager.py          # Integration tests — spawn many worker processes
  test_registry.py
  test_synthetic.py
  test_cli.py
  test_control.py
  test_observatory_example.py
  test_gui_*.py            # requires PySide6
```

## Running Tests

```bash
# Core tests (no GUI)
python -m pytest tests/ --ignore=tests/test_gui_app.py --ignore=tests/test_gui_model.py -q

# Single test module
python -m pytest tests/test_manager.py -q

# Improvement-specific tests
python -m pytest tests/test_improvements.py -q

# Integration tests can be slow (spawn worker processes)
# Run test_manager.py in isolation for reliable results
```

**Test note**: `test_manager_runs_basic_ao_pipeline_and_verifies_all_stages` and other integration tests that spawn many worker processes may fail with `MemoryError` when run as part of the full suite on memory-constrained machines (many `spawn`-based worker subprocesses exhaust page table or mmap slot limits). Run in isolation or increase system `vm.max_map_count` if this occurs.

## Package Info

- Package name on PyPI: `shmpipeline` (v1.0.1)
- License: GPL-3.0-only
- Required deps: `numba>=0.60`, `numpy>=1.26,<3`, `pyshmem>=1.0.0`, `PyYAML>=6.0`
- Optional extras: `gpu` (torch), `control` (fastapi/uvicorn/httpx), `gui` (PySide6/pyqtgraph)
- Python: 3.9–3.13
- GitHub: `https://github.com/jacotay7/shmpipeline`

## pyshmem Private API Coupling

shmpipeline intentionally accesses pyshmem internals only in `runtime.py`. The `shm_cleanup.py` module no longer accesses any pyshmem internals — it delegates to the public `pyshmem.unlink_quiet()` API.

| Private attribute / method | Location used | Reason |
|---------------------------|---------------|--------|
| `_mark_write_started()`, `_finish_write()` | `runtime.py` | Zero-copy locked write path |
| `_array` | `runtime.py` | Direct buffer view for CPU fast path |

Any changes to pyshmem internals must be coordinated with `runtime.py`.
