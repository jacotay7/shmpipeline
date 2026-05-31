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
| `PipelineManager` | `manager.py` | Top-level orchestrator: build, start, stop, shutdown |
| `PipelineConfig` | `config.py` | Immutable validated config loaded from YAML or dict |
| `PipelineGraph` | `graph.py` | DAG view, validation, serialization |
| `KernelRegistry` | `registry.py` | Plugin registry for kernel/source/sink kinds |
| `Kernel` / `KernelContext` | `kernel.py` | Base class for compute kernels |
| `Source` / `SourceContext` | `source.py` | Base class for data source plugins |
| `Sink` / `SinkContext` | `sink.py` | Base class for data sink plugins |
| `run_kernel_process` | `runtime.py` | Worker process entry point |
| `SyntheticInputConfig` | `synthetic.py` | Test/demo data generators |
| `shm_cleanup` | `shm_cleanup.py` | Noise-free POSIX segment teardown |
| `scheduling.py` | | CPU affinity placement policies |
| `cli.py` | | `shmpipeline` CLI (validate, describe, run, serve) |
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
        # self.output_buffer is pre-allocated numpy array (CPU) or torch tensor (GPU)
        self.factor = context.config.parameters.get("factor", 1.0)

    def compute_into(self, trigger_input, output, auxiliary_inputs):
        # Write result directly into output buffer — no allocation
        output[...] = trigger_input * self.factor
```

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
| `cpu.custom_operation` / `gpu.custom_operation` | Numba/torch expression evaluated at runtime |
| `cpu.raise_error` / `gpu.raise_error` | Test kernel that raises intentionally |

CPU kernels use Numba JIT (`@njit(cache=True)`) in hot paths when available; fall back to NumPy otherwise.

## Worker Runtime

Each worker process (`run_kernel_process` in `runtime.py`):
1. Opens pyshmem streams for its input, auxiliaries, and output.
2. Waits for the trigger stream's write count to advance.
3. Acquires sorted locks on all streams to read a consistent snapshot.
4. Calls `kernel.compute_into(trigger, output_view, auxiliaries)`.
5. Publishes the output (directly into the locked output_view for CPU, via `output_buffer` copy for GPU).
6. Emits rolling metrics (exec time, jitter, throughput) via a `multiprocessing.Pipe`.

**Performance note**: CPU kernels use a zero-copy fast path that writes directly into the shared-memory buffer without an intermediate `output_buffer` copy. This requires accessing pyshmem private API (`_mark_write_started`, `_finish_write`, `_array`). This coupling is intentional and documented.

**GPU note**: GPU kernels always go through a `kernel.output_buffer` (CUDA tensor) → copy to shared CUDA IPC tensor path. Safe-reads are used even inside locks because CUDA IPC memory can be stale across process boundaries.

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
```

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
  registry.py           # KernelRegistry + default registry
  runtime.py            # run_kernel_process (worker entry point)
  scheduling.py         # WorkerPlacementPolicy
  shm_cleanup.py        # quiet POSIX unlink helpers
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
      ...
    gpu/
      base.py           # GpuKernel base class (allocates output_buffer on GPU)
      _common.py        # torch helpers
      ...
tests/
  conftest.py           # shm_prefix fixture (auto-cleanup)
  test_config.py
  test_graph.py
  test_kernels.py
  test_manager.py       # Integration tests — spawn many worker processes
  test_registry.py
  test_synthetic.py
  test_cli.py
  test_control.py
  test_observatory_example.py
  test_gui_*.py         # requires PySide6
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

**Test note**: `test_manager_runs_basic_ao_pipeline_and_verifies_all_stages` and other integration tests that spawn many worker processes may fail with `MemoryError` when run as part of the full suite on memory-constrained machines (many `spawn`-based worker subprocesses exhaust page table or mmap slot limits). Run in isolation or increase system `vm.max_map_count` if this occurs.

## Package Info

- Package name on PyPI: `shmpipeline` (v1.0.1)
- License: GPL-3.0-only
- Required deps: `numba>=0.60`, `numpy>=1.26,<3`, `pyshmem>=1.0.0`, `PyYAML>=6.0`
- Optional extras: `gpu` (torch), `control` (fastapi/uvicorn/httpx), `gui` (PySide6/pyqtgraph)
- Python: 3.9–3.13
- GitHub: `https://github.com/jacotay7/shmpipeline`

## pyshmem Private API Coupling

shmpipeline intentionally accesses pyshmem internals:

| Private attribute / method | Location used | Reason |
|---------------------------|---------------|--------|
| `_mark_write_started()`, `_finish_write()` | `runtime.py` | Zero-copy locked write path |
| `_array` | `runtime.py` | Direct buffer view for CPU fast path |
| `_data_shm._name`, `_metadata_shm._name`, `_gpu_handle_shm._name` | `shm_cleanup.py` | Direct POSIX unlink without resource_tracker |
| `_LOCAL_GPU_TENSORS` | `shm_cleanup.py` | Clear cached GPU tensor on unlink |
| `_data_name()`, `_metadata_name()`, `_gpu_handle_name()`, `_lock_path()` | `shm_cleanup.py` | Reconstruct segment names by stream name |

Any changes to pyshmem internals must be coordinated with these two files.
