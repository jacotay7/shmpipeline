# Runtime And Observability

`shmpipeline` separates configuration loading from runtime execution. That gives you a stable preflight step before any workers or streams are created.

## Lifecycle

A `PipelineManager` moves through the following states:

- `initialized`
- `built`
- `running`
- `paused`
- `failed`
- `stopped`

The standard sequence is:

1. `build()` validates config and creates shared-memory streams.
2. `start()` spawns worker processes.
3. `pause()` and `resume()` control workers without rebuilding.
4. `stop()` ends worker processes but keeps streams available.
5. `shutdown()` closes local handles and optionally unlinks streams.

## Graph Introspection

`PipelineGraph` lets you inspect the pipeline before startup. It reports:

- source streams that must be driven externally
- sink streams that terminate the graph
- orphaned streams unused by all kernels
- upstream and downstream kernel dependencies
- ambiguous multiple-producer errors

## Runtime snapshots

`PipelineManager.status()` returns state, worker status, failures, metrics, synthetic-input status, and a summary.

`PipelineManager.runtime_snapshot()` adds:

- a timestamp
- the derived graph payload

## Worker health states

Workers are surfaced with one of these health states:

- `starting`
- `waiting-input`
- `active`
- `idle`
- `paused`
- `failed`
- `stopped`

The runtime also reports per-worker timing and throughput fields such as:

- `avg_exec_us`
- `jitter_us_rms`
- `throughput_hz`
- `frames_processed`
- `idle_s`
- `last_metric_age_s`

## Synthetic inputs

Synthetic input writers are useful for demos, regression tests, and viewer-driven debugging.

Available patterns:

- `constant`
- `random`
- `ramp`
- `sine`
- `impulse`
- `checkerboard`

Typical flow:

```python
from shmpipeline import PipelineConfig, PipelineManager, SyntheticInputConfig

config = PipelineConfig.from_yaml("pipeline.yaml")
manager = PipelineManager(config)
manager.build()
manager.start()
manager.start_synthetic_input(
    SyntheticInputConfig(stream_name="input_frame", pattern="random", rate_hz=500.0)
)
```

## Viewer behavior

Viewer windows run in separate spawned Python processes.

For GPU streams:

- enable `cpu_mirror: true` when you need CPU-side readers
- GPU viewers can still fall back to direct CUDA attachment when no CPU mirror is present

## Failure handling

Worker failures are accumulated by the manager and surfaced through:

- `failures`
- `events`
- `raise_if_failed()`

A good operational pattern is to poll `runtime_snapshot()` and stop the pipeline immediately if the manager reaches `failed`.
