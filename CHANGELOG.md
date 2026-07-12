# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Declarative stream initialization.** Shared-memory entries accept an
  `initial` mapping with `constant`, seeded `normal`, explicit `values`, or
  `identity` patterns. `PipelineManager.build()` publishes initial data
  directly into the final CPU/GPU buffer before workers or sources start, and
  the GUI preserves/edits the initializer YAML.
- **1-kHz batched tomography path.** The GPU tomographic controller vectorizes
  a `(8, rows, columns)` camera cube, and the runtime supports the controller's
  opt-in locked borrowed GPU input. `pipeline_gpu_batched.yaml` removes seven
  CUDA IPC snapshots and sustained a 1-kHz controller rate on the development
  RTX 5090 while retaining 65536 slopes and 4096 actuators.

- **Synchronized multi-input kernels.** `KernelConfig` accepts `inputs: [...]`
  and `trigger_policy` (`any_new` or `all_new`); the legacy single `input` is
  normalized to a one-item trigger tuple. The worker runtime waits on the vector
  of trigger streams, re-checks all publication counts inside the sorted lock
  scope before computing, and advances the consumed counts only after a
  successful publication. Graph, document round-trip, and manager notify policy
  all understand the trigger vector.
- **`cpu.concatenate` / `gpu.concatenate`** kernels: synchronized fan-in that
  validates dtype and non-concatenated dimensions and writes directly into the
  output (`trigger_policy: all_new`, variable trigger arity).
- **`gpu.spot_centroid`** with the same contract as `cpu.spot_centroid`.
- **Fused AO controllers** `cpu.tip_tilt_controller` / `gpu.tip_tilt_controller`
  (spot centroid + leaky integrator + affine rotation) and
  `gpu.tomographic_controller` (batched single-cube or eight-stream `all_new`
  WFS calibration + centroid + reconstruction + integration + command clip).
- **Built-in endpoints** in the default registry so example and test pipelines
  run through `shmpipeline run` without an entry-point package:
  `synthetic.array` (self-paced single-stream CPU/GPU pattern source),
  `synthetic.frame_set` (coordinated multi-output camera source), and
  `null.sink` (on-device drain with optional `device_delay_s` and consume-time
  percentiles).
- **Multi-output sources.** `SourceConfig` accepts `streams: [...]` (mutually
  exclusive with `stream:`); such a source overrides `produce(writers)` and
  publishes every output stream itself, so one coordinator can drive several
  streams with controlled per-stream jitter and optional drop injection.
- **`cpu.tomographic_controller`** host-side fused controller mirroring the GPU
  kernel (eight-WFS calibration + centroids + reconstruction + control + clip),
  so the CPU and GPU example topologies are identical.
- **End-to-end latency tracing** in the tomographic benchmark
  (`run_benchmark.py --trace-latency`): stamps a frame_id per camera set,
  propagates it to the DM command, and reports input-to-sink latency
  percentiles, plus a versioned JSON report with git/version/platform/CPU/CUDA
  metadata, `generate_calibrations.py`, and `expected_dimensions.json`.
- **Frame-id propagation and `matching_frame_id` synchronization.** Kernels can
  gate a multi-input fan-in on pyshmem's `frame_id` token via
  `synchronization: {mode: matching_frame_id, max_skew_generations, max_wait_s,
  on_skew: drop_older}`, so a barrier combines only inputs from the same
  generation, drops lagging branches deterministically, and reports skew/skip/
  timeout counters. `synthetic.frame_set` stamps one token per generation across
  all cameras, and any kernel can forward the token with `propagate_frame_id:
  true`. Token handling is fully opt-in; the default worker path is unchanged.
  Requires `pyshmem>=1.2.0`.
- **`plugin_metrics()`** hook on source/sink plugins; the manager merges the
  result into each endpoint status snapshot. `_SinkController` additionally
  reports `missed_writes` (publication-count gaps) for every sink.
- **`examples/tomographic_ao_stress/`**: a synthetic three-loop tomographic AO
  stress workload with CPU and GPU pipelines, fused sustained/batched variants,
  and a multi-terminal benchmark runner (`--gpu-unbatched` exercises the
  eight-camera `all_new` barrier at full scale).

### Fixed

- OpenBLAS thread control now also probes NumPy's bundled `libopenblas` and the
  `scipy_openblas` symbol names, so `blas_threads` overrides take effect on more
  installations.

## [1.0.4] - 2026-07-11

### Performance and usability

- Plumbed `poll_interval` through pyshmem's multi-stream lock acquisition and
  require pyshmem 1.1.1 or newer.
- Added cached GPU auxiliary snapshots and reduced allocations in the worker
  hot loop.
- Added a first-class `shmpipeline benchmark` command and documented that
  benchmark spacing is terminal inter-arrival time, not end-to-end latency.
- Accepted plain mapping configurations in `PipelineManager`.

### Lifecycle and maintenance

- Added Python 3.9 to the test matrix and made test shared-memory cleanup
  discover every segment by prefix. The `gui` extra (`pyqtgraph>=0.14`,
  recent `PySide6`) requires Python >=3.10, so Python 3.9 CI installs
  `.[test]` only and skips the GUI test files; the `test`/`control` extras
  now also pull in `eval_type_backport` on Python <3.10, which pydantic
  needs to evaluate `X | None`-style annotations in `control/api.py` at
  class-definition time.
- Added contributor and security policies.
- Documented a benign PyTorch CUDA IPC teardown warning that can appear on
  GPU pipeline exit; it is not caused by shutdown ordering and does not
  indicate a leak (see the troubleshooting guide).
- Fixed the `checkerboard` synthetic input pattern: it was accepted as a
  valid `SyntheticInputConfig.pattern` value (and offered in the GUI) but
  silently fell through to an unrelated fallback instead of producing a
  checkerboard. It now renders a real tiled pattern (CPU and GPU) sized by
  `period`; an unhandled pattern name now raises instead of failing silently.

## [1.0.3] - 2026-07-11

### Correctness and pyshmem integration

- Keep worker borrowed views inside their pyshmem lock scope.
- Publish CPU and GPU outputs through pyshmem's exception-safe writable-view
  transactions rather than private sequence and storage attributes.
- Use level-triggered pyshmem waits and enable stream notifications for worker
  and sink trigger streams.
- Preserve externally attached streams on shutdown unless explicit external
  unlinking is requested.
- Require pyshmem 1.1.0 or newer and document the POSIX platform boundary.

## [1.0.2] - 2026-06-02

### Added

- **Multi-output kernels.** Kernels may declare `outputs: [...]` and implement
  `compute_into_multiple`; `output_arity` validates the expected count. CPU
  kernels keep the zero-copy fast path across all outputs; GPU kernels allocate
  one `output_buffer` per output.
- **`PipelineManager.add_kernel()`** hot-reloads a new stage (and any new
  streams) into a running pipeline without stopping existing workers, with full
  re-validation and rollback on failure.
- **`PipelineManager.benchmark()`** drives a running pipeline (optionally with a
  synthetic source) and reports throughput and p50/p90/p99 frame latency plus
  per-worker metrics.
- **Role-based control-plane authorization** with hierarchical `read` /
  `control` / `admin` scopes via `tokens=` on `create_control_app`. The single
  `token=` form still grants full access.
- **SSE auto-reconnect.** `RemoteManagerClient.stream_events()` reconnects with
  exponential backoff and resumes from the last event id via `Last-Event-ID`;
  exposed on the GUI session as `RemotePipelineSession.stream_events`.
- **Optional source/sink call timeouts.** `read_timeout` (sources) and
  `consume_timeout` (sinks) fail a plugin whose `read()`/`consume()` blocks
  longer than allowed, isolating the controller from a hung plugin.
- **YAML config error locations.** `PipelineConfig.from_yaml` now annotates
  `ConfigValidationError` messages with the source file and line number.
- **Test isolation.** Heavy `slow` integration tests run in a separate
  `pytest` invocation so accumulated C-level JIT state cannot exhaust memory
  during the main suite.

### Changed

- Coverage is now gated at 80% in CI for the core library and control plane
  (GPU kernels and the PySide6 GUI are excluded — they cannot be exercised on
  CI runners). Lint (`ruff check` / `ruff format --check`) and the coverage
  gate are documented in `CLAUDE.md` as the definition of done.
- The catch-all `test_improvements.py` was removed.

## [1.0.1]

There is now a standard and defined/supported way to add "plugins" which are sources/sinks for the pipeline.
This should like you create hardware drivers as sources and sinks for shared memory objects.

## [1.0.0]

This will be the first stable public release of `shmpipeline`.

### Added

- Validated YAML pipeline configuration for CPU and GPU shared-memory streams.
- Process-supervised pipeline execution with explicit build, start, pause,
  resume, stop, and shutdown lifecycle control.
- Built-in CPU and GPU kernel families, including arithmetic, affine,
  centroiding, flattening, and leaky-integrator operations.
- CLI workflows for `validate`, `describe`, and `run`.
- Pipeline graph introspection and validation before worker startup.
- Runtime snapshots, rolling worker metrics, synthetic inputs, and shared-memory
  viewers.
- Desktop GUI for editing, validating, running, and inspecting pipelines.
- Optional HTTP control plane for remote lifecycle commands, JSON snapshots,
  and SSE event streaming.
- Example pipelines ranging from simple affine transforms to observatory-scale
  adaptive optics flows.
- Multi-platform CI and a PyPI publish workflow.

### Changed

- Release metadata now targets the stable `1.0.0` package release.
- The README now leads with user-facing installation guidance instead of
  editable-only development commands.
- CI now validates source and wheel distributions before release publishing.

### Notes

- GPU support remains optional and depends on a compatible PyTorch and CUDA
  environment.
- Third-party kernel extension is supported programmatically through the
  registry; automatic plugin discovery remains post-`1.0.0` work.
