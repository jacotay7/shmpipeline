# Changelog

All notable changes to this project will be documented in this file.

## [1.0.3]

### Correctness and pyshmem integration

- Keep worker borrowed views inside their pyshmem lock scope.
- Publish CPU and GPU outputs through pyshmem's exception-safe writable-view
  transactions rather than private sequence and storage attributes.
- Use level-triggered pyshmem waits and enable stream notifications for worker
  and sink trigger streams.
- Preserve externally attached streams on shutdown unless explicit external
  unlinking is requested.
- Require pyshmem 1.1.0 or newer and document the POSIX platform boundary.

The format is based on Keep a Changelog, and this project follows Semantic
Versioning.

## [1.0.2]

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
- **Test isolation.** Heavy `slow` integration tests are auto-forked via
  `pytest-forked` (added to the `test` extra) to keep C-level JIT state from
  accumulating across the suite.

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
