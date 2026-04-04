# Build Plan

## Completed Foundation

The original proof-of-concept scope is no longer a plan item. It is now the
baseline for the repository.

- Built the pipeline manager on top of `pyshmem` rather than reimplementing
  shared-memory transport.
- Shipped YAML-driven configuration via immutable dataclass models with
  validation for shared-memory definitions, kernel bindings, storage backends,
  and required parameters.
- Implemented a process-per-kernel runtime with child-process stream reopen,
  worker lifecycle events, surfaced failures, and pause/resume/stop/shutdown
  control.
- Landed a coherent manager state machine covering `build`, `start`, `pause`,
  `resume`, `stop`, and `shutdown`.
- Split built-in kernels cleanly by backend and expanded beyond the initial
  proof-of-concept set to include both CPU and GPU kernel families.
- Added GPU support with CUDA-backed streams and GPU examples that mirror the
  CPU examples.
- Established a real test suite across config validation, kernel behavior,
  manager supervision, and GUI document modeling.
- Added a desktop GUI for editing, validating, and running pipeline
  definitions.

## Completed Operability Work

What used to be planned as Phase 2 is now mostly shipped.

- Added first-class CLI entry points for `validate`, `describe`, and `run`
  workflows.
- Exposed pipeline graph introspection through `PipelineGraph`, CLI output,
  and GUI previews.
- Added graph-level validation for producer/consumer wiring and related config
  errors before worker startup.
- Added runtime snapshots, rolling worker metrics, structured failure
  reporting, and GUI runtime status panes.
- Refactored worker placement into an explicit scheduling policy surface with
  a default round-robin implementation.
- Added light and dark GUI themes with light as the default startup theme.
- Added backend synthetic input generation plus GUI controls for starting,
  stopping, and reconfiguring test inputs.
- Rebuilt the shared-memory viewer path so vector and image streams can be
  inspected from the GUI again.
- Moved viewer windows into separate spawned Python processes and surfaced
  stream-rate metadata in the viewer status.
- Expanded test coverage for CLI flows, graph derivation, synthetic inputs,
  GUI behavior, and GPU regressions.
- Added repo automation including Ruff checks and GitHub Actions coverage for
  supported Python versions.
- Updated README examples and operational documentation to cover the shipped
  headless and introspection workflows.

## Current Baseline

Today the repository already provides:

- YAML-only pipeline authoring with validated CPU and GPU stream definitions
- a package-installable CLI and desktop GUI
- graph introspection before process startup
- runtime snapshots and rolling worker metrics for GUI and CLI consumers
- synthetic input generation for tests, demos, and live GUI operation
- CPU and GPU example pipelines with parity across the built-in kernel family
- automated tests, Ruff checks, and CI workflows

That changes the planning problem again. The next phase is no longer about
adding the first operability surface. It is about hardening the shipped system,
closing the rough edges that appear under sustained GPU workloads, and making
extension workflows clearer.

## Next Phase Theme

The next phase is about hardening, polish, and extension points.

The project now has real headless and GUI workflows, but there are still a few
practical gaps: GPU lifecycle cleanup needs to be quieter and more predictable,
viewer and runtime observability can still be refined, and the extension story
for larger deployments is still mostly implicit.

## Remaining Goals

1. Harden GPU shared-memory lifecycle and shutdown behavior.
2. Refine runtime observability and viewer ergonomics for long-running
   pipelines.
3. Close the remaining documentation and examples gaps for advanced workflows.
4. Make the extension and deployment surface more explicit.
5. Keep a focused backlog for advanced scheduling and visualization without
   destabilizing the current core.

## Remaining Workstreams

### 1. GPU Lifecycle And Cleanup

The runtime now works correctly under realistic GPU examples, but shutdown and
resource cleanup still need another pass.

Scope:

- Eliminate or substantially reduce the current GPU shared-memory teardown
  noise, especially `multiprocessing.resource_tracker` warnings.
- Audit the full create/open/close/unlink lifecycle for CUDA-backed streams so
  long-running sessions and repeated rebuilds are predictable.
- Tighten manager shutdown behavior around viewer processes, worker teardown,
  and remaining local stream handles.
- Add regression coverage for repeated build/start/stop/shutdown cycles on GPU
  pipelines.

Why this matters:

- It removes operator confusion around successful runs that still end with
  noisy warnings.
- It makes GPU workflows feel production-ready rather than merely functional.

### 2. Observability And Viewer Polish

The current metrics surface is useful, but it can be made more operationally
complete.

Scope:

- Continue refining the relationship between worker metrics, shared-memory
  metadata, and viewer status so rates and timing remain easy to interpret.
- Keep passive-viewer behavior explicit for CPU, GPU, and CPU-mirror-backed
  GPU streams.
- Improve how runtime status communicates stalled inputs, inactive sources, and
  degraded workers.
- Add a small amount of additional operator-facing troubleshooting context
  where it helps, without turning the runtime into a full telemetry stack.

Why this matters:

- The project is now usable enough that debugging quality matters more than raw
  feature count.
- Better operational feedback reduces the need to inspect shared memory by
  hand.

### 3. Documentation And Advanced Examples

The core workflows are documented, but the operational guidance is still thin
in places.

Scope:

- Expand the README and example documentation for mixed CPU/GPU pipelines,
  synthetic input strategies, CPU mirrors, and viewer behavior.
- Add an operator-oriented troubleshooting section for common runtime and GPU
  issues.
- Document lifecycle expectations for build/rebuild/shutdown in both CLI and
  GUI flows.
- Add one or two more realistic example pipelines that demonstrate the current
  best practices rather than only the kernel primitives.

Why this matters:

- The project now has enough surface area that users need guidance on how to
  use it well, not just how to call the API.

### 4. Extension And Packaging Surface

The package is installable and automated, but the extension story is still
mostly inferred from the codebase.

Scope:

- Clarify how third-party kernels should be packaged, registered, and tested.
- Decide whether plugin discovery belongs in the next phase or stays a manual
  integration story for now.
- Review packaging metadata and optional dependency guidance so CPU-only, GPU,
  GUI, and test installs are clearly documented.
- Keep the public API surface small and explicit as these extension points are
  documented.

Why this matters:

- A clearer extension story makes the project easier to adopt in real systems
  without forcing premature plugin infrastructure.

### 5. Deferred Advanced Work

These are real follow-on items, but they should stay behind hardening work.

Scope:

- richer graph visualization in the GUI or CLI
- historical metrics export or external telemetry integration
- more advanced scheduling strategies beyond local affinity hints
- plugin or third-party kernel discovery workflows
- remote control or multi-host orchestration

Why this matters:

- These are attractive expansions, but they should not compete with stability
  and clarity in the current single-host runtime.

## Proposed Exit Criteria For The Next Phase

The next phase is complete when the repository can support the following story
cleanly:

- GPU example pipelines can be built, run, and shut down repeatedly without the
  current cleanup noise or stale-handle confusion.
- Runtime status and viewer surfaces communicate pipeline health and stream
  rates clearly enough for day-to-day debugging.
- The README and examples document the supported CPU/GPU, synthetic-input, and
  viewer workflows well enough that users do not need to reverse-engineer the
  intended patterns from tests.
- The project has an explicit documented path for extending the built-in kernel
  set.
- The remaining advanced work is clearly separated into backlog rather than
  being mixed into near-term execution work.

## Later Backlog

These remain good follow-on items after the next hardening phase:

- richer visualization of pipeline graphs
- historical metrics export or external telemetry integration
- plugin discovery or third-party kernel packaging workflows
- remote control or multi-host orchestration
- more advanced scheduling strategies beyond local affinity hints
