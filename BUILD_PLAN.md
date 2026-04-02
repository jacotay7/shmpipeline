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

## Current Baseline

Today the project already provides:

- a stable config, runtime, and manager core
- CPU and GPU built-in kernels with example pipelines
- worker supervision and structured failure propagation
- a GUI for pipeline editing and runtime control
- automated tests for the shipped scaffold

That changes the planning problem. The next phase should focus less on proving
the architecture and more on making the system easier to operate, inspect, and
extend.

## Next Phase Theme

Phase 2 is about operability and introspection.

The repository has a working execution engine, but it still lacks the tooling
that makes the engine easy to run headlessly, reason about from the outside,
and tune under real workloads. The next phase should turn the current scaffold
into a more complete developer tool and runtime platform.

## Phase 2 Goals

1. Add a first-class command-line interface for headless workflows.
2. Expose pipeline graph and configuration introspection APIs.
3. Add runtime metrics and diagnostics that are useful in both CLI and GUI
   flows.
4. Improve worker placement and runtime policy beyond the current basic
   round-robin CPU slot assignment.
5. Improve the GUI so it is theme-aware, operationally useful, and reliable
  for live data inspection.
6. Harden packaging, documentation, and tests around the new operational
  surface area.

## Phase 2 Workstreams

### 1. CLI And Headless Operation

Deliver a supported command-line entry point for working with YAML pipelines
without opening the GUI.

Scope:

- Add a `shmpipeline` CLI with subcommands such as `validate`, `describe`,
  and `run`.
- Support loading a pipeline from YAML, validating it, building it, starting
  it, and shutting it down cleanly from a terminal session.
- Return meaningful exit codes for config errors, runtime failures, and user
  interrupts.
- Expose runtime options such as logging level, startup timeout, and shutdown
  behavior.
- Ensure the CLI reuses the same config and manager APIs as the GUI rather than
  introducing a second execution path.

Why this matters:

- It unlocks automation, CI smoke tests, remote execution, and scripted
  demonstrations.
- It gives the project a user-facing interface that matches the maturity of
  the existing runtime.

### 2. Pipeline Graph Introspection

Make pipeline structure visible as data instead of leaving it implicit inside
`PipelineConfig` and the manager.

Scope:

- Add a graph model that derives nodes, edges, stream producers, and stream
  consumers from the loaded config.
- Provide a programmatic API for graph queries such as upstream/downstream
  dependencies and orphaned resources.
- Add static validation for graph-level issues where appropriate, including
  ambiguous bindings and configuration patterns that should be rejected before
  process startup.
- Support human-readable CLI output for `describe` and a machine-readable form
  such as JSON for tooling integration.
- Reuse this graph model in the GUI so the editor and runtime surface reflect
  the same structure.

Why this matters:

- It gives users a way to reason about larger pipelines before they run.
- It creates a shared foundation for later visualization, metrics overlays, and
  debugging tools.

### 3. Metrics And Diagnostics

Add lightweight observability so runtime behavior can be inspected without
attaching a debugger.

Scope:

- Record per-worker lifecycle state, processed-frame counts, last successful
  execution time, and failure metadata.
- Capture basic latency and throughput metrics at the worker level.
- Add manager APIs for retrieving a runtime snapshot suitable for GUI polling
  and CLI reporting.
- Surface worker exceptions and restart/stop outcomes in a more structured,
  user-facing form.
- Keep the design minimal: start with in-process aggregation through the
  existing event channel before considering external metrics backends.

Why this matters:

- It reduces time-to-diagnosis when pipelines stall or underperform.
- It gives the GUI and future CLI commands concrete runtime state to display.

### 4. Scheduling And Runtime Policy

The current CPU placement logic is a reasonable baseline, but it should become
an explicit policy surface rather than a hidden implementation detail.

Scope:

- Refactor worker placement into a strategy or policy abstraction.
- Preserve the current best-effort behavior as the default fallback.
- Add config-level or manager-level hooks for affinity preferences where the
  platform supports them.
- Improve backend-aware startup checks, especially around GPU device
  availability and resource mismatches.
- Define how future policy decisions interact with mixed CPU/GPU pipelines.

Why this matters:

- It keeps the current implementation simple while making room for better
  workload placement later.
- It avoids baking future scheduling limits into the current manager API.

### 5. GUI Usability, Test Inputs, And Live Viewers

The GUI should move from basic control surface to practical operator tool.

Scope:

- Add explicit light and dark theme support with light as the default startup
  theme rather than forcing a default-dark presentation.
- Make theme choice a first-class setting in the GUI so users can switch
  without restarting the app.
- Keep theme implementation deliberate and limited in scope: shared color
  tokens, widget styling that matches the current Qt surface, and no forked UI
  logic per theme.
- Add backend support for synthetic input sources that can drive pipelines at
  the maximum sustainable rate for testing and demos.
- Start the synthetic input work in the backend so the same mechanism can be
  used from tests, future CLI flows, and the GUI.
- Support deterministic random generation with a fixed seed, ramp patterns,
  and a small initial set of other common patterns such as constant, impulse,
  and sine.
- Define synthetic-source controls in terms of target input stream, pattern,
  rate policy, dtype, shape, and reproducibility so they integrate cleanly with
  existing config and manager APIs.
- Add GUI controls for starting and stopping these test injectors once the
  backend API exists.
- Repair the shared-memory viewer path so launching a live viewer from the GUI
  is reliable again.
- Treat the current `pyqtgraph` viewer failure as a compatibility bug to be
  fixed at the root, not worked around by requiring a default-dark UI or a
  fragile import chain.
- Avoid viewer startup paths that implicitly drag in optional matplotlib
  integrations when they are not needed for image display.
- Add dedicated live views for image-like and vector-like shared memory with
  frame-rate-oriented refresh behavior suitable for active pipelines.
- Make viewer launch available for any shared-memory resource from inside the
  GUI, with clear handling for unsupported shapes or unavailable streams.

Why this matters:

- Theme support makes the GUI usable in more environments without baking in a
  default-dark assumption.
- Synthetic test inputs make it much easier to benchmark, debug, and validate
  full pipeline chains without external producers.
- Reliable live viewers are essential if the GUI is going to be used to
  inspect running pipelines rather than only configure them.

### 6. Documentation, Examples, And Test Expansion

The next phase introduces new operational surfaces, so the supporting material
needs to grow with it.

Scope:

- Add CLI-focused documentation and examples that cover validate/describe/run
  workflows.
- Add tests for graph derivation, metrics reporting, CLI behavior, synthetic
  input generation, theme selection, and viewer launch behavior.
- Keep multiprocessing tests aligned with `spawn` semantics and real
  `pyshmem` resources.
- Add failure-path coverage for CLI exit codes and headless shutdown handling.
- Add regression coverage for the current viewer initialization failure so the
  GUI does not silently regress on dependency interactions.
- Document expected behavior for mixed CPU/GPU pipelines and runtime
  observability fields.
- Document the supported synthetic patterns and the semantics of fixed-seed
  reproducibility.

Why this matters:

- The project already has a strong testing culture; the new work should extend
  that discipline instead of bypassing it.

## Proposed Phase 2 Exit Criteria

Phase 2 is complete when the repository can support the following end-to-end
story:

- A user can validate and run a YAML pipeline from the command line.
- A user can inspect the pipeline graph without starting worker processes.
- The manager can report a structured runtime snapshot for active pipelines.
- The GUI and CLI consume the same underlying introspection and runtime-status
  APIs.
- A user can switch between light and dark themes, with light as the default.
- A user can inject deterministic synthetic test patterns into pipeline inputs
  from shared backend APIs and from the GUI.
- A user can open live shared-memory viewers from the GUI without triggering
  the current viewer startup failure.
- The new features are covered by tests and documented in the README and
  examples.

## Later Backlog

These items remain good follow-on work, but they do not need to be in the next
phase unless the scope expands:

- richer visualization of pipeline graphs
- historical metrics export or external telemetry integration
- plugin or third-party kernel packaging workflows
- remote control or multi-host orchestration
- more advanced scheduling strategies beyond local affinity hints