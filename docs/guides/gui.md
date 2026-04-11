# GUI Guide

The `shmpipeline-gui` desktop application is the interactive surface for editing YAML, validating configs, running pipelines, and inspecting live streams.

## Launch

Install the GUI extra:

```bash
pip install "shmpipeline[gui]"
```

Start the application:

```bash
shmpipeline-gui
```

## Main areas of the application

The main window combines several responsibilities:

- shared-memory table editor
- kernel table editor
- validation status and graph summary
- runtime controls for build, start, pause, resume, stop, and shutdown
- live worker metrics
- synthetic input management
- viewer launching for configured streams

:::{figure} ../_static/images/gui/gui-main-window.png
:alt: Main shmpipeline GUI window showing shared-memory, kernel, and runtime panels.
:width: 100%

Main window with shared-memory definitions, kernel stages, runtime metrics, and lifecycle controls.
:::

## Typical editing flow

1. Load an existing pipeline YAML file or start from an empty document.
2. Add shared-memory definitions.
3. Add kernel stages and wire their inputs and outputs.
4. Validate the config before building.
5. Build the pipeline.
6. Start workers and inspect runtime status.
7. Launch viewers for selected streams.
8. Stop or shut down the pipeline when finished.

## Shared memory editor

The shared-memory dialog exposes:

- stream name
- shape
- dtype
- storage backend
- GPU device
- CPU mirror toggle

Use CPU mirrors on GPU streams when you need CPU-side readers such as host-side viewers or external tools.

:::{figure} ../_static/images/gui/gui-shared-memory-dialog.png
:alt: Shared-memory editor area in the shmpipeline GUI showing stream definitions and viewer controls.
:width: 100%

Shared-memory editor with stream metadata, viewer launch controls, and synthetic-input actions.
:::

## Kernel editor

The kernel dialog exposes:

- kernel name
- kernel kind
- trigger input
- output
- auxiliary YAML
- operation string for custom-operation kernels
- parameter YAML
- read timeout
- pause polling interval

:::{figure} ../_static/images/gui/gui-kernel-dialog.png
:alt: Kernel editor area in the shmpipeline GUI showing pipeline stages, kinds, inputs, outputs, and auxiliary bindings.
:width: 100%

Kernel editor with stage ordering, kernel kinds, trigger inputs, outputs, and auxiliary stream bindings.
:::

## Runtime controls

The main window follows the same state model as `PipelineManager`:

- build
- start
- pause
- resume
- stop
- shutdown

The runtime table shows worker identity and health, including PID, CPU slot, frame counts, timing, and throughput.

:::{figure} ../_static/images/gui/gui-runtime-status.png
:alt: Runtime status table in the shmpipeline GUI showing worker PIDs, health, frame counts, and throughput metrics.
:width: 100%

Runtime metrics view showing worker liveness, frame counts, average execution time, jitter, and throughput.
:::

## Synthetic inputs

The GUI can start and stop synthetic writers for built input streams. This is useful for:

- smoke-testing a new pipeline without an external producer
- driving viewers during demos
- reproducing deterministic test scenarios

:::{figure} ../_static/images/gui/gui-synthetic-input-dialog.png
:alt: Synthetic input configuration dialog in the shmpipeline GUI.
:width: 40%

Synthetic-input dialog for selecting the target stream, pattern, rate, seed, and pattern-specific parameters.
:::

## Stream viewers

Viewer windows are launched as separate processes so they can keep updating even while the main GUI remains responsive.

The viewer status distinguishes between:

- stream rate derived from shared-memory metadata
- viewer refresh rate derived from the local timer

:::{figure} ../_static/images/gui/gui-viewer-window.png
:alt: Stream viewer window in the shmpipeline GUI displaying a live grayscale image and update rates.
:width: 70%

Viewer window showing a live stream, stream update rates, and local viewer refresh statistics.
:::

## Screenshot assets

The documentation uses stable filenames from `docs/_static/images/gui/`. If the GUI layout changes, replace the images in that directory and rebuild the docs.
