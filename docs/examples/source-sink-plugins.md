# Source And Sink Plugins

This example adds first-class endpoint plugins at both ends of the pipeline.

It packages:

- `example.simulated_camera` as a source entry point
- `example.npy_frame_sink` as a sink entry point

The pipeline itself is:

```text
example.simulated_camera -> cpu.scale -> example.npy_frame_sink
```

## Files

- `examples/source_sink_plugins/pyproject.toml`
- `examples/source_sink_plugins/pipeline.yaml`
- `examples/source_sink_plugins/run_example.py`
- `examples/source_sink_plugins/src/shmpipeline_example_endpoints/__init__.py`

## Install

Install the example package into the same environment as `shmpipeline`:

```bash
pip install -e .
pip install -e examples/source_sink_plugins
```

Restart any already-running CLI, GUI, or control-server process after the
install so the new entry points are visible.

## Run

```bash
python examples/source_sink_plugins/run_example.py
```

The demo waits until the sink has written a handful of processed frames to
`examples/source_sink_plugins/output/`, verifies the saved array shape and
dtype, and then shuts the manager down.

## What it demonstrates

- packaged entry-point discovery for non-kernel plugins
- a source plugin that owns pipeline ingress cadence
- a sink plugin that turns a shared-memory stream into files on disk
- GUI-friendly plugin kinds that appear in the `Sources` and `Sinks` editors
