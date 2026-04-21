# Source And Sink Plugin Package Example

This example shows a small installable plugin package that exposes one source
and one sink through `shmpipeline` entry points.

It demonstrates two things at once:

- how to package custom endpoint plugins so the default registry, CLI, control
  server, and GUIs can discover them automatically
- how to build a pipeline where `shmpipeline` owns both the input source and
  the output sink instead of assuming another process handles the pipeline
  edges

## Plugins

- `example.simulated_camera`: a CPU source that emits a moving Gaussian spot
  image with configurable amplitude, background, noise, and cadence
- `example.npy_frame_sink`: a CPU sink that writes processed frames to `.npy`
  files in an output directory

## Install

Install the main package first if needed, then install the example plugin
package in editable mode:

```bash
pip install -e .
pip install -e examples/source_sink_plugins
```

Entry points are discovered when Python starts, so restart any already-running
GUI, CLI, or server process after installing the example package.

## Run

```bash
python examples/source_sink_plugins/run_example.py
```

The example builds a source -> kernel -> sink pipeline:

- source: `example.simulated_camera`
- kernel: `cpu.scale`
- sink: `example.npy_frame_sink`

It waits until a few processed frames are written under
`examples/source_sink_plugins/output/`, validates the saved array shape and
dtype, and then shuts the pipeline down.

## Try It In The GUI

After installing the example package, open `shmpipeline-gui` and create a
pipeline using the kinds:

- `example.simulated_camera`
- `example.npy_frame_sink`

Those kinds should appear in the `Sources` and `Sinks` editors and, when the
GUI is connected to a server running in the same environment, in the runtime
status tabs as well.
