# shmpipeline

## Links

- [PyPI](https://pypi.org/project/shmpipeline/)
- [Read the Docs](https://shmpipeline.readthedocs.io/en/latest/)
- [Issues](https://github.com/jacotay7/shmpipeline/issues)

## What It Is

`shmpipeline` builds local, process-based compute pipelines on top of named
shared-memory streams from `pyshmem`.

It is a good fit when you want explicit dataflow, process isolation, and
shared-memory throughput without giving up a small Python API, YAML-driven
configuration, or interactive tooling.

Typical uses include:

- adaptive optics and other real-time sensor / control pipelines
- CPU or GPU processing graphs around shared-memory streams
- validating, inspecting, and running pipelines from Python or the CLI
- editing and supervising pipelines from a desktop GUI
- extending the runtime with custom kernels, sources, and sinks

Good starting points:

- [Installation guide](https://shmpipeline.readthedocs.io/en/latest/getting-started/installation.html)
- [Quickstart](https://shmpipeline.readthedocs.io/en/latest/getting-started/quickstart.html)
- [Worked examples](https://shmpipeline.readthedocs.io/en/latest/examples/index.html)

## Installation

Full docs: [Installation guide](https://shmpipeline.readthedocs.io/en/latest/getting-started/installation.html)

Choose the smallest install that matches your workflow:

- Base runtime and CLI: `pip install shmpipeline`
- GPU support: `pip install "shmpipeline[gpu]"`
- Desktop GUI: `pip install "shmpipeline[gui]"`
- Remote control service: `pip install "shmpipeline[control]"`
- Editable source install: `pip install -e .`
- Full local development environment: `pip install -e ".[control,gpu,gui,test,docs]"`

Quick smoke test against a checked-in example:

```bash
shmpipeline validate examples/affine_transformation/pipeline.yaml
shmpipeline describe examples/affine_transformation/pipeline.yaml --json
```

## Open The GUI For The Observatory AO Example

Relevant docs:

- [GUI guide](https://shmpipeline.readthedocs.io/en/latest/guides/gui.html)
- [Observatory AO system example](https://shmpipeline.readthedocs.io/en/latest/examples/observatory-ao-system.html)

From the repository root:

```bash
pip install "shmpipeline[gui]"
shmpipeline-gui examples/observatory_ao_system/pipeline.yaml
```

The full GUI can auto-launch a local loopback control server when you press
`Build` or `Start`.

If you want to launch the server yourself first:

```bash
shmpipeline serve examples/observatory_ao_system/pipeline.yaml --host 127.0.0.1 --port 8765
shmpipeline-gui examples/observatory_ao_system/pipeline.yaml
```

## Quickstart For The Python API

Full docs:

- [Quickstart](https://shmpipeline.readthedocs.io/en/latest/getting-started/quickstart.html)
- [Configuration guide](https://shmpipeline.readthedocs.io/en/latest/getting-started/configuration.html)

Minimal pipeline config:

```yaml
shared_memory:
  - name: input_frame
    shape: [4]
    dtype: float32
    storage: cpu

  - name: scaled_frame
    shape: [4]
    dtype: float32
    storage: cpu

kernels:
  - name: scale_stage
    kind: cpu.scale
    input: input_frame
    output: scaled_frame
    parameters:
      factor: 2.0
```

Run it from Python:

```python
import numpy as np

from shmpipeline import PipelineConfig, PipelineManager

config = PipelineConfig.from_yaml("pipeline.yaml")
manager = PipelineManager(config)
manager.build()
manager.start()

manager.get_stream("input_frame").write(np.array([1, 2, 3, 4], dtype=np.float32))
result = manager.get_stream("scaled_frame").read_new(timeout=2.0)
print(result)

manager.stop()
manager.shutdown()
```

If you prefer the CLI for the same pipeline:

```bash
shmpipeline validate pipeline.yaml
shmpipeline describe pipeline.yaml
shmpipeline run pipeline.yaml --duration 5.0
shmpipeline benchmark pipeline.yaml --duration 5.0 \
  --source input_frame:random:1000 --json
```

## Popular Example Pages

- [Affine transformation](https://shmpipeline.readthedocs.io/en/latest/examples/affine-transformation.html)
- [Basic AO system](https://shmpipeline.readthedocs.io/en/latest/examples/basic-ao-system.html)
- [Observatory AO system](https://shmpipeline.readthedocs.io/en/latest/examples/observatory-ao-system.html)
- [Source and sink plugins](https://shmpipeline.readthedocs.io/en/latest/examples/source-sink-plugins.html)

## More Documentation

Start with these in roughly the order most users need them:

1. [Configuration guide](https://shmpipeline.readthedocs.io/en/latest/getting-started/configuration.html) for the full YAML model, parameters, and shared-memory definitions.
2. [Worked examples](https://shmpipeline.readthedocs.io/en/latest/examples/index.html) for complete CPU, GPU, custom-operation, AO, and plugin-backed pipelines.
3. [CLI guide](https://shmpipeline.readthedocs.io/en/latest/guides/cli.html) for `validate`, `describe`, `run`, `benchmark`, and `serve`.
4. [GUI guide](https://shmpipeline.readthedocs.io/en/latest/guides/gui.html) for editing configs, validating locally, inspecting graphs, and launching viewers.
5. [Performance guide](https://shmpipeline.readthedocs.io/en/latest/guides/performance.html) for benchmark baselines, lock polling, placement, and CPU/GPU tuning.
6. [Runtime guide](https://shmpipeline.readthedocs.io/en/latest/guides/runtime.html) for lifecycle, metrics, worker health, and synthetic inputs.
7. [Extensions guide](https://shmpipeline.readthedocs.io/en/latest/guides/extensions.html) for custom kernels, sources, and sinks.
8. [Control plane guide](https://shmpipeline.readthedocs.io/en/latest/guides/control-plane.html) for remote management, SSE events, and the Python client.
9. [API reference](https://shmpipeline.readthedocs.io/en/latest/reference/index.html), [core API](https://shmpipeline.readthedocs.io/en/latest/reference/core.html), and [kernel catalog](https://shmpipeline.readthedocs.io/en/latest/reference/kernels.html) for the detailed reference surface.
10. [Troubleshooting](https://shmpipeline.readthedocs.io/en/latest/guides/troubleshooting.html) for common setup and runtime issues.
