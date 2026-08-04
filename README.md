# shmpipeline

[![CI](https://github.com/jacotay7/shmpipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/jacotay7/shmpipeline/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/shmpipeline.svg)](https://pypi.org/project/shmpipeline/)
[![Python](https://img.shields.io/pypi/pyversions/shmpipeline.svg)](https://pypi.org/project/shmpipeline/)
[![Docs](https://img.shields.io/badge/docs-shmpipeline.readthedocs.io-teal.svg)](https://shmpipeline.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Documentation: [shmpipeline.readthedocs.io](https://shmpipeline.readthedocs.io/en/latest/)**

**Real-time compute pipelines from a YAML file: one process per stage, zero-copy shared memory between them.**

<p align="center">
  <img src="docs/_static/images/pipeline_showcase.webp" width="860" alt="A live adaptive-optics control loop running as a shmpipeline dataflow graph: WFS image, measured centroids and DM commands above five kernel stages labelled with their measured execution times.">
</p>

You describe a graph of compute stages in YAML. `shmpipeline` builds it: every
kernel gets its own OS process pinned to a CPU core, and the stages hand data to
one another through named [`pyshmem`](https://github.com/jacotay7/pyshmem)
shared-memory streams — no queues, no serialisation, no copies. The design
target is the lowest latency a Python-configured system can reach, which is why
it is built for **adaptive-optics and other real-time sensor/control loops**.

What you get on top of the runtime: a CLI to validate, describe, run, and
benchmark a pipeline; a desktop GUI to edit and supervise one; a REST + SSE
control plane to drive it remotely; and entry-point plugins so your own kernels,
sources, and sinks are first-class.

The clip above is the [observatory AO
example](examples/observatory_ao_system/) actually running — a 256² Shack-Hartmann
image reduced to 1024 mirror commands through five stages, at about **1.8 kHz**.
Every number on it is measured while it renders, so it moves a little run to
run; the table below cites one versioned snapshot.

## Install

```bash
pip install shmpipeline                    # runtime + CLI
pip install "shmpipeline[gpu]"             # + torch GPU kernels
pip install "shmpipeline[gui]"             # + PySide6 desktop GUI
pip install "shmpipeline[control]"         # + FastAPI control plane
pip install -e ".[control,gpu,gui,test,docs]"   # full development environment
```

## Quickstart

A pipeline is a set of shared-memory streams and the kernels that connect them:

```yaml
shared_memory:
  - {name: input_frame,  shape: [4], dtype: float32, storage: cpu}
  - {name: scaled_frame, shape: [4], dtype: float32, storage: cpu}

kernels:
  - name: scale_stage
    kind: cpu.scale          # or gpu.scale
    input: input_frame
    output: scaled_frame
    parameters: {factor: 2.0}
```

Run it from Python:

```python
import numpy as np

from shmpipeline import PipelineConfig, PipelineManager

manager = PipelineManager(PipelineConfig.from_yaml("pipeline.yaml"))
manager.build()  # create the shared-memory streams
manager.start()  # spawn one pinned worker process per kernel

manager.get_stream("input_frame").write(
    np.array([1, 2, 3, 4], dtype=np.float32)
)
print(manager.get_stream("scaled_frame").read_new(timeout=2.0))

manager.stop()
manager.shutdown()
```

…or from the CLI, without writing any Python:

```bash
shmpipeline validate pipeline.yaml
shmpipeline describe pipeline.yaml --json
shmpipeline run      pipeline.yaml --duration 5.0
shmpipeline benchmark pipeline.yaml --duration 5.0 --source input_frame:random:1000
shmpipeline-gui      pipeline.yaml          # edit and supervise it in the GUI
```

See the **[Quickstart](https://shmpipeline.readthedocs.io/en/latest/getting-started/quickstart.html)**,
the **[Configuration guide](https://shmpipeline.readthedocs.io/en/latest/getting-started/configuration.html)**
for the full YAML model, and the
**[worked examples](https://shmpipeline.readthedocs.io/en/latest/examples/index.html)**
for complete CPU, GPU, custom-operation, and plugin-backed pipelines.

## Benchmarks

The [observatory AO example](examples/observatory_ao_system/) — a five-stage
control loop turning a 256×256 Shack-Hartmann image into 1024 deformable-mirror
commands — on an AMD Ryzen 9 9950X3D. Each stage is a separate pinned process;
per-stage times are what the live workers reported:

| Stage | Kernel kind | Publishes | Exec time | Jitter (RMS) |
| --- | --- | ---: | ---: | ---: |
| centroid | `cpu.shack_hartmann_centroid` | 32×32×2 | 61 µs | 0.91 µs |
| flatten | `cpu.flatten` | 2048 | 12 µs | 0.35 µs |
| reconstruct | `cpu.affine_transform` | 1024 | 81 µs | 0.99 µs |
| integrate | `cpu.leaky_integrator` | 1024 | 12 µs | 0.35 µs |
| clip | `cpu.custom_operation` | 1024 | 18 µs | 0.36 µs |

End to end the loop sustains **1,817 Hz**, with terminal frame spacing of
0.60 ms at p50 and 0.63 ms at p99. Note that the runtime reports terminal
*inter-arrival spacing*, not true end-to-end latency, and that this example
still uses the legacy `cpu.shack_hartmann_centroid` compatibility kernel — new
AO systems should use the `shmpipeline-ao` plugin's `ao.*` kinds. The raw artifact is
[versioned with the benchmarks](benchmarks/results/rtx5090-linux-observatory-ao-2026-08-04.json);
reproduce it with:

```bash
python benchmarks/benchmark_pipeline.py examples/observatory_ao_system/pipeline.yaml \
  --duration 5 --warmup 1 --source obs_wfs_image:random --json-out result.json
```

See the **[performance guide](https://shmpipeline.readthedocs.io/en/latest/guides/performance.html)**
for baselines, lock polling, placement, and CPU/GPU tuning, and
[`benchmarks/results/`](benchmarks/results/) for the dated snapshot history.

## Features

- **A pipeline is a config file** — streams, kernels, sources, and sinks in
  validated YAML, with unknown-key and reference errors reported against the
  line that caused them; see
  **[Configuration](https://shmpipeline.readthedocs.io/en/latest/getting-started/configuration.html)**.
- **One process per kernel** — each stage runs in its own OS process with CPU
  affinity (round-robin by default, or your own placement policy), so a slow
  stage cannot stall its neighbours and the GIL is never in the path.
- **Zero-copy shared memory** — stages exchange data through `pyshmem` streams
  with futex-backed level-triggered waits; kernels write straight into the
  locked output buffer rather than allocating and copying.
- **A kernel library, CPU and GPU** — copy, scale, elementwise arithmetic,
  affine transform, flatten, concatenate, reduce, leaky integrator, spot
  centroiding, and a runtime-compiled `custom_operation`; CPU kernels JIT
  through Numba, GPU kernels through torch. See the
  **[kernel catalog](https://shmpipeline.readthedocs.io/en/latest/reference/kernels.html)**.
- **Multi-input and multi-output stages** — fan-in and fan-out with an explicit
  `trigger_policy`, plus a frame-id barrier that synchronises several cameras
  and reports skew instead of silently mixing generations.
- **Live pipeline surgery** — `restart()` replaces only the failed workers,
  and `add_kernel()` adds a stage to a *running* pipeline, rolling back cleanly
  if the spawn fails.
- **Measured, not asserted** — `PipelineManager.benchmark()` reports
  throughput, frame spacing percentiles, and per-worker exec time and jitter;
  see the **[runtime guide](https://shmpipeline.readthedocs.io/en/latest/guides/runtime.html)**.
- **Three ways to drive it** — the `shmpipeline`
  **[CLI](https://shmpipeline.readthedocs.io/en/latest/guides/cli.html)**, the
  **[desktop GUI](https://shmpipeline.readthedocs.io/en/latest/guides/gui.html)**
  for editing and supervising, and a
  **[REST + SSE control plane](https://shmpipeline.readthedocs.io/en/latest/guides/control-plane.html)**
  with scoped tokens and auto-reconnecting event streams.
- **Extensible by entry point** — register your own kernels, sources, and sinks
  from your own package, no fork required; see
  **[Extensions](https://shmpipeline.readthedocs.io/en/latest/guides/extensions.html)**.
  Adaptive optics is itself a plugin: `shmpipeline-ao` supplies the `ao.*`
  kinds (`ao.cpu.shack_hartmann_slopes`, `ao.cpu.reconstruct`,
  `ao.cpu.detector_calibration`, …) and its modes resolve to ordinary pipeline
  YAML. The in-tree `cpu.shack_hartmann_centroid`, `cpu.tip_tilt_controller`,
  and `cpu.tomographic_controller` kinds remain only as compatibility shims and
  are slated for removal in 2.0.

See the **[API reference](https://shmpipeline.readthedocs.io/en/latest/reference/index.html)**
for the detailed surface, and
**[Troubleshooting](https://shmpipeline.readthedocs.io/en/latest/guides/troubleshooting.html)**
for common setup and runtime issues.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The gates CI enforces are lint, an 80%
coverage floor, and a changelog entry for every user-facing change:

```bash
ruff check . && ruff format --check .
python -m pytest -m "not slow" -q && python -m pytest -m slow -q
```

## License

MIT — see [LICENSE](LICENSE).
