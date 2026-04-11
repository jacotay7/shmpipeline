# Quick Start

This quick start builds a small CPU pipeline, pushes one frame through it, and reads the result back from shared memory.

## Python API

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

## Minimal YAML config

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

## CLI flow

Validate the config:

```bash
shmpipeline validate pipeline.yaml
```

Inspect the pipeline graph:

```bash
shmpipeline describe pipeline.yaml
```

Run the pipeline for a bounded duration:

```bash
shmpipeline run pipeline.yaml --duration 5.0
```

## Next steps

- Read [configuration](configuration.md) for the full YAML model.
- Read [runtime and observability](../guides/runtime.md) for lifecycle and health details.
- Read [worked examples](../examples/index.md) for larger pipelines.
