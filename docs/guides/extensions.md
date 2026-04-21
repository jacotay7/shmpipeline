# Custom Kernel, Source, And Sink Extensions

`shmpipeline` supports third-party kernels, sources, and sinks through the
registry surface used by `PipelineManager` and the control service.

There are two supported discovery paths:

- explicit registry extension in Python code
- packaged entry-point discovery through `shmpipeline.kernels`,
  `shmpipeline.sources`, and `shmpipeline.sinks`

Programmatic registration is still useful in tests and embedded applications.

## Kernel Example

- define a `Kernel` subclass
- validate the stage-specific config in `validate_config`
- implement `compute_into`
- extend the default registry
- pass the extended registry into `PipelineManager`

## Example

```python
import numpy as np

from shmpipeline import Kernel, PipelineConfig, PipelineManager, get_default_registry


class BiasCpuKernel(Kernel):
    kind = "example.bias"
    storage = "cpu"

    @classmethod
    def validate_config(cls, config, shared_memory):
        super().validate_config(config, shared_memory)
        if "bias" not in config.parameters:
            raise ValueError("example.bias requires a 'bias' parameter")

    def compute_into(self, trigger_input, output, auxiliary_inputs):
        output[...] = np.asarray(trigger_input) + float(self.context.config.parameters["bias"])


config = PipelineConfig.from_yaml("pipeline.yaml")
registry = get_default_registry().extended(BiasCpuKernel)
manager = PipelineManager(config, registry=registry)
```

## Source Example

Sources write payloads into one configured stream. The runtime owns the thread
loop and calls `read()` repeatedly.

```python
import numpy as np

from shmpipeline import Source


class DemoCamera(Source):
    kind = "example.camera"
    storage = "cpu"

    def read(self):
        if self.wait(self.context.config.poll_interval):
            return None
        return np.arange(4, dtype=np.float32)
```

## Sink Example

Sinks consume payloads from one configured stream. The runtime waits for new
stream writes and calls `consume()` for each payload.

```python
import numpy as np

from shmpipeline import Sink


class DemoDisplay(Sink):
    kind = "example.display"
    storage = "cpu"

    def consume(self, value):
        print(np.asarray(value))
```

## Packaged Discovery

Expose plugins through entry points so `get_default_registry()` and the GUIs
can discover them automatically in the active Python environment.

```toml
[project.entry-points."shmpipeline.kernels"]
"example.bias" = "example_plugins:BiasCpuKernel"

[project.entry-points."shmpipeline.sources"]
"example.camera" = "example_plugins:DemoCamera"

[project.entry-points."shmpipeline.sinks"]
"example.display" = "example_plugins:DemoDisplay"
```

The entry-point name must match the plugin class `kind`.

## Design guidance

- Keep custom kernels explicit and small.
- Treat `validate_config` as the contract boundary for user-facing config errors.
- Keep CPU and GPU kernels separate when storage semantics differ.
- Reuse shared-memory config validation rather than duplicating it in the plugin body.
- Keep source and sink plugins cooperatively stoppable; they run in manager-owned threads.
- Use `Source.wait(...)` when a source needs a simple cadence without manual stop-event handling.
- Return `None` from `Source.read()` when no frame is ready yet instead of busy-spinning.

## Testing guidance

A new plugin should normally ship with:

- config validation tests
- numerical or payload behavior tests
- manager end-to-end tests when the plugin participates in pipeline execution
- GUI or control-plane discovery tests if the plugin is shipped as a package

## Where to look in the API

- `Kernel`
- `KernelContext`
- `KernelRegistry`
- `Source`
- `SourceContext`
- `Sink`
- `SinkContext`
- `get_default_registry()`
