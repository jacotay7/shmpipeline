# Custom Kernel Extensions

`shmpipeline` supports third-party kernels through explicit registry extension.

The current `1.0` extension model is programmatic:

- define a `Kernel` subclass
- validate the stage-specific config in `validate_config`
- implement `compute_into`
- extend the default registry
- pass the extended registry into `PipelineManager`

Automatic plugin discovery is intentionally not part of the current release.

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

## Design guidance

- Keep custom kernels explicit and small.
- Treat `validate_config` as the contract boundary for user-facing config errors.
- Keep CPU and GPU kernels separate when storage semantics differ.
- Reuse shared-memory config validation rather than duplicating it in the kernel body.

## Testing guidance

A new kernel should normally ship with:

- config validation tests
- numerical behavior tests
- manager end-to-end tests when the kernel participates in pipeline execution

## Where to look in the API

- `Kernel`
- `KernelContext`
- `KernelRegistry`
- `get_default_registry()`
