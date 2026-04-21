"""Public package surface for shmpipeline."""

from importlib import import_module

__version__ = "1.0.0"

_EXPORTS = {
    "Kernel": ("shmpipeline.kernel", "Kernel"),
    "KernelConfig": ("shmpipeline.config", "KernelConfig"),
    "KernelContext": ("shmpipeline.kernel", "KernelContext"),
    "KernelRegistry": ("shmpipeline.registry", "KernelRegistry"),
    "PipelineConfig": ("shmpipeline.config", "PipelineConfig"),
    "PipelineGraph": ("shmpipeline.graph", "PipelineGraph"),
    "PipelineManager": ("shmpipeline.manager", "PipelineManager"),
    "PipelineState": ("shmpipeline.state", "PipelineState"),
    "SharedMemoryConfig": ("shmpipeline.config", "SharedMemoryConfig"),
    "Sink": ("shmpipeline.sink", "Sink"),
    "SinkConfig": ("shmpipeline.config", "SinkConfig"),
    "SinkContext": ("shmpipeline.sink", "SinkContext"),
    "Source": ("shmpipeline.source", "Source"),
    "SourceConfig": ("shmpipeline.config", "SourceConfig"),
    "SourceContext": ("shmpipeline.source", "SourceContext"),
    "SyntheticInputConfig": (
        "shmpipeline.synthetic",
        "SyntheticInputConfig",
    ),
    "get_default_registry": (
        "shmpipeline.registry",
        "get_default_registry",
    ),
}

__all__ = [
    "Kernel",
    "KernelConfig",
    "KernelContext",
    "KernelRegistry",
    "PipelineConfig",
    "PipelineGraph",
    "PipelineManager",
    "PipelineState",
    "SharedMemoryConfig",
    "Sink",
    "SinkConfig",
    "SinkContext",
    "Source",
    "SourceConfig",
    "SourceContext",
    "SyntheticInputConfig",
    "__version__",
    "get_default_registry",
]


def __getattr__(name: str):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
