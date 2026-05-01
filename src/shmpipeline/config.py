"""Immutable configuration models for shared-memory pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from shmpipeline.errors import ConfigValidationError


def _expect_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{context} must be a mapping")
    return value


def _reject_unexpected_keys(
    data: Mapping[str, Any],
    *,
    context: str,
    allowed: set[str],
) -> None:
    unexpected = sorted(set(data) - allowed)
    if unexpected:
        joined = ", ".join(repr(key) for key in unexpected)
        raise ConfigValidationError(
            f"{context} contains unsupported fields: {joined}"
        )


def _normalize_name(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{context} must be a non-empty string")
    return value.strip()


def _normalize_shape(value: Any, *, context: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigValidationError(
            f"{context} must contain at least one dimension"
        )
    shape = tuple(int(axis) for axis in value)
    if any(axis <= 0 for axis in shape):
        raise ConfigValidationError(f"{context} dimensions must be positive")
    return shape


def _normalize_storage(value: Any) -> str:
    storage = _normalize_name(value, context="storage").lower()
    if storage not in {"cpu", "gpu"}:
        raise ConfigValidationError(
            f"storage must be either 'cpu' or 'gpu', got {storage!r}"
        )
    return storage


def _normalize_names(value: Any, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigValidationError(
            f"{context} must contain at least one item"
        )
    return tuple(_normalize_name(item, context=context) for item in value)


def _normalize_parameters(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigValidationError(f"{context} must be a mapping")
    return dict(value)


def _normalize_auxiliary_bindings(
    value: Any,
    *,
    context: str,
) -> tuple[AuxiliaryBinding, ...]:
    if value is None or value == () or value == []:
        return ()
    if isinstance(value, Mapping):
        return tuple(
            AuxiliaryBinding(
                alias=_normalize_name(alias, context=f"{context} alias"),
                name=_normalize_name(
                    stream_name,
                    context=f"{context} stream name",
                ),
            )
            for alias, stream_name in value.items()
        )
    return tuple(
        AuxiliaryBinding(alias=item, name=item)
        for item in _normalize_names(value, context=context)
    )


def _normalize_positive_float(value: Any, *, context: str) -> float:
    try:
        normalized = float(value)
    except Exception as exc:
        raise ConfigValidationError(f"{context} must be a number") from exc
    if normalized <= 0.0:
        raise ConfigValidationError(f"{context} must be positive")
    return normalized


@dataclass(frozen=True)
class SharedMemoryConfig:
    """Configuration for one named shared-memory resource.

    A shared-memory record defines the storage backend, tensor shape, and dtype
    for one named stream in the pipeline graph. GPU streams may additionally
    declare a CUDA device and an optional CPU mirror for host-side readers.
    """

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    storage: str = "cpu"
    gpu_device: str | None = None
    cpu_mirror: bool | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SharedMemoryConfig":
        """Build a normalized shared-memory configuration from a mapping."""
        data = _expect_mapping(raw, context="shared_memory entry")
        _reject_unexpected_keys(
            data,
            context="shared_memory entry",
            allowed={
                "name",
                "shape",
                "dtype",
                "storage",
                "gpu_device",
                "cpu_mirror",
            },
        )
        name = _normalize_name(data.get("name"), context="shared memory name")
        shape = _normalize_shape(
            data.get("shape"), context=f"shape for {name}"
        )
        try:
            dtype = np.dtype(data.get("dtype"))
        except Exception as exc:
            raise ConfigValidationError(
                f"invalid dtype for shared memory {name!r}: {data.get('dtype')!r}"
            ) from exc
        storage = _normalize_storage(data.get("storage", "cpu"))
        gpu_device = data.get("gpu_device")
        if gpu_device is not None:
            gpu_device = _normalize_name(
                gpu_device,
                context=f"gpu_device for shared memory {name}",
            )
        cpu_mirror = data.get("cpu_mirror")
        if cpu_mirror is not None and not isinstance(cpu_mirror, bool):
            raise ConfigValidationError(
                f"cpu_mirror for shared memory {name!r} must be boolean"
            )
        return cls(
            name=name,
            shape=shape,
            dtype=dtype,
            storage=storage,
            gpu_device=gpu_device,
            cpu_mirror=cpu_mirror,
        )

    def __post_init__(self) -> None:
        """Validate storage-specific fields after normalization."""
        if self.storage == "cpu" and self.gpu_device is not None:
            raise ConfigValidationError(
                f"CPU shared memory {self.name!r} cannot declare gpu_device"
            )
        if self.storage == "gpu" and self.gpu_device is None:
            raise ConfigValidationError(
                f"GPU shared memory {self.name!r} requires gpu_device"
            )


@dataclass(frozen=True)
class AuxiliaryBinding:
    """Bind one kernel-local auxiliary alias to a shared-memory stream."""

    alias: str
    name: str


@dataclass(frozen=True)
class KernelConfig:
    """Configuration for one compute kernel in the pipeline.

    Each kernel consumes one trigger input stream, may read zero or more
    auxiliary streams, and writes one output stream. The `kind` field resolves
    through the active :class:`shmpipeline.registry.KernelRegistry`.
    """

    name: str
    kind: str
    input: str
    output: str
    auxiliary: tuple[AuxiliaryBinding, ...] = field(default_factory=tuple)
    operation: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    read_timeout: float = 1.0
    pause_sleep: float = 0.01

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "KernelConfig":
        """Build a normalized kernel configuration from a mapping."""
        data = _expect_mapping(raw, context="kernel entry")
        if "inputs" in data or "outputs" in data:
            raise ConfigValidationError(
                "kernel entry must use 'input', 'output', and optional "
                "'auxiliary'; 'inputs' and 'outputs' are not supported"
            )
        _reject_unexpected_keys(
            data,
            context="kernel entry",
            allowed={
                "name",
                "kind",
                "input",
                "output",
                "auxiliary",
                "operation",
                "parameters",
                "read_timeout",
                "pause_sleep",
            },
        )
        name = _normalize_name(data.get("name"), context="kernel name")
        kind = _normalize_name(
            data.get("kind"), context=f"kind for kernel {name}"
        )
        input_name = data.get("input")
        output_name = data.get("output")
        auxiliary_raw = data.get("auxiliary", ())
        input_name = _normalize_name(input_name, context=f"input for {name}")
        output_name = _normalize_name(
            output_name, context=f"output for {name}"
        )
        auxiliary = _normalize_auxiliary_bindings(
            auxiliary_raw,
            context=f"auxiliary for {name}",
        )
        operation = data.get("operation")
        if operation is not None:
            operation = _normalize_name(
                operation, context=f"operation for {name}"
            )
        parameters = _normalize_parameters(
            data.get("parameters", {}),
            context=f"parameters for kernel {name!r}",
        )
        read_timeout = _normalize_positive_float(
            data.get("read_timeout", 1.0),
            context=f"read_timeout for kernel {name!r}",
        )
        pause_sleep = _normalize_positive_float(
            data.get("pause_sleep", 0.01),
            context=f"pause_sleep for kernel {name!r}",
        )
        return cls(
            name=name,
            kind=kind,
            input=input_name,
            output=output_name,
            auxiliary=tuple(auxiliary),
            operation=operation,
            parameters=dict(parameters),
            read_timeout=read_timeout,
            pause_sleep=pause_sleep,
        )

    def __post_init__(self) -> None:
        """Validate basic kernel wiring constraints."""
        if self.input == self.output:
            raise ConfigValidationError(
                f"kernel {self.name!r} cannot use the same shared memory for "
                "both input and output"
            )
        auxiliary_names = set(self.auxiliary_names)
        if len(auxiliary_names) != len(self.auxiliary_names):
            raise ConfigValidationError(
                f"kernel {self.name!r} cannot reuse the same auxiliary stream more than once"
            )
        auxiliary_aliases = [binding.alias for binding in self.auxiliary]
        if len(set(auxiliary_aliases)) != len(auxiliary_aliases):
            raise ConfigValidationError(
                f"kernel {self.name!r} cannot reuse the same auxiliary alias more than once"
            )
        if self.input in auxiliary_names:
            raise ConfigValidationError(
                f"kernel {self.name!r} cannot reuse the trigger input as auxiliary"
            )
        if self.output in auxiliary_names:
            raise ConfigValidationError(
                f"kernel {self.name!r} cannot reuse the output as auxiliary"
            )

    @property
    def all_inputs(self) -> tuple[str, ...]:
        """Return the trigger input followed by ordered auxiliary streams."""
        return (self.input, *self.auxiliary_names)

    @property
    def auxiliary_names(self) -> tuple[str, ...]:
        """Return auxiliary shared-memory names in config order."""
        return tuple(binding.name for binding in self.auxiliary)

    @property
    def auxiliary_aliases(self) -> tuple[str, ...]:
        """Return auxiliary aliases in config order."""
        return tuple(binding.alias for binding in self.auxiliary)

    @property
    def auxiliary_by_alias(self) -> dict[str, str]:
        """Return auxiliary bindings keyed by expression alias."""
        return {binding.alias: binding.name for binding in self.auxiliary}


@dataclass(frozen=True)
class SourceConfig:
    """Configuration for one manager-owned source plugin.

    Sources write payloads into one shared-memory stream and are discovered
    through the active :class:`shmpipeline.registry.KernelRegistry`.
    """

    name: str
    kind: str
    stream: str
    auxiliary: tuple[AuxiliaryBinding, ...] = field(default_factory=tuple)
    parameters: dict[str, Any] = field(default_factory=dict)
    poll_interval: float = 0.01

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SourceConfig":
        """Build a normalized source configuration from a mapping."""
        data = _expect_mapping(raw, context="source entry")
        _reject_unexpected_keys(
            data,
            context="source entry",
            allowed={
                "name",
                "kind",
                "stream",
                "auxiliary",
                "parameters",
                "poll_interval",
            },
        )
        name = _normalize_name(data.get("name"), context="source name")
        kind = _normalize_name(
            data.get("kind"), context=f"kind for source {name}"
        )
        stream = _normalize_name(
            data.get("stream"), context=f"stream for source {name}"
        )
        auxiliary = _normalize_auxiliary_bindings(
            data.get("auxiliary", ()),
            context=f"auxiliary for {name}",
        )
        parameters = _normalize_parameters(
            data.get("parameters", {}),
            context=f"parameters for source {name!r}",
        )
        poll_interval = _normalize_positive_float(
            data.get("poll_interval", 0.01),
            context=f"poll_interval for source {name!r}",
        )
        return cls(
            name=name,
            kind=kind,
            stream=stream,
            auxiliary=auxiliary,
            parameters=parameters,
            poll_interval=poll_interval,
        )

    @property
    def auxiliary_names(self) -> tuple[str, ...]:
        """Return auxiliary shared-memory names in config order."""
        return tuple(binding.name for binding in self.auxiliary)

    @property
    def auxiliary_aliases(self) -> tuple[str, ...]:
        """Return auxiliary aliases in config order."""
        return tuple(binding.alias for binding in self.auxiliary)

    @property
    def auxiliary_by_alias(self) -> dict[str, str]:
        """Return auxiliary bindings keyed by alias."""
        return {binding.alias: binding.name for binding in self.auxiliary}


@dataclass(frozen=True)
class SinkConfig:
    """Configuration for one manager-owned sink plugin.

    Sinks consume payloads from one shared-memory stream and are discovered
    through the active :class:`shmpipeline.registry.KernelRegistry`.
    """

    name: str
    kind: str
    stream: str
    auxiliary: tuple[AuxiliaryBinding, ...] = field(default_factory=tuple)
    parameters: dict[str, Any] = field(default_factory=dict)
    read_timeout: float = 1.0
    pause_sleep: float = 0.01

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SinkConfig":
        """Build a normalized sink configuration from a mapping."""
        data = _expect_mapping(raw, context="sink entry")
        _reject_unexpected_keys(
            data,
            context="sink entry",
            allowed={
                "name",
                "kind",
                "stream",
                "auxiliary",
                "parameters",
                "read_timeout",
                "pause_sleep",
            },
        )
        name = _normalize_name(data.get("name"), context="sink name")
        kind = _normalize_name(
            data.get("kind"), context=f"kind for sink {name}"
        )
        stream = _normalize_name(
            data.get("stream"), context=f"stream for sink {name}"
        )
        auxiliary = _normalize_auxiliary_bindings(
            data.get("auxiliary", ()),
            context=f"auxiliary for {name}",
        )
        parameters = _normalize_parameters(
            data.get("parameters", {}),
            context=f"parameters for sink {name!r}",
        )
        read_timeout = _normalize_positive_float(
            data.get("read_timeout", 1.0),
            context=f"read_timeout for sink {name!r}",
        )
        pause_sleep = _normalize_positive_float(
            data.get("pause_sleep", 0.01),
            context=f"pause_sleep for sink {name!r}",
        )
        return cls(
            name=name,
            kind=kind,
            stream=stream,
            auxiliary=auxiliary,
            parameters=parameters,
            read_timeout=read_timeout,
            pause_sleep=pause_sleep,
        )

    @property
    def auxiliary_names(self) -> tuple[str, ...]:
        """Return auxiliary shared-memory names in config order."""
        return tuple(binding.name for binding in self.auxiliary)

    @property
    def auxiliary_aliases(self) -> tuple[str, ...]:
        """Return auxiliary aliases in config order."""
        return tuple(binding.alias for binding in self.auxiliary)

    @property
    def auxiliary_by_alias(self) -> dict[str, str]:
        """Return auxiliary bindings keyed by alias."""
        return {binding.alias: binding.name for binding in self.auxiliary}


@dataclass(frozen=True)
class PipelineConfig:
    """Complete pipeline configuration loaded from YAML or a mapping.

    This is the primary configuration object used by the CLI, GUI, and
    :class:`shmpipeline.manager.PipelineManager`. It groups the named stream
    definitions and the ordered kernel stages that consume them.
    """

    shared_memory: tuple[SharedMemoryConfig, ...]
    kernels: tuple[KernelConfig, ...]
    sources: tuple[SourceConfig, ...] = field(default_factory=tuple)
    sinks: tuple[SinkConfig, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PipelineConfig":
        """Create a pipeline configuration from a plain mapping."""
        data = _expect_mapping(raw, context="pipeline config")
        _reject_unexpected_keys(
            data,
            context="pipeline config",
            allowed={"shared_memory", "kernels", "sources", "sinks"},
        )
        shared_memory_raw = data.get("shared_memory")
        kernels_raw = data.get("kernels", [])
        sources_raw = data.get("sources", [])
        sinks_raw = data.get("sinks", [])
        if not isinstance(shared_memory_raw, list) or not shared_memory_raw:
            raise ConfigValidationError(
                "pipeline config must define a non-empty shared_memory list"
            )
        if not isinstance(kernels_raw, list):
            raise ConfigValidationError(
                "pipeline config field 'kernels' must be a list"
            )
        if not isinstance(sources_raw, list):
            raise ConfigValidationError(
                "pipeline config field 'sources' must be a list"
            )
        if not isinstance(sinks_raw, list):
            raise ConfigValidationError(
                "pipeline config field 'sinks' must be a list"
            )
        if not kernels_raw and not sources_raw and not sinks_raw:
            raise ConfigValidationError(
                "pipeline config must define at least one kernel, source, or sink"
            )
        return cls(
            shared_memory=tuple(
                SharedMemoryConfig.from_dict(item)
                for item in shared_memory_raw
            ),
            kernels=tuple(
                KernelConfig.from_dict(item) for item in kernels_raw
            ),
            sources=tuple(
                SourceConfig.from_dict(item) for item in sources_raw
            ),
            sinks=tuple(SinkConfig.from_dict(item) for item in sinks_raw),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load a pipeline configuration from a YAML file."""
        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    def __post_init__(self) -> None:
        """Validate name uniqueness and shared-memory references."""
        shared_names = [item.name for item in self.shared_memory]
        if len(shared_names) != len(set(shared_names)):
            raise ConfigValidationError("shared memory names must be unique")

        kernel_names = [item.name for item in self.kernels]
        if len(kernel_names) != len(set(kernel_names)):
            raise ConfigValidationError("kernel names must be unique")

        source_names = [item.name for item in self.sources]
        if len(source_names) != len(set(source_names)):
            raise ConfigValidationError("source names must be unique")

        sink_names = [item.name for item in self.sinks]
        if len(sink_names) != len(set(sink_names)):
            raise ConfigValidationError("sink names must be unique")

        node_names = [*kernel_names, *source_names, *sink_names]
        if len(node_names) != len(set(node_names)):
            duplicates = sorted(
                {name for name in node_names if node_names.count(name) > 1}
            )
            raise ConfigValidationError(
                "pipeline node names must be unique across kernels, "
                "sources, and sinks: "
                + ", ".join(repr(name) for name in duplicates)
            )

        available = set(shared_names)
        for kernel in self.kernels:
            missing = [
                name
                for name in (*kernel.all_inputs, kernel.output)
                if name not in available
            ]
            if missing:
                missing_list = ", ".join(sorted(set(missing)))
                raise ConfigValidationError(
                    f"kernel {kernel.name!r} references undefined shared "
                    f"memory: {missing_list}"
                )
        for source in self.sources:
            missing = [
                name
                for name in (source.stream, *source.auxiliary_names)
                if name not in available
            ]
            if missing:
                missing_list = ", ".join(sorted(set(missing)))
                raise ConfigValidationError(
                    f"source {source.name!r} references undefined shared "
                    f"memory: {missing_list}"
                )
        for sink in self.sinks:
            missing = [
                name
                for name in (sink.stream, *sink.auxiliary_names)
                if name not in available
            ]
            if missing:
                missing_list = ", ".join(sorted(set(missing)))
                raise ConfigValidationError(
                    f"sink {sink.name!r} references undefined shared "
                    f"memory: {missing_list}"
                )

    @property
    def shared_memory_by_name(self) -> dict[str, SharedMemoryConfig]:
        """Return shared-memory definitions keyed by name."""
        return {item.name: item for item in self.shared_memory}
