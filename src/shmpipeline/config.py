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
        raise ConfigValidationError(f"{context} must contain at least one item")
    return tuple(_normalize_name(item, context=context) for item in value)


@dataclass(frozen=True)
class SharedMemoryConfig:
    """Configuration for one named shared-memory resource."""

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
        name = _normalize_name(data.get("name"), context="shared memory name")
        shape = _normalize_shape(data.get("shape"), context=f"shape for {name}")
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
class KernelConfig:
    """Configuration for one compute kernel in the pipeline."""

    name: str
    kind: str
    input: str
    output: str
    auxiliary: tuple[str, ...] = field(default_factory=tuple)
    parameters: dict[str, Any] = field(default_factory=dict)
    read_timeout: float = 1.0
    pause_sleep: float = 0.01

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "KernelConfig":
        """Build a normalized kernel configuration from a mapping."""
        data = _expect_mapping(raw, context="kernel entry")
        name = _normalize_name(data.get("name"), context="kernel name")
        kind = _normalize_name(data.get("kind"), context=f"kind for kernel {name}")
        input_name = data.get("input")
        output_name = data.get("output")
        auxiliary_raw = data.get("auxiliary", ())
        if input_name is None and "inputs" in data:
            legacy_inputs = _normalize_names(
                data.get("inputs"),
                context=f"inputs for {name}",
            )
            input_name = legacy_inputs[0]
            auxiliary_raw = legacy_inputs[1:]
        if output_name is None and "outputs" in data:
            legacy_outputs = _normalize_names(
                data.get("outputs"),
                context=f"outputs for {name}",
            )
            if len(legacy_outputs) != 1:
                raise ConfigValidationError(
                    f"kernel {name!r} requires exactly one output"
                )
            output_name = legacy_outputs[0]
        input_name = _normalize_name(input_name, context=f"input for {name}")
        output_name = _normalize_name(output_name, context=f"output for {name}")
        if auxiliary_raw in (None, ()):  # normalize omitted auxiliary values
            auxiliary = ()
        else:
            auxiliary = _normalize_names(
                auxiliary_raw,
                context=f"auxiliary for {name}",
            )
        parameters = data.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ConfigValidationError(
                f"parameters for kernel {name!r} must be a mapping"
            )
        read_timeout = float(data.get("read_timeout", 1.0))
        pause_sleep = float(data.get("pause_sleep", 0.01))
        if read_timeout <= 0.0:
            raise ConfigValidationError(
                f"read_timeout for kernel {name!r} must be positive"
            )
        if pause_sleep <= 0.0:
            raise ConfigValidationError(
                f"pause_sleep for kernel {name!r} must be positive"
            )
        return cls(
            name=name,
            kind=kind,
            input=input_name,
            output=output_name,
            auxiliary=tuple(auxiliary),
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
        if self.input in set(self.auxiliary):
            raise ConfigValidationError(
                f"kernel {self.name!r} cannot reuse the trigger input as auxiliary"
            )
        if self.output in set(self.auxiliary):
            raise ConfigValidationError(
                f"kernel {self.name!r} cannot reuse the output as auxiliary"
            )

    @property
    def all_inputs(self) -> tuple[str, ...]:
        """Return the trigger input followed by ordered auxiliary streams."""
        return (self.input, *self.auxiliary)


@dataclass(frozen=True)
class PipelineConfig:
    """Complete pipeline configuration loaded from YAML or a mapping."""

    shared_memory: tuple[SharedMemoryConfig, ...]
    kernels: tuple[KernelConfig, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PipelineConfig":
        """Create a pipeline configuration from a plain mapping."""
        data = _expect_mapping(raw, context="pipeline config")
        shared_memory_raw = data.get("shared_memory")
        kernels_raw = data.get("kernels")
        if not isinstance(shared_memory_raw, list) or not shared_memory_raw:
            raise ConfigValidationError(
                "pipeline config must define a non-empty shared_memory list"
            )
        if not isinstance(kernels_raw, list) or not kernels_raw:
            raise ConfigValidationError(
                "pipeline config must define a non-empty kernels list"
            )
        return cls(
            shared_memory=tuple(
                SharedMemoryConfig.from_dict(item) for item in shared_memory_raw
            ),
            kernels=tuple(KernelConfig.from_dict(item) for item in kernels_raw),
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

    @property
    def shared_memory_by_name(self) -> dict[str, SharedMemoryConfig]:
        """Return shared-memory definitions keyed by name."""
        return {item.name: item for item in self.shared_memory}