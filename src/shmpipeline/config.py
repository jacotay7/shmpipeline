"""Immutable configuration models for shared-memory pipelines."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from shmpipeline.errors import ConfigValidationError


class _LineAnnotatedDict(dict):
    """A ``dict`` that remembers the 1-based YAML line it was parsed from."""

    __line__: int | None = None


class _LineMarkLoader(yaml.SafeLoader):
    """YAML loader that annotates every mapping with its source line."""


def _construct_line_mapping(loader: _LineMarkLoader, node: yaml.Node):
    mapping = _LineAnnotatedDict(
        yaml.SafeLoader.construct_mapping(loader, node, deep=True)
    )
    mapping.__line__ = node.start_mark.line + 1
    return mapping


_LineMarkLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_line_mapping,
)


def _index_config_lines(raw: Any) -> dict[str, int]:
    """Map each named config entry to the YAML line it was declared on."""
    index: dict[str, int] = {}
    if not isinstance(raw, Mapping):
        return index
    for section in ("shared_memory", "kernels", "sources", "sinks"):
        entries = raw.get(section)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            name = entry.get("name")
            line = getattr(entry, "__line__", None)
            if isinstance(name, str) and line is not None:
                index.setdefault(name.strip(), line)
    return index


def _augment_error_with_line(message: str, raw: Any, path: Path) -> str:
    """Append the source file and (when resolvable) line to an error message."""
    index = _index_config_lines(raw)
    for token in re.findall(r"'([^']*)'", message):
        line = index.get(token.strip())
        if line is not None:
            return f"{message} (in {path}, line {line})"
    return f"{message} (in {path})"


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


def _normalize_optional_positive_float(
    value: Any, *, context: str
) -> float | None:
    """Return a positive float, or ``None`` when the value is unset."""
    if value is None:
        return None
    return _normalize_positive_float(value, context=context)


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
    notify: bool | None = None
    mode: str = "create_or_attach"
    initial: dict[str, Any] | None = None

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
                "notify",
                "mode",
                "initial",
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
        notify = data.get("notify")
        if notify is not None and not isinstance(notify, bool):
            raise ConfigValidationError(
                f"notify for shared memory {name!r} must be boolean"
            )
        mode = _normalize_name(
            data.get("mode", "create_or_attach"),
            context=f"mode for shared memory {name}",
        ).lower()
        if mode not in {"create", "attach", "create_or_attach", "replace"}:
            raise ConfigValidationError(
                f"mode for shared memory {name!r} must be one of "
                "'create', 'attach', 'create_or_attach', or 'replace'"
            )
        initial_raw = data.get("initial")
        initial = None
        if initial_raw is not None:
            initial = _normalize_parameters(
                initial_raw,
                context=f"initial value for shared memory {name!r}",
            )
            pattern = _normalize_name(
                initial.get("pattern"),
                context=f"initial pattern for shared memory {name!r}",
            ).lower()
            if pattern not in {"constant", "normal", "values", "identity"}:
                raise ConfigValidationError(
                    f"initial pattern for shared memory {name!r} must be "
                    "'constant', 'normal', 'values', or 'identity'"
                )
            initial["pattern"] = pattern
            allowed_initial = {
                "constant": {"pattern", "value"},
                "normal": {"pattern", "seed", "mean", "std"},
                "values": {"pattern", "values"},
                "identity": {"pattern", "scale"},
            }[pattern]
            unexpected = sorted(set(initial) - allowed_initial)
            if unexpected:
                raise ConfigValidationError(
                    f"initial value for shared memory {name!r} contains "
                    f"unsupported fields: {', '.join(unexpected)}"
                )
            if pattern == "constant" and not isinstance(
                initial.get("value"), (int, float, bool, complex)
            ):
                raise ConfigValidationError(
                    f"constant initializer for shared memory {name!r} "
                    "requires numeric 'value'"
                )
            if pattern == "normal":
                std = initial.get("std", 1.0)
                if not isinstance(std, (int, float)) or std < 0.0:
                    raise ConfigValidationError(
                        f"normal initializer for shared memory {name!r} "
                        "requires non-negative numeric 'std'"
                    )
                seed = initial.get("seed", 0)
                if not isinstance(seed, int) or isinstance(seed, bool):
                    raise ConfigValidationError(
                        f"normal initializer for shared memory {name!r} "
                        "requires integer 'seed'"
                    )
            if pattern == "values" and "values" not in initial:
                raise ConfigValidationError(
                    f"values initializer for shared memory {name!r} "
                    "requires 'values'"
                )
            if pattern == "identity" and len(shape) != 2:
                raise ConfigValidationError(
                    f"identity initializer for shared memory {name!r} "
                    "requires a 2D shape"
                )
        return cls(
            name=name,
            shape=shape,
            dtype=dtype,
            storage=storage,
            gpu_device=gpu_device,
            cpu_mirror=cpu_mirror,
            notify=notify,
            mode=mode,
            initial=initial,
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


_SYNC_MODES = ("count", "matching_frame_id")
_SYNC_ON_SKEW = ("drop_older",)


@dataclass(frozen=True)
class SynchronizationConfig:
    """Cross-input synchronization policy for a multi-input kernel.

    ``mode`` ``"count"`` (default) fires on publication counts alone. Mode
    ``"matching_frame_id"`` additionally requires every trigger to carry the
    same propagated ``frame_id`` token before the kernel runs, so a fan-in only
    combines inputs from the same hardware generation. When branches skew, the
    ``on_skew`` policy (``drop_older``) advances past the lagging generations,
    bounded by ``max_skew_generations`` and ``max_wait_s`` so a stalled branch
    cannot block forever.
    """

    mode: str = "count"
    on_skew: str = "drop_older"
    max_skew_generations: int = 16
    max_wait_s: float | None = None

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any], *, context: str
    ) -> "SynchronizationConfig":
        """Build and validate a synchronization policy from a mapping."""
        data = _expect_mapping(raw, context=context)
        _reject_unexpected_keys(
            data,
            context=context,
            allowed={
                "mode",
                "on_skew",
                "max_skew_generations",
                "max_wait_s",
                "timeout",
            },
        )
        mode = str(data.get("mode", "count")).strip().lower()
        if mode not in _SYNC_MODES:
            raise ConfigValidationError(
                f"{context}: mode must be one of {_SYNC_MODES}"
            )
        on_skew = str(data.get("on_skew", "drop_older")).strip().lower()
        if on_skew not in _SYNC_ON_SKEW:
            raise ConfigValidationError(
                f"{context}: on_skew must be one of {_SYNC_ON_SKEW}"
            )
        max_skew = data.get("max_skew_generations", 16)
        if not isinstance(max_skew, int) or max_skew < 1:
            raise ConfigValidationError(
                f"{context}: max_skew_generations must be a positive integer"
            )
        # 'timeout' is accepted as an alias for max_wait_s to match the plan.
        max_wait = data.get("max_wait_s", data.get("timeout"))
        max_wait_s = _normalize_optional_positive_float(
            max_wait, context=f"{context} max_wait_s"
        )
        return cls(
            mode=mode,
            on_skew=on_skew,
            max_skew_generations=int(max_skew),
            max_wait_s=max_wait_s,
        )


@dataclass(frozen=True)
class KernelConfig:
    """Configuration for one compute kernel in the pipeline.

    Each kernel consumes one or more trigger input streams, may read zero or more
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
    poll_interval: float = 1e-5
    outputs: tuple[str, ...] = field(default_factory=tuple)
    inputs: tuple[str, ...] = field(default_factory=tuple)
    trigger_policy: str = "any_new"
    synchronization: SynchronizationConfig | None = None
    propagate_frame_id: bool = False

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "KernelConfig":
        """Build a normalized kernel configuration from a mapping."""
        data = _expect_mapping(raw, context="kernel entry")
        if "input" in data and "inputs" in data:
            raise ConfigValidationError(
                "kernel entry must use either 'input' or 'inputs', not both"
            )
        if "output" in data and "outputs" in data:
            raise ConfigValidationError(
                "kernel entry must use either 'output' or 'outputs', not both"
            )
        _reject_unexpected_keys(
            data,
            context="kernel entry",
            allowed={
                "name",
                "kind",
                "input",
                "inputs",
                "output",
                "outputs",
                "auxiliary",
                "operation",
                "parameters",
                "read_timeout",
                "pause_sleep",
                "poll_interval",
                "trigger_policy",
                "synchronization",
                "propagate_frame_id",
            },
        )
        name = _normalize_name(data.get("name"), context="kernel name")
        kind = _normalize_name(
            data.get("kind"), context=f"kind for kernel {name}"
        )
        if "inputs" in data:
            inputs = _normalize_names(
                data.get("inputs"), context=f"inputs for {name}"
            )
        else:
            inputs = (
                _normalize_name(
                    data.get("input"), context=f"input for {name}"
                ),
            )
        input_name = inputs[0]
        auxiliary_raw = data.get("auxiliary", ())
        if "outputs" in data:
            outputs = _normalize_names(
                data.get("outputs"), context=f"outputs for {name}"
            )
        else:
            outputs = (
                _normalize_name(
                    data.get("output"), context=f"output for {name}"
                ),
            )
        output_name = outputs[0]
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
        poll_interval = _normalize_positive_float(
            data.get("poll_interval", 1e-5),
            context=f"poll_interval for kernel {name!r}",
        )
        trigger_policy = _normalize_name(
            data.get(
                "trigger_policy",
                "all_new" if len(inputs) > 1 else "any_new",
            ),
            context=f"trigger_policy for kernel {name!r}",
        ).lower()
        if trigger_policy not in {"any_new", "all_new"}:
            raise ConfigValidationError(
                f"trigger_policy for kernel {name!r} must be either "
                "'any_new' or 'all_new'"
            )
        synchronization = None
        if "synchronization" in data:
            synchronization = SynchronizationConfig.from_dict(
                data["synchronization"],
                context=f"synchronization for kernel {name!r}",
            )
        propagate_frame_id = bool(data.get("propagate_frame_id", False))
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
            poll_interval=poll_interval,
            outputs=tuple(outputs),
            inputs=tuple(inputs),
            trigger_policy=trigger_policy,
            synchronization=synchronization,
            propagate_frame_id=propagate_frame_id,
        )

    def __post_init__(self) -> None:
        """Validate basic kernel wiring constraints."""
        all_outputs = self.all_outputs
        trigger_inputs = self.trigger_inputs
        if len(set(trigger_inputs)) != len(trigger_inputs):
            raise ConfigValidationError(
                f"kernel {self.name!r} cannot declare the same trigger input "
                "stream more than once"
            )
        if (
            self.synchronization is not None
            and self.synchronization.mode == "matching_frame_id"
            and self.trigger_policy != "all_new"
        ):
            raise ConfigValidationError(
                f"kernel {self.name!r} synchronization mode "
                "'matching_frame_id' requires trigger_policy 'all_new'"
            )
        if len(set(all_outputs)) != len(all_outputs):
            raise ConfigValidationError(
                f"kernel {self.name!r} cannot declare the same output stream "
                "more than once"
            )
        if any(name in all_outputs for name in trigger_inputs):
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
        if any(name in auxiliary_names for name in trigger_inputs):
            raise ConfigValidationError(
                f"kernel {self.name!r} cannot reuse the trigger input as auxiliary"
            )
        if any(output in auxiliary_names for output in all_outputs):
            raise ConfigValidationError(
                f"kernel {self.name!r} cannot reuse the output as auxiliary"
            )

    @property
    def all_inputs(self) -> tuple[str, ...]:
        """Return trigger inputs followed by ordered auxiliary streams."""
        return (*self.trigger_inputs, *self.auxiliary_names)

    @property
    def trigger_inputs(self) -> tuple[str, ...]:
        """Return every dynamic trigger stream in declaration order."""
        return self.inputs if self.inputs else (self.input,)

    @property
    def requires_matching_frame_id(self) -> bool:
        """Return whether this kernel gates on matching frame-id tokens."""
        return (
            self.synchronization is not None
            and self.synchronization.mode == "matching_frame_id"
        )

    @property
    def tracks_frame_id(self) -> bool:
        """Return whether the worker reads/propagates frame-id tokens.

        Gated so the default hot path never touches the token metadata.
        """
        return self.propagate_frame_id or self.requires_matching_frame_id

    @property
    def all_outputs(self) -> tuple[str, ...]:
        """Return every output stream in declaration order.

        Single-output kernels report ``(output,)``; multi-output kernels
        configured with ``outputs:`` report the full ordered tuple.  The
        primary :attr:`output` is always ``all_outputs[0]``.
        """
        return self.outputs if self.outputs else (self.output,)

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
    read_timeout: float | None = None
    streams: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SourceConfig":
        """Build a normalized source configuration from a mapping."""
        data = _expect_mapping(raw, context="source entry")
        if "stream" in data and "streams" in data:
            raise ConfigValidationError(
                "source entry must use either 'stream' or 'streams', not both"
            )
        _reject_unexpected_keys(
            data,
            context="source entry",
            allowed={
                "name",
                "kind",
                "stream",
                "streams",
                "auxiliary",
                "parameters",
                "poll_interval",
                "read_timeout",
            },
        )
        name = _normalize_name(data.get("name"), context="source name")
        kind = _normalize_name(
            data.get("kind"), context=f"kind for source {name}"
        )
        if "streams" in data:
            streams = _normalize_names(
                data.get("streams"), context=f"streams for {name}"
            )
        else:
            streams = (
                _normalize_name(
                    data.get("stream"), context=f"stream for source {name}"
                ),
            )
        stream = streams[0]
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
        read_timeout = _normalize_optional_positive_float(
            data.get("read_timeout"),
            context=f"read_timeout for source {name!r}",
        )
        return cls(
            name=name,
            kind=kind,
            stream=stream,
            auxiliary=auxiliary,
            parameters=parameters,
            poll_interval=poll_interval,
            read_timeout=read_timeout,
            streams=tuple(streams),
        )

    def __post_init__(self) -> None:
        """Validate multi-output stream wiring."""
        output_streams = self.output_streams
        if len(set(output_streams)) != len(output_streams):
            raise ConfigValidationError(
                f"source {self.name!r} cannot declare the same output stream "
                "more than once"
            )

    @property
    def output_streams(self) -> tuple[str, ...]:
        """Return every stream this source publishes, in declaration order."""
        return self.streams if self.streams else (self.stream,)

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
    consume_timeout: float | None = None

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
                "consume_timeout",
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
        consume_timeout = _normalize_optional_positive_float(
            data.get("consume_timeout"),
            context=f"consume_timeout for sink {name!r}",
        )
        return cls(
            name=name,
            kind=kind,
            stream=stream,
            auxiliary=auxiliary,
            parameters=parameters,
            read_timeout=read_timeout,
            pause_sleep=pause_sleep,
            consume_timeout=consume_timeout,
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
        raw = yaml.load(
            config_path.read_text(encoding="utf-8"), Loader=_LineMarkLoader
        )
        try:
            return cls.from_dict(raw)
        except ConfigValidationError as exc:
            raise ConfigValidationError(
                _augment_error_with_line(str(exc), raw, config_path)
            ) from exc

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
                for name in (*kernel.all_inputs, *kernel.all_outputs)
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
