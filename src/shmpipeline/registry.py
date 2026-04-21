"""Kernel registry used by the manager and worker runtime."""

from __future__ import annotations

from functools import partial
from importlib.metadata import entry_points
from importlib.util import find_spec
from typing import Callable, Mapping

from shmpipeline.config import (
    KernelConfig,
    SharedMemoryConfig,
    SinkConfig,
    SourceConfig,
)
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernel import Kernel, KernelContext
from shmpipeline.kernels.cpu import (
    AddConstantCpuKernel,
    AffineTransformCpuKernel,
    CopyCpuKernel,
    CustomOperationCpuKernel,
    ElementwiseAddCpuKernel,
    ElementwiseDivideCpuKernel,
    ElementwiseMultiplyCpuKernel,
    ElementwiseSubtractCpuKernel,
    FlattenCpuKernel,
    LeakyIntegratorCpuKernel,
    RaiseErrorCpuKernel,
    ScaleCpuKernel,
    ScaleOffsetCpuKernel,
    ShackHartmannCentroidCpuKernel,
)
from shmpipeline.sink import Sink, SinkContext
from shmpipeline.source import Source, SourceContext

_TORCH_AVAILABLE = find_spec("torch") is not None

if _TORCH_AVAILABLE:
    _DEFAULT_GPU_KINDS = (
        "gpu.add_constant",
        "gpu.affine_transform",
        "gpu.copy",
        "gpu.custom_operation",
        "gpu.elementwise_add",
        "gpu.elementwise_divide",
        "gpu.elementwise_multiply",
        "gpu.elementwise_subtract",
        "gpu.flatten",
        "gpu.leaky_integrator",
        "gpu.raise_error",
        "gpu.scale",
        "gpu.scale_offset",
        "gpu.shack_hartmann_centroid",
    )


def _load_default_gpu_kernel(kind: str) -> type[Kernel]:
    from shmpipeline.kernels.gpu import (
        AddConstantGpuKernel,
        AffineTransformGpuKernel,
        CopyGpuKernel,
        CustomOperationGpuKernel,
        ElementwiseAddGpuKernel,
        ElementwiseDivideGpuKernel,
        ElementwiseMultiplyGpuKernel,
        ElementwiseSubtractGpuKernel,
        FlattenGpuKernel,
        LeakyIntegratorGpuKernel,
        RaiseErrorGpuKernel,
        ScaleGpuKernel,
        ScaleOffsetGpuKernel,
        ShackHartmannCentroidGpuKernel,
    )

    return {
        AddConstantGpuKernel.kind: AddConstantGpuKernel,
        AffineTransformGpuKernel.kind: AffineTransformGpuKernel,
        CopyGpuKernel.kind: CopyGpuKernel,
        CustomOperationGpuKernel.kind: CustomOperationGpuKernel,
        ElementwiseAddGpuKernel.kind: ElementwiseAddGpuKernel,
        ElementwiseDivideGpuKernel.kind: ElementwiseDivideGpuKernel,
        ElementwiseMultiplyGpuKernel.kind: ElementwiseMultiplyGpuKernel,
        ElementwiseSubtractGpuKernel.kind: ElementwiseSubtractGpuKernel,
        FlattenGpuKernel.kind: FlattenGpuKernel,
        LeakyIntegratorGpuKernel.kind: LeakyIntegratorGpuKernel,
        RaiseErrorGpuKernel.kind: RaiseErrorGpuKernel,
        ScaleGpuKernel.kind: ScaleGpuKernel,
        ScaleOffsetGpuKernel.kind: ScaleOffsetGpuKernel,
        ShackHartmannCentroidGpuKernel.kind: ShackHartmannCentroidGpuKernel,
    }[kind]


def _entry_points_for_group(group: str):
    discovered = entry_points()
    if hasattr(discovered, "select"):
        return tuple(discovered.select(group=group))
    return tuple(discovered.get(group, ()))


def _load_entry_point_plugin(entry_point, *, expected_kind: str, validator):
    plugin_cls = entry_point.load()
    actual_kind = validator(plugin_cls)
    if actual_kind != expected_kind:
        raise ValueError(
            "entry point "
            f"{entry_point.group}:{entry_point.name} resolved to kind "
            f"{actual_kind!r}, expected {expected_kind!r}"
        )
    return plugin_cls


def _discover_entry_point_loaders(
    group: str,
    *,
    validator,
    existing_kinds: set[str],
) -> dict[str, Callable[[], object]]:
    loaders: dict[str, Callable[[], object]] = {}
    for entry_point in _entry_points_for_group(group):
        kind = entry_point.name.strip()
        if not kind or kind in existing_kinds or kind in loaders:
            continue
        loaders[kind] = partial(
            _load_entry_point_plugin,
            entry_point,
            expected_kind=kind,
            validator=validator,
        )
    return loaders


class KernelRegistry:
    """Resolve kernel kinds to implementation classes.

    Registries are the extension point for third-party kernels, sources, and
    sinks. The default registry contains the built-in CPU kernels, lazily loads
    GPU kernels when the optional torch dependency is available, and also picks
    up packaged plugins exposed through entry points.
    """

    def __init__(
        self,
        kernels: Mapping[str, type[Kernel]],
        lazy_kernels: Mapping[str, Callable[[], type[Kernel]]] | None = None,
        sources: Mapping[str, type[Source]] | None = None,
        lazy_sources: Mapping[str, Callable[[], type[Source]]] | None = None,
        sinks: Mapping[str, type[Sink]] | None = None,
        lazy_sinks: Mapping[str, Callable[[], type[Sink]]] | None = None,
    ) -> None:
        """Store static mappings of registered plugin implementations."""
        self._kernels = dict(kernels)
        self._lazy_kernels = dict(lazy_kernels or {})
        self._sources = dict(sources or {})
        self._lazy_sources = dict(lazy_sources or {})
        self._sinks = dict(sinks or {})
        self._lazy_sinks = dict(lazy_sinks or {})

    def get(self, kind: str) -> type[Kernel]:
        """Return the implementation class for a kernel kind."""
        kernel_cls = self._kernels.get(kind)
        if kernel_cls is not None:
            return kernel_cls
        loader = self._lazy_kernels.get(kind)
        if loader is None:
            raise ConfigValidationError(f"unknown kernel kind: {kind!r}")
        kernel_cls = loader()
        self._kernels[kind] = kernel_cls
        self._lazy_kernels.pop(kind, None)
        return kernel_cls

    def kinds(self) -> tuple[str, ...]:
        """Return registered kernel kinds in sorted order."""
        return tuple(sorted({*self._kernels, *self._lazy_kernels}))

    def source_kinds(self) -> tuple[str, ...]:
        """Return registered source kinds in sorted order."""
        return tuple(sorted({*self._sources, *self._lazy_sources}))

    def sink_kinds(self) -> tuple[str, ...]:
        """Return registered sink kinds in sorted order."""
        return tuple(sorted({*self._sinks, *self._lazy_sinks}))

    def register(
        self,
        kernel_cls: type[Kernel],
        *,
        replace: bool = False,
    ) -> None:
        """Register one kernel implementation class on this registry."""
        kind = _kernel_kind(kernel_cls)
        if not replace and (
            kind in self._kernels or kind in self._lazy_kernels
        ):
            raise ValueError(f"kernel kind {kind!r} is already registered")
        self._kernels[kind] = kernel_cls
        self._lazy_kernels.pop(kind, None)

    def get_source(self, kind: str) -> type[Source]:
        """Return the implementation class for a source kind."""
        source_cls = self._sources.get(kind)
        if source_cls is not None:
            return source_cls
        loader = self._lazy_sources.get(kind)
        if loader is None:
            raise ConfigValidationError(f"unknown source kind: {kind!r}")
        source_cls = loader()
        self._sources[kind] = source_cls
        self._lazy_sources.pop(kind, None)
        return source_cls

    def register_source(
        self,
        source_cls: type[Source],
        *,
        replace: bool = False,
    ) -> None:
        """Register one source implementation class on this registry."""
        kind = _source_kind(source_cls)
        if not replace and (
            kind in self._sources or kind in self._lazy_sources
        ):
            raise ValueError(f"source kind {kind!r} is already registered")
        self._sources[kind] = source_cls
        self._lazy_sources.pop(kind, None)

    def get_sink(self, kind: str) -> type[Sink]:
        """Return the implementation class for a sink kind."""
        sink_cls = self._sinks.get(kind)
        if sink_cls is not None:
            return sink_cls
        loader = self._lazy_sinks.get(kind)
        if loader is None:
            raise ConfigValidationError(f"unknown sink kind: {kind!r}")
        sink_cls = loader()
        self._sinks[kind] = sink_cls
        self._lazy_sinks.pop(kind, None)
        return sink_cls

    def register_sink(
        self,
        sink_cls: type[Sink],
        *,
        replace: bool = False,
    ) -> None:
        """Register one sink implementation class on this registry."""
        kind = _sink_kind(sink_cls)
        if not replace and (kind in self._sinks or kind in self._lazy_sinks):
            raise ValueError(f"sink kind {kind!r} is already registered")
        self._sinks[kind] = sink_cls
        self._lazy_sinks.pop(kind, None)

    def extended(
        self,
        *kernel_classes: type[Kernel],
        sources: tuple[type[Source], ...] = (),
        sinks: tuple[type[Sink], ...] = (),
        replace: bool = False,
    ) -> "KernelRegistry":
        """Return a new registry extended with additional plugin classes."""
        registry = KernelRegistry(
            self._kernels,
            self._lazy_kernels,
            self._sources,
            self._lazy_sources,
            self._sinks,
            self._lazy_sinks,
        )
        for kernel_cls in kernel_classes:
            registry.register(kernel_cls, replace=replace)
        for source_cls in sources:
            registry.register_source(source_cls, replace=replace)
        for sink_cls in sinks:
            registry.register_sink(sink_cls, replace=replace)
        return registry

    def extended_sources(
        self,
        *source_classes: type[Source],
        replace: bool = False,
    ) -> "KernelRegistry":
        """Return a new registry extended with additional source classes."""
        return self.extended(sources=source_classes, replace=replace)

    def extended_sinks(
        self,
        *sink_classes: type[Sink],
        replace: bool = False,
    ) -> "KernelRegistry":
        """Return a new registry extended with additional sink classes."""
        return self.extended(sinks=sink_classes, replace=replace)

    def validate(
        self,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Validate one kernel binding against shared-memory definitions."""
        self.get(config.kind).validate_config(config, shared_memory)

    def create(
        self,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> Kernel:
        """Instantiate a kernel after validation."""
        kernel_cls = self.get(config.kind)
        kernel_cls.validate_config(config, shared_memory)
        return kernel_cls(
            KernelContext(config=config, shared_memory=shared_memory)
        )

    def validate_source(
        self,
        config: SourceConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Validate one source binding against shared-memory definitions."""
        self.get_source(config.kind).validate_config(config, shared_memory)

    def create_source(
        self,
        config: SourceConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> Source:
        """Instantiate a source after validation."""
        source_cls = self.get_source(config.kind)
        source_cls.validate_config(config, shared_memory)
        return source_cls(
            SourceContext(config=config, shared_memory=shared_memory)
        )

    def validate_sink(
        self,
        config: SinkConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Validate one sink binding against shared-memory definitions."""
        self.get_sink(config.kind).validate_config(config, shared_memory)

    def create_sink(
        self,
        config: SinkConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> Sink:
        """Instantiate a sink after validation."""
        sink_cls = self.get_sink(config.kind)
        sink_cls.validate_config(config, shared_memory)
        return sink_cls(SinkContext(config=config, shared_memory=shared_memory))


def _kernel_kind(kernel_cls: type[Kernel]) -> str:
    if not isinstance(kernel_cls, type) or not issubclass(kernel_cls, Kernel):
        raise TypeError("kernel_cls must be a Kernel subclass")
    kind = getattr(kernel_cls, "kind", None)
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("kernel_cls.kind must be a non-empty string")
    return kind.strip()


def _source_kind(source_cls: type[Source]) -> str:
    if not isinstance(source_cls, type) or not issubclass(source_cls, Source):
        raise TypeError("source_cls must be a Source subclass")
    kind = getattr(source_cls, "kind", None)
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("source_cls.kind must be a non-empty string")
    return kind.strip()


def _sink_kind(sink_cls: type[Sink]) -> str:
    if not isinstance(sink_cls, type) or not issubclass(sink_cls, Sink):
        raise TypeError("sink_cls must be a Sink subclass")
    kind = getattr(sink_cls, "kind", None)
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("sink_cls.kind must be a non-empty string")
    return kind.strip()


_DEFAULT_KERNELS: dict[str, type[Kernel]] = {
    AddConstantCpuKernel.kind: AddConstantCpuKernel,
    AffineTransformCpuKernel.kind: AffineTransformCpuKernel,
    CopyCpuKernel.kind: CopyCpuKernel,
    CustomOperationCpuKernel.kind: CustomOperationCpuKernel,
    ElementwiseAddCpuKernel.kind: ElementwiseAddCpuKernel,
    ElementwiseDivideCpuKernel.kind: ElementwiseDivideCpuKernel,
    ElementwiseMultiplyCpuKernel.kind: ElementwiseMultiplyCpuKernel,
    ElementwiseSubtractCpuKernel.kind: ElementwiseSubtractCpuKernel,
    FlattenCpuKernel.kind: FlattenCpuKernel,
    LeakyIntegratorCpuKernel.kind: LeakyIntegratorCpuKernel,
    RaiseErrorCpuKernel.kind: RaiseErrorCpuKernel,
    ScaleCpuKernel.kind: ScaleCpuKernel,
    ScaleOffsetCpuKernel.kind: ScaleOffsetCpuKernel,
    ShackHartmannCentroidCpuKernel.kind: ShackHartmannCentroidCpuKernel,
}

_DEFAULT_LAZY_KERNELS: dict[str, Callable[[], type[Kernel]]] = {}

if _TORCH_AVAILABLE:
    _DEFAULT_LAZY_KERNELS = {
        kind: partial(_load_default_gpu_kernel, kind)
        for kind in _DEFAULT_GPU_KINDS
    }

_DEFAULT_LAZY_KERNELS.update(
    _discover_entry_point_loaders(
        "shmpipeline.kernels",
        validator=_kernel_kind,
        existing_kinds={*_DEFAULT_KERNELS, *_DEFAULT_LAZY_KERNELS},
    )
)

_DEFAULT_SOURCES: dict[str, type[Source]] = {}
_DEFAULT_LAZY_SOURCES = _discover_entry_point_loaders(
    "shmpipeline.sources",
    validator=_source_kind,
    existing_kinds=set(_DEFAULT_SOURCES),
)

_DEFAULT_SINKS: dict[str, type[Sink]] = {}
_DEFAULT_LAZY_SINKS = _discover_entry_point_loaders(
    "shmpipeline.sinks",
    validator=_sink_kind,
    existing_kinds=set(_DEFAULT_SINKS),
)

_DEFAULT_REGISTRY = KernelRegistry(
    _DEFAULT_KERNELS,
    _DEFAULT_LAZY_KERNELS,
    _DEFAULT_SOURCES,
    _DEFAULT_LAZY_SOURCES,
    _DEFAULT_SINKS,
    _DEFAULT_LAZY_SINKS,
)


def get_default_registry() -> KernelRegistry:
    """Return the default registry with built-ins and discovered plugins."""
    return _DEFAULT_REGISTRY
