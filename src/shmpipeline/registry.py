"""Kernel registry used by the manager and worker runtime."""

from __future__ import annotations

from functools import partial
from importlib.util import find_spec
from typing import Callable, Mapping

from shmpipeline.config import KernelConfig, SharedMemoryConfig
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


class KernelRegistry:
    """Resolve kernel kinds to implementation classes.

    Registries are the extension point for third-party kernels. The default
    registry contains the built-in CPU kernels and lazily loads GPU kernels when
    the optional torch dependency is available.
    """

    def __init__(
        self,
        kernels: Mapping[str, type[Kernel]],
        lazy_kernels: Mapping[str, Callable[[], type[Kernel]]] | None = None,
    ) -> None:
        """Store a static mapping of registered kernel implementations."""
        self._kernels = dict(kernels)
        self._lazy_kernels = dict(lazy_kernels or {})

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

    def extended(
        self,
        *kernel_classes: type[Kernel],
        replace: bool = False,
    ) -> "KernelRegistry":
        """Return a new registry extended with additional kernel classes."""
        registry = KernelRegistry(self._kernels, self._lazy_kernels)
        for kernel_cls in kernel_classes:
            registry.register(kernel_cls, replace=replace)
        return registry

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


def _kernel_kind(kernel_cls: type[Kernel]) -> str:
    if not isinstance(kernel_cls, type) or not issubclass(kernel_cls, Kernel):
        raise TypeError("kernel_cls must be a Kernel subclass")
    kind = getattr(kernel_cls, "kind", None)
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("kernel_cls.kind must be a non-empty string")
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

_DEFAULT_REGISTRY = KernelRegistry(_DEFAULT_KERNELS, _DEFAULT_LAZY_KERNELS)


def get_default_registry() -> KernelRegistry:
    """Return the built-in kernel registry."""
    return _DEFAULT_REGISTRY
