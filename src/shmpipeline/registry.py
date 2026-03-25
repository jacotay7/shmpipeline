"""Kernel registry used by the manager and worker runtime."""

from __future__ import annotations

from typing import Mapping

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernel import Kernel, KernelContext
from shmpipeline.kernels.cpu import AddConstantCpuKernel
from shmpipeline.kernels.cpu import AffineTransformCpuKernel
from shmpipeline.kernels.cpu import CopyCpuKernel
from shmpipeline.kernels.cpu import CustomOperationCpuKernel
from shmpipeline.kernels.cpu import ElementwiseAddCpuKernel
from shmpipeline.kernels.cpu import ElementwiseDivideCpuKernel
from shmpipeline.kernels.cpu import ElementwiseMultiplyCpuKernel
from shmpipeline.kernels.cpu import ElementwiseSubtractCpuKernel
from shmpipeline.kernels.cpu import FlattenCpuKernel
from shmpipeline.kernels.cpu import LeakyIntegratorCpuKernel
from shmpipeline.kernels.cpu import RaiseErrorCpuKernel
from shmpipeline.kernels.cpu import ScaleCpuKernel
from shmpipeline.kernels.cpu import ScaleOffsetCpuKernel
from shmpipeline.kernels.cpu import ShackHartmannCentroidCpuKernel


class KernelRegistry:
    """Resolve kernel kinds to implementation classes."""

    def __init__(self, kernels: Mapping[str, type[Kernel]]) -> None:
        """Store a static mapping of registered kernel implementations."""
        self._kernels = dict(kernels)

    def get(self, kind: str) -> type[Kernel]:
        """Return the implementation class for a kernel kind."""
        try:
            return self._kernels[kind]
        except KeyError as exc:
            raise ConfigValidationError(f"unknown kernel kind: {kind!r}") from exc

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
        return kernel_cls(KernelContext(config=config, shared_memory=shared_memory))


_DEFAULT_REGISTRY = KernelRegistry(
    {
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
)


def get_default_registry() -> KernelRegistry:
    """Return the built-in kernel registry."""
    return _DEFAULT_REGISTRY