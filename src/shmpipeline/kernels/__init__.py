"""Built-in kernel implementations."""

from importlib.util import find_spec

from shmpipeline.kernels.cpu import (
    AddConstantCpuKernel,
    AffineTransformCpuKernel,
    CopyCpuKernel,
    CpuKernel,
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
    from shmpipeline.kernels.gpu import (
        AddConstantGpuKernel as AddConstantGpuKernel,
    )
    from shmpipeline.kernels.gpu import (
        AffineTransformGpuKernel as AffineTransformGpuKernel,
    )
    from shmpipeline.kernels.gpu import (
        CopyGpuKernel as CopyGpuKernel,
    )
    from shmpipeline.kernels.gpu import (
        CustomOperationGpuKernel as CustomOperationGpuKernel,
    )
    from shmpipeline.kernels.gpu import (
        ElementwiseAddGpuKernel as ElementwiseAddGpuKernel,
    )
    from shmpipeline.kernels.gpu import (
        ElementwiseDivideGpuKernel as ElementwiseDivideGpuKernel,
    )
    from shmpipeline.kernels.gpu import (
        ElementwiseMultiplyGpuKernel as ElementwiseMultiplyGpuKernel,
    )
    from shmpipeline.kernels.gpu import (
        ElementwiseSubtractGpuKernel as ElementwiseSubtractGpuKernel,
    )
    from shmpipeline.kernels.gpu import (
        FlattenGpuKernel as FlattenGpuKernel,
    )
    from shmpipeline.kernels.gpu import (
        GpuKernel as GpuKernel,
    )
    from shmpipeline.kernels.gpu import (
        LeakyIntegratorGpuKernel as LeakyIntegratorGpuKernel,
    )
    from shmpipeline.kernels.gpu import (
        RaiseErrorGpuKernel as RaiseErrorGpuKernel,
    )
    from shmpipeline.kernels.gpu import (
        ScaleGpuKernel as ScaleGpuKernel,
    )
    from shmpipeline.kernels.gpu import (
        ScaleOffsetGpuKernel as ScaleOffsetGpuKernel,
    )
    from shmpipeline.kernels.gpu import (
        ShackHartmannCentroidGpuKernel as ShackHartmannCentroidGpuKernel,
    )

__all__ = [
    "AddConstantCpuKernel",
    "AffineTransformCpuKernel",
    "CopyCpuKernel",
    "CpuKernel",
    "CustomOperationCpuKernel",
    "ElementwiseAddCpuKernel",
    "ElementwiseDivideCpuKernel",
    "ElementwiseMultiplyCpuKernel",
    "ElementwiseSubtractCpuKernel",
    "FlattenCpuKernel",
    "LeakyIntegratorCpuKernel",
    "RaiseErrorCpuKernel",
    "ScaleCpuKernel",
    "ScaleOffsetCpuKernel",
    "ShackHartmannCentroidCpuKernel",
]

if _TORCH_AVAILABLE:
    __all__.extend(
        [
            "AddConstantGpuKernel",
            "AffineTransformGpuKernel",
            "CopyGpuKernel",
            "CustomOperationGpuKernel",
            "ElementwiseAddGpuKernel",
            "ElementwiseDivideGpuKernel",
            "ElementwiseMultiplyGpuKernel",
            "ElementwiseSubtractGpuKernel",
            "FlattenGpuKernel",
            "GpuKernel",
            "LeakyIntegratorGpuKernel",
            "RaiseErrorGpuKernel",
            "ScaleGpuKernel",
            "ScaleOffsetGpuKernel",
            "ShackHartmannCentroidGpuKernel",
        ]
    )
