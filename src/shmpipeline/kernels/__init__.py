"""Built-in kernel implementations."""

from importlib import import_module
from importlib.util import find_spec

_TORCH_AVAILABLE = find_spec("torch") is not None

_EXPORTS = {
    "AddConstantCpuKernel": (
        "shmpipeline.kernels.cpu",
        "AddConstantCpuKernel",
    ),
    "AffineTransformCpuKernel": (
        "shmpipeline.kernels.cpu",
        "AffineTransformCpuKernel",
    ),
    "CopyCpuKernel": ("shmpipeline.kernels.cpu", "CopyCpuKernel"),
    "CpuKernel": ("shmpipeline.kernels.cpu", "CpuKernel"),
    "CustomOperationCpuKernel": (
        "shmpipeline.kernels.cpu",
        "CustomOperationCpuKernel",
    ),
    "ElementwiseAddCpuKernel": (
        "shmpipeline.kernels.cpu",
        "ElementwiseAddCpuKernel",
    ),
    "ElementwiseDivideCpuKernel": (
        "shmpipeline.kernels.cpu",
        "ElementwiseDivideCpuKernel",
    ),
    "ElementwiseMultiplyCpuKernel": (
        "shmpipeline.kernels.cpu",
        "ElementwiseMultiplyCpuKernel",
    ),
    "ElementwiseSubtractCpuKernel": (
        "shmpipeline.kernels.cpu",
        "ElementwiseSubtractCpuKernel",
    ),
    "FlattenCpuKernel": ("shmpipeline.kernels.cpu", "FlattenCpuKernel"),
    "LeakyIntegratorCpuKernel": (
        "shmpipeline.kernels.cpu",
        "LeakyIntegratorCpuKernel",
    ),
    "RaiseErrorCpuKernel": (
        "shmpipeline.kernels.cpu",
        "RaiseErrorCpuKernel",
    ),
    "ScaleCpuKernel": ("shmpipeline.kernels.cpu", "ScaleCpuKernel"),
    "ScaleOffsetCpuKernel": (
        "shmpipeline.kernels.cpu",
        "ScaleOffsetCpuKernel",
    ),
    "SpotCentroidCpuKernel": (
        "shmpipeline.kernels.cpu",
        "SpotCentroidCpuKernel",
    ),
    "ShackHartmannCentroidCpuKernel": (
        "shmpipeline.kernels.cpu",
        "ShackHartmannCentroidCpuKernel",
    ),
}

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
    "SpotCentroidCpuKernel",
    "ShackHartmannCentroidCpuKernel",
]

if _TORCH_AVAILABLE:
    _EXPORTS.update(
        {
            "AddConstantGpuKernel": (
                "shmpipeline.kernels.gpu",
                "AddConstantGpuKernel",
            ),
            "AffineTransformGpuKernel": (
                "shmpipeline.kernels.gpu",
                "AffineTransformGpuKernel",
            ),
            "CopyGpuKernel": ("shmpipeline.kernels.gpu", "CopyGpuKernel"),
            "CustomOperationGpuKernel": (
                "shmpipeline.kernels.gpu",
                "CustomOperationGpuKernel",
            ),
            "ElementwiseAddGpuKernel": (
                "shmpipeline.kernels.gpu",
                "ElementwiseAddGpuKernel",
            ),
            "ElementwiseDivideGpuKernel": (
                "shmpipeline.kernels.gpu",
                "ElementwiseDivideGpuKernel",
            ),
            "ElementwiseMultiplyGpuKernel": (
                "shmpipeline.kernels.gpu",
                "ElementwiseMultiplyGpuKernel",
            ),
            "ElementwiseSubtractGpuKernel": (
                "shmpipeline.kernels.gpu",
                "ElementwiseSubtractGpuKernel",
            ),
            "FlattenGpuKernel": (
                "shmpipeline.kernels.gpu",
                "FlattenGpuKernel",
            ),
            "GpuKernel": ("shmpipeline.kernels.gpu", "GpuKernel"),
            "LeakyIntegratorGpuKernel": (
                "shmpipeline.kernels.gpu",
                "LeakyIntegratorGpuKernel",
            ),
            "RaiseErrorGpuKernel": (
                "shmpipeline.kernels.gpu",
                "RaiseErrorGpuKernel",
            ),
            "ScaleGpuKernel": ("shmpipeline.kernels.gpu", "ScaleGpuKernel"),
            "ScaleOffsetGpuKernel": (
                "shmpipeline.kernels.gpu",
                "ScaleOffsetGpuKernel",
            ),
            "ShackHartmannCentroidGpuKernel": (
                "shmpipeline.kernels.gpu",
                "ShackHartmannCentroidGpuKernel",
            ),
        }
    )
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
