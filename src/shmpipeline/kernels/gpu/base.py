"""Base classes and helpers for GPU kernels."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from shmpipeline.kernel import Kernel

_NUMPY_TO_TORCH_DTYPES = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
}


def torch_dtype_from_numpy(dtype: np.dtype) -> torch.dtype:
    """Map a NumPy dtype to the corresponding torch dtype."""
    normalized = np.dtype(dtype)
    try:
        return _NUMPY_TO_TORCH_DTYPES[normalized]
    except KeyError as exc:
        raise TypeError(f"unsupported GPU dtype: {normalized!r}") from exc


def as_gpu_tensor(value: Any, *, device: torch.device | str) -> torch.Tensor:
    """Convert one input into a tensor on the requested CUDA device."""
    if isinstance(value, torch.Tensor):
        if value.device == torch.device(device):
            return value
        return value.to(device)
    return torch.as_tensor(value, device=device)


class GpuKernel(Kernel):
    """Base class for GPU shared-memory kernels."""

    storage = "gpu"

    def __init__(self, context) -> None:
        """Allocate the reusable output buffer directly on the target GPU."""
        self.context = context
        self.device = torch.device(context.output_spec.gpu_device or "cuda")
        self.output_buffer = torch.empty(
            self.context.output_spec.shape,
            dtype=torch_dtype_from_numpy(self.context.output_spec.dtype),
            device=self.device,
        )
