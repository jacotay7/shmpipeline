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
        """Store output metadata and defer compatibility staging allocations."""
        self.context = context
        self.device = torch.device(context.output_spec.gpu_device or "cuda")
        self._output_buffers: list[torch.Tensor] | None = None

    @property
    def output_dtypes(self) -> tuple[torch.dtype, ...]:
        """Return output dtypes without reserving device storage."""
        return tuple(
            torch_dtype_from_numpy(spec.dtype)
            for spec in self.context.output_specs
        )

    @property
    def output_dtype(self) -> torch.dtype:
        """Return the primary output dtype without reserving device storage."""
        return torch_dtype_from_numpy(self.context.output_spec.dtype)

    @property
    def output_buffers(self) -> list[torch.Tensor]:
        """Lazily allocate legacy private staging buffers on first access.

        Runtime publication writes directly into pyshmem output views.  The
        property remains for third-party kernels that intentionally use the
        historical staging API, without charging every built-in GPU worker for
        an otherwise unused output-sized allocation.
        """
        if self._output_buffers is None:
            self._output_buffers = [
                torch.empty(
                    spec.shape,
                    dtype=torch_dtype_from_numpy(spec.dtype),
                    device=torch.device(spec.gpu_device or "cuda"),
                )
                for spec in self.context.output_specs
            ]
        return self._output_buffers

    @property
    def output_buffer(self) -> torch.Tensor:
        """Return the lazily allocated primary compatibility buffer."""
        return self.output_buffers[0]
