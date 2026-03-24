"""Base classes for future GPU kernels."""

from __future__ import annotations

from shmpipeline.kernel import Kernel


class GpuKernel(Kernel):
    """Base class for GPU shared-memory kernels."""

    storage = "gpu"