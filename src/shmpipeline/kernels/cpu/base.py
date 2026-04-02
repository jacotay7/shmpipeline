"""Base classes and shared validation for CPU kernels."""

from __future__ import annotations

from shmpipeline.kernel import Kernel


class CpuKernel(Kernel):
    """Base class for CPU shared-memory kernels."""

    storage = "cpu"
