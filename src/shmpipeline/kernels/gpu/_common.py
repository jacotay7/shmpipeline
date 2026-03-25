"""Shared helpers for GPU kernel implementations."""

from __future__ import annotations

from shmpipeline.kernels.cpu._common import require_numeric_parameter
from shmpipeline.kernels.cpu._common import validate_binary_same_shape_and_dtype
from shmpipeline.kernels.cpu._common import validate_same_dtype
from shmpipeline.kernels.cpu._common import validate_unary_same_shape_and_dtype

__all__ = [
    "require_numeric_parameter",
    "validate_binary_same_shape_and_dtype",
    "validate_same_dtype",
    "validate_unary_same_shape_and_dtype",
]