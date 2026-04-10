"""CPU affine transformation kernel."""

from __future__ import annotations

import ctypes
from ctypes.util import find_library
from typing import Any, Mapping

import numpy as np

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernels.cpu._common import affine_transform_array
from shmpipeline.kernels.cpu.base import CpuKernel

_OPENBLAS_LIBRARY = None
_OPENBLAS_SET_NUM_THREADS = None
_OPENBLAS_THREAD_COUNT: int | None = None


def _load_openblas_set_num_threads():
    global _OPENBLAS_LIBRARY, _OPENBLAS_SET_NUM_THREADS
    if _OPENBLAS_SET_NUM_THREADS is not None:
        return _OPENBLAS_SET_NUM_THREADS
    if _OPENBLAS_LIBRARY is False:
        return None

    candidates = (
        find_library("openblas"),
        find_library("blas"),
        "libopenblas.dylib",
        "libblas.dylib",
    )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            library = ctypes.CDLL(candidate)
        except OSError:
            continue
        setter = getattr(library, "openblas_set_num_threads", None)
        if setter is None:
            continue
        setter.argtypes = [ctypes.c_int]
        setter.restype = None
        _OPENBLAS_LIBRARY = library
        _OPENBLAS_SET_NUM_THREADS = setter
        return setter

    _OPENBLAS_LIBRARY = False
    return None


def _configure_openblas_threads(thread_count: int) -> None:
    global _OPENBLAS_THREAD_COUNT
    if _OPENBLAS_THREAD_COUNT == thread_count:
        return
    setter = _load_openblas_set_num_threads()
    if setter is None:
        return
    setter(thread_count)
    _OPENBLAS_THREAD_COUNT = thread_count


class AffineTransformCpuKernel(CpuKernel):
    """Apply an affine transform `y = A x + b` to a vector input."""

    kind = "cpu.affine_transform"
    auxiliary_arity = 2

    def __init__(self, context) -> None:
        """Cache auxiliary aliases used on every compute call."""
        super().__init__(context)
        self._matrix_alias = self.context.config.auxiliary_aliases[0]
        self._offset_alias = self.context.config.auxiliary_aliases[1]
        _configure_openblas_threads(self._resolve_blas_threads())

    def _resolve_blas_threads(self) -> int:
        value = self.context.config.parameters.get("blas_threads", 1)
        return int(value)

    @classmethod
    def validate_config(
        cls,
        config: KernelConfig,
        shared_memory: Mapping[str, SharedMemoryConfig],
    ) -> None:
        """Validate vector, matrix, offset, and output compatibility."""
        super().validate_config(config, shared_memory)
        vector_spec = shared_memory[config.input]
        matrix_spec = shared_memory[config.auxiliary_names[0]]
        offset_spec = shared_memory[config.auxiliary_names[1]]
        output_spec = shared_memory[config.output]

        if len(vector_spec.shape) != 1:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires a 1D input vector"
            )
        if len(matrix_spec.shape) != 2:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires a 2D transform matrix"
            )
        if len(offset_spec.shape) != 1:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires a 1D offset vector"
            )
        if len(output_spec.shape) != 1:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires a 1D output vector"
            )
        if matrix_spec.shape[1] != vector_spec.shape[0]:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires matrix columns to match "
                "input vector length"
            )
        if matrix_spec.shape[0] != offset_spec.shape[0]:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires matrix rows to match "
                "offset vector length"
            )
        if output_spec.shape[0] != matrix_spec.shape[0]:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires output vector length to "
                "match matrix rows"
            )
        dtypes = {
            vector_spec.dtype,
            matrix_spec.dtype,
            offset_spec.dtype,
            output_spec.dtype,
        }
        if len(dtypes) != 1:
            raise ConfigValidationError(
                f"kernel {config.name!r} requires matching dtypes across all "
                "affine inputs and outputs"
            )
        blas_threads = config.parameters.get("blas_threads")
        if blas_threads is not None and (
            isinstance(blas_threads, bool)
            or not isinstance(blas_threads, int)
            or blas_threads <= 0
        ):
            raise ConfigValidationError(
                f"kernel {config.name!r} requires optional parameter "
                "'blas_threads' to be a positive integer"
            )

    def compute_into(
        self,
        trigger_input: Any,
        output: Any,
        auxiliary_inputs: Mapping[str, Any],
    ) -> None:
        """Compute the affine transform into the reusable output buffer."""
        vector = np.asarray(trigger_input)
        matrix = np.asarray(auxiliary_inputs[self._matrix_alias])
        offset = np.asarray(auxiliary_inputs[self._offset_alias])
        affine_transform_array(matrix, vector, offset, output)
