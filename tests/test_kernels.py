from __future__ import annotations

import numpy as np
import pytest

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.kernel import KernelContext
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


pytestmark = pytest.mark.unit


def _make_shared_memory(specs: list[dict]) -> dict[str, SharedMemoryConfig]:
    return {
        spec["name"]: SharedMemoryConfig.from_dict(spec)
        for spec in specs
    }


def _instantiate_kernel(
    kernel_cls,
    *,
    input_shape,
    output_shape,
    input_dtype="float32",
    output_dtype=None,
    auxiliary: list[dict] | None = None,
    parameters: dict | None = None,
    operation: str | None = None,
):
    auxiliary = auxiliary or []
    parameters = parameters or {}
    output_dtype = output_dtype or input_dtype
    shared_specs = [
        {
            "name": "input",
            "shape": list(input_shape),
            "dtype": input_dtype,
            "storage": "cpu",
        },
        {
            "name": "output",
            "shape": list(output_shape),
            "dtype": output_dtype,
            "storage": "cpu",
        },
    ]
    auxiliary_config: list[str] | dict[str, str] = []
    if auxiliary:
        if any("alias" in item for item in auxiliary):
            auxiliary_config = {
                item.get("alias", item["name"]): item["name"]
                for item in auxiliary
            }
        else:
            auxiliary_config = [item["name"] for item in auxiliary]
        shared_specs.extend(
            {
                "name": item["name"],
                "shape": list(item["shape"]),
                "dtype": item.get("dtype", input_dtype),
                "storage": "cpu",
            }
            for item in auxiliary
        )
    shared_memory = _make_shared_memory(shared_specs)
    config_dict = {
        "name": "kernel_under_test",
        "kind": kernel_cls.kind,
        "input": "input",
        "output": "output",
        "auxiliary": auxiliary_config,
        "parameters": parameters,
    }
    if operation is not None:
        config_dict["operation"] = operation
    config = KernelConfig.from_dict(config_dict)
    kernel_cls.validate_config(config, shared_memory)
    return kernel_cls(KernelContext(config=config, shared_memory=shared_memory))


def test_copy_kernel_copies_input():
    kernel = _instantiate_kernel(CopyCpuKernel, input_shape=(4,), output_shape=(4,))
    output = np.empty(4, dtype=np.float32)
    payload = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    kernel.compute_into(payload, output, {})

    np.testing.assert_allclose(output, payload)


def test_scale_kernel_scales_input():
    kernel = _instantiate_kernel(
        ScaleCpuKernel,
        input_shape=(4,),
        output_shape=(4,),
        parameters={"factor": 2.5},
    )
    output = np.empty(4, dtype=np.float32)
    payload = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)

    kernel.compute_into(payload, output, {})

    np.testing.assert_allclose(output, payload * 2.5)


def test_add_constant_kernel_adds_constant():
    kernel = _instantiate_kernel(
        AddConstantCpuKernel,
        input_shape=(4,),
        output_shape=(4,),
        parameters={"constant": 3.0},
    )
    output = np.empty(4, dtype=np.float32)
    payload = np.array([0.5, 1.5, -2.0, 4.0], dtype=np.float32)

    kernel.compute_into(payload, output, {})

    np.testing.assert_allclose(output, payload + 3.0)


def test_elementwise_add_kernel_adds_arrays():
    kernel = _instantiate_kernel(
        ElementwiseAddCpuKernel,
        input_shape=(4,),
        output_shape=(4,),
        auxiliary=[{"name": "aux", "shape": (4,)}],
    )
    output = np.empty(4, dtype=np.float32)
    lhs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    rhs = np.array([0.5, -1.0, 2.0, 1.5], dtype=np.float32)

    kernel.compute_into(lhs, output, {"aux": rhs})

    np.testing.assert_allclose(output, lhs + rhs)


def test_elementwise_subtract_kernel_subtracts_arrays():
    kernel = _instantiate_kernel(
        ElementwiseSubtractCpuKernel,
        input_shape=(4,),
        output_shape=(4,),
        auxiliary=[{"name": "aux", "shape": (4,)}],
    )
    output = np.empty(4, dtype=np.float32)
    lhs = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    rhs = np.array([1.0, 0.5, 2.0, 4.0], dtype=np.float32)

    kernel.compute_into(lhs, output, {"aux": rhs})

    np.testing.assert_allclose(output, lhs - rhs)


def test_elementwise_multiply_kernel_multiplies_arrays():
    kernel = _instantiate_kernel(
        ElementwiseMultiplyCpuKernel,
        input_shape=(4,),
        output_shape=(4,),
        auxiliary=[{"name": "aux", "shape": (4,)}],
    )
    output = np.empty(4, dtype=np.float32)
    lhs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    rhs = np.array([2.0, 0.5, -1.0, 3.0], dtype=np.float32)

    kernel.compute_into(lhs, output, {"aux": rhs})

    np.testing.assert_allclose(output, lhs * rhs)


def test_elementwise_divide_kernel_divides_arrays():
    kernel = _instantiate_kernel(
        ElementwiseDivideCpuKernel,
        input_shape=(4,),
        output_shape=(4,),
        auxiliary=[{"name": "aux", "shape": (4,)}],
    )
    output = np.empty(4, dtype=np.float32)
    lhs = np.array([2.0, 6.0, 8.0, 9.0], dtype=np.float32)
    rhs = np.array([2.0, 3.0, 4.0, 1.5], dtype=np.float32)

    kernel.compute_into(lhs, output, {"aux": rhs})

    np.testing.assert_allclose(output, lhs / rhs)


def test_scale_offset_kernel_applies_gain_and_offset():
    kernel = _instantiate_kernel(
        ScaleOffsetCpuKernel,
        input_shape=(4,),
        output_shape=(4,),
        auxiliary=[{"name": "offset", "shape": (4,)}],
        parameters={"gain": 1.75},
    )
    output = np.empty(4, dtype=np.float32)
    payload = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    offset = np.array([0.5, 1.0, -1.0, 2.0], dtype=np.float32)

    kernel.compute_into(payload, output, {"offset": offset})

    np.testing.assert_allclose(output, 1.75 * payload - offset)


def test_flatten_kernel_flattens_input():
    kernel = _instantiate_kernel(FlattenCpuKernel, input_shape=(2, 3), output_shape=(6,))
    output = np.empty(6, dtype=np.float32)
    payload = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    kernel.compute_into(payload, output, {})

    np.testing.assert_allclose(output, payload.ravel())


def test_affine_transform_kernel_applies_matrix_and_offset():
    kernel = _instantiate_kernel(
        AffineTransformCpuKernel,
        input_shape=(3,),
        output_shape=(2,),
        auxiliary=[
            {"name": "matrix", "shape": (2, 3)},
            {"name": "offset", "shape": (2,)},
        ],
    )
    output = np.empty(2, dtype=np.float32)
    vector = np.array([2.0, -1.0, 4.0], dtype=np.float32)
    matrix = np.array([[1.0, 2.0, -1.0], [0.5, 0.0, 3.0]], dtype=np.float32)
    offset = np.array([1.0, -2.0], dtype=np.float32)

    kernel.compute_into(vector, output, {"matrix": matrix, "offset": offset})

    np.testing.assert_allclose(output, matrix @ vector + offset)


def test_leaky_integrator_kernel_updates_state():
    kernel = _instantiate_kernel(
        LeakyIntegratorCpuKernel,
        input_shape=(3,),
        output_shape=(3,),
        parameters={"leak": 0.9, "gain": 0.5},
    )
    first_output = np.empty(3, dtype=np.float32)
    second_output = np.empty(3, dtype=np.float32)
    first_input = np.array([2.0, -2.0, 4.0], dtype=np.float32)
    second_input = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    kernel.compute_into(first_input, first_output, {})
    kernel.compute_into(second_input, second_output, {})

    expected_first = 0.5 * first_input
    expected_second = 0.9 * expected_first + 0.5 * second_input
    np.testing.assert_allclose(first_output, expected_first)
    np.testing.assert_allclose(second_output, expected_second)


def test_centroid_kernel_computes_tile_centroids():
    kernel = _instantiate_kernel(
        ShackHartmannCentroidCpuKernel,
        input_shape=(4, 4),
        output_shape=(2, 2, 2),
        parameters={"tile_size": 2},
    )
    output = np.empty((2, 2, 2), dtype=np.float32)
    image = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 4.0, 2.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )

    kernel.compute_into(image, output, {})

    expected = np.array(
        [
            [[0.5, 0.5], [0.16666667, -0.16666667]],
            [[-0.5, -0.5], [0.5, 0.5]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)


def test_custom_operation_kernel_supports_min_max_aliases():
    kernel = _instantiate_kernel(
        CustomOperationCpuKernel,
        input_shape=(2, 2),
        output_shape=(2, 2),
        auxiliary=[
            {"name": "dark_frame", "shape": (2, 2), "alias": "dark"},
            {"name": "flat_field", "shape": (2, 2), "alias": "flat"},
            {"name": "high_frame", "shape": (2, 2), "alias": "high"},
        ],
        operation="max(input, dark) - min(flat, high)",
    )
    output = np.empty((2, 2), dtype=np.float32)
    image = np.array([[1.0, 5.0], [2.0, 8.0]], dtype=np.float32)
    dark = np.array([[3.0, 4.0], [1.0, 10.0]], dtype=np.float32)
    flat = np.array([[2.0, 9.0], [6.0, 2.0]], dtype=np.float32)
    high = np.array([[4.0, 3.0], [5.0, 5.0]], dtype=np.float32)

    kernel.compute_into(
        image,
        output,
        {"dark": dark, "flat": flat, "high": high},
    )

    expected = np.maximum(image, dark) - np.minimum(flat, high)
    np.testing.assert_allclose(output, expected)


def test_raise_error_kernel_raises_configured_message():
    kernel = _instantiate_kernel(
        RaiseErrorCpuKernel,
        input_shape=(1,),
        output_shape=(1,),
        parameters={"message": "boom"},
    )

    with pytest.raises(RuntimeError, match="boom"):
        kernel.compute_into(
            np.array([1.0], dtype=np.float32),
            np.empty(1, dtype=np.float32),
            {},
        )