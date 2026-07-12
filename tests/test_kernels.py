from __future__ import annotations

import numpy as np
import pytest

from shmpipeline.config import KernelConfig, SharedMemoryConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.kernel import KernelContext
from shmpipeline.kernels.cpu import (
    AddConstantCpuKernel,
    AffineTransformCpuKernel,
    ConcatenateCpuKernel,
    CopyCpuKernel,
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
    SpotCentroidCpuKernel,
    TipTiltControllerCpuKernel,
)

try:
    import torch
except Exception:  # pragma: no cover - exercised when torch is unavailable
    torch = None

if torch is not None:
    from shmpipeline.kernels.gpu import (
        AddConstantGpuKernel,
        AffineTransformGpuKernel,
        ConcatenateGpuKernel,
        CopyGpuKernel,
        CustomOperationGpuKernel,
        ElementwiseAddGpuKernel,
        ElementwiseDivideGpuKernel,
        ElementwiseMultiplyGpuKernel,
        ElementwiseSubtractGpuKernel,
        FlattenGpuKernel,
        LeakyIntegratorGpuKernel,
        RaiseErrorGpuKernel,
        ScaleGpuKernel,
        ScaleOffsetGpuKernel,
        ShackHartmannCentroidGpuKernel,
        TomographicControllerGpuKernel,
    )
else:  # pragma: no cover - exercised only when torch is unavailable
    AddConstantGpuKernel = None
    AffineTransformGpuKernel = None
    ConcatenateGpuKernel = None
    CopyGpuKernel = None
    CustomOperationGpuKernel = None
    ElementwiseAddGpuKernel = None
    ElementwiseDivideGpuKernel = None
    ElementwiseMultiplyGpuKernel = None
    ElementwiseSubtractGpuKernel = None
    FlattenGpuKernel = None
    LeakyIntegratorGpuKernel = None
    RaiseErrorGpuKernel = None
    ScaleGpuKernel = None
    ScaleOffsetGpuKernel = None
    ShackHartmannCentroidGpuKernel = None
    TomographicControllerGpuKernel = None


pytestmark = pytest.mark.unit

CUDA_AVAILABLE = torch is not None and torch.cuda.is_available()


def _make_shared_memory(specs: list[dict]) -> dict[str, SharedMemoryConfig]:
    return {spec["name"]: SharedMemoryConfig.from_dict(spec) for spec in specs}


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
    storage: str = "cpu",
    gpu_device: str = "cuda:0",
):
    auxiliary = auxiliary or []
    parameters = parameters or {}
    output_dtype = output_dtype or input_dtype
    shared_specs = [
        {
            "name": "input",
            "shape": list(input_shape),
            "dtype": input_dtype,
            "storage": storage,
        },
        {
            "name": "output",
            "shape": list(output_shape),
            "dtype": output_dtype,
            "storage": storage,
        },
    ]
    if storage == "gpu":
        shared_specs[0]["gpu_device"] = gpu_device
        shared_specs[1]["gpu_device"] = gpu_device
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
                "storage": storage,
                **({"gpu_device": gpu_device} if storage == "gpu" else {}),
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
    return kernel_cls(
        KernelContext(config=config, shared_memory=shared_memory)
    )


def _device_array(value, *, storage: str):
    if storage == "gpu":
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA is not available")
        return torch.as_tensor(value, device="cuda:0")
    return np.asarray(value)


def _empty_output(shape, *, dtype=np.float32, storage: str):
    if storage == "gpu":
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA is not available")
        return torch.empty(
            shape,
            dtype=torch.float32 if dtype == np.float32 else None,
            device="cuda:0",
        )
    return np.empty(shape, dtype=dtype)


def _assert_allclose(observed, expected, *, rtol=1e-7, atol=0.0):
    if torch is not None and isinstance(observed, torch.Tensor):
        observed = observed.detach().cpu().numpy()
    if torch is not None and isinstance(expected, torch.Tensor):
        expected = expected.detach().cpu().numpy()
    np.testing.assert_allclose(observed, expected, rtol=rtol, atol=atol)


def test_copy_kernel_copies_input():
    kernel = _instantiate_kernel(
        CopyCpuKernel, input_shape=(4,), output_shape=(4,)
    )
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
    kernel = _instantiate_kernel(
        FlattenCpuKernel, input_shape=(2, 3), output_shape=(6,)
    )
    output = np.empty(6, dtype=np.float32)
    payload = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    kernel.compute_into(payload, output, {})

    np.testing.assert_allclose(output, payload.ravel())


def test_concatenate_kernel_combines_all_trigger_inputs():
    shared_memory = _make_shared_memory(
        [
            {"name": "a", "shape": [2], "dtype": "float32"},
            {"name": "b", "shape": [3], "dtype": "float32"},
            {"name": "output", "shape": [5], "dtype": "float32"},
        ]
    )
    config = KernelConfig.from_dict(
        {
            "name": "join",
            "kind": "cpu.concatenate",
            "inputs": ["a", "b"],
            "output": "output",
        }
    )
    ConcatenateCpuKernel.validate_config(config, shared_memory)
    kernel = ConcatenateCpuKernel(
        KernelContext(config=config, shared_memory=shared_memory)
    )
    output = np.empty(5, dtype=np.float32)
    kernel.compute_into(
        (
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0, 5.0], dtype=np.float32),
        ),
        output,
        {},
    )
    np.testing.assert_array_equal(output, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_concatenate_kernel_rejects_non_all_new_policy():
    shared_memory = _make_shared_memory(
        [
            {"name": "a", "shape": [2], "dtype": "float32"},
            {"name": "b", "shape": [2], "dtype": "float32"},
            {"name": "output", "shape": [4], "dtype": "float32"},
        ]
    )
    config = KernelConfig.from_dict(
        {
            "name": "join",
            "kind": "cpu.concatenate",
            "inputs": ["a", "b"],
            "trigger_policy": "any_new",
            "output": "output",
        }
    )
    with pytest.raises(ConfigValidationError, match="all_new"):
        ConcatenateCpuKernel.validate_config(config, shared_memory)


def test_fused_cpu_tip_tilt_controller_updates_closed_loop_state():
    shared_memory = _make_shared_memory(
        [
            {"name": "image", "shape": [4, 4], "dtype": "float32"},
            {"name": "rotation", "shape": [2, 2], "dtype": "float32"},
            {"name": "bias", "shape": [2], "dtype": "float32"},
            {"name": "output", "shape": [2], "dtype": "float32"},
        ]
    )
    config = KernelConfig.from_dict(
        {
            "name": "tip_tilt",
            "kind": "cpu.tip_tilt_controller",
            "input": "image",
            "output": "output",
            "auxiliary": ["rotation", "bias"],
            "parameters": {"leak": 0.9, "control_gain": 0.5},
        }
    )
    TipTiltControllerCpuKernel.validate_config(config, shared_memory)
    kernel = TipTiltControllerCpuKernel(
        KernelContext(config=config, shared_memory=shared_memory)
    )
    image = np.zeros((4, 4), dtype=np.float32)
    image[3, 3] = 1.0
    rotation = np.eye(2, dtype=np.float32)
    bias = np.zeros(2, dtype=np.float32)
    output = np.empty(2, dtype=np.float32)
    kernel.compute_into(image, output, {"rotation": rotation, "bias": bias})
    np.testing.assert_allclose(output, [0.75, 0.75])
    kernel.compute_into(image, output, {"rotation": rotation, "bias": bias})
    np.testing.assert_allclose(output, [1.425, 1.425])


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_fused_batched_gpu_tomographic_controller_runs_small_geometry():
    specs = [
        {"name": "images", "shape": [8, 4, 4]},
        {"name": "dark", "shape": [8, 4, 4]},
        {"name": "flat", "shape": [8, 4, 4]},
        {"name": "slope_offset", "shape": [8, 1, 1, 2]},
        {"name": "matrix", "shape": [2, 16]},
        {"name": "reconstructor_bias", "shape": [2]},
        {"name": "command_offset", "shape": [2]},
        {"name": "command_low", "shape": [2]},
        {"name": "command_high", "shape": [2]},
        {"name": "output", "shape": [2]},
    ]
    shared_memory = _make_shared_memory(
        [
            {
                **spec,
                "dtype": "float32",
                "storage": "gpu",
                "gpu_device": "cuda:0",
            }
            for spec in specs
        ]
    )
    config = KernelConfig.from_dict(
        {
            "name": "tomo",
            "kind": "gpu.tomographic_controller",
            "input": "images",
            "trigger_policy": "all_new",
            "output": "output",
            "auxiliary": {
                "wfs_dark": "dark",
                "wfs_inverse_flat": "flat",
                "wfs_slope_offset": "slope_offset",
                "reconstructor": "matrix",
                "reconstructor_bias": "reconstructor_bias",
                "command_offset": "command_offset",
                "command_low": "command_low",
                "command_high": "command_high",
            },
            "parameters": {"tile_size": 4},
        }
    )
    TomographicControllerGpuKernel.validate_config(config, shared_memory)
    kernel = TomographicControllerGpuKernel(
        KernelContext(config=config, shared_memory=shared_memory)
    )
    device = "cuda:0"
    inputs = torch.ones((8, 4, 4), dtype=torch.float32, device=device)
    auxiliary = {
        "wfs_dark": torch.zeros((8, 4, 4), device=device),
        "wfs_inverse_flat": torch.ones((8, 4, 4), device=device),
        "wfs_slope_offset": torch.zeros((8, 1, 1, 2), device=device),
        "reconstructor": torch.zeros((2, 16), device=device),
        "reconstructor_bias": torch.tensor([0.5, -0.5], device=device),
        "command_offset": torch.zeros(2, device=device),
        "command_low": torch.full((2,), -1.0, device=device),
        "command_high": torch.full((2,), 1.0, device=device),
    }
    output = torch.empty(2, device=device)
    kernel.compute_into(inputs, output, auxiliary)
    _assert_allclose(output, [0.5, -0.5])


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


def test_affine_transform_kernel_accepts_blas_threads_override():
    kernel = _instantiate_kernel(
        AffineTransformCpuKernel,
        input_shape=(3,),
        output_shape=(2,),
        auxiliary=[
            {"name": "matrix", "shape": (2, 3)},
            {"name": "offset", "shape": (2,)},
        ],
        parameters={"blas_threads": 2},
    )

    assert kernel._resolve_blas_threads() == 2


def test_affine_transform_kernel_rejects_invalid_blas_threads():
    shared_memory = _make_shared_memory(
        [
            {
                "name": "input",
                "shape": [3],
                "dtype": "float32",
                "storage": "cpu",
            },
            {
                "name": "output",
                "shape": [2],
                "dtype": "float32",
                "storage": "cpu",
            },
            {
                "name": "matrix",
                "shape": [2, 3],
                "dtype": "float32",
                "storage": "cpu",
            },
            {
                "name": "offset",
                "shape": [2],
                "dtype": "float32",
                "storage": "cpu",
            },
        ]
    )
    config = KernelConfig.from_dict(
        {
            "name": "kernel_under_test",
            "kind": AffineTransformCpuKernel.kind,
            "input": "input",
            "output": "output",
            "auxiliary": ["matrix", "offset"],
            "parameters": {"blas_threads": 0},
        }
    )

    with pytest.raises(
        ConfigValidationError,
        match="'blas_threads' to be a positive integer",
    ):
        AffineTransformCpuKernel.validate_config(config, shared_memory)


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


def test_leaky_integrator_kernel_uses_auxiliary_gain_when_override_enabled():
    kernel = _instantiate_kernel(
        LeakyIntegratorCpuKernel,
        input_shape=(1,),
        output_shape=(1,),
        parameters={"leak": 0.0, "gain": 0.5},
        auxiliary=[
            {"name": "gain_stream", "alias": "gain", "shape": (1,)},
            {
                "name": "override_stream",
                "alias": "override_enabled",
                "shape": (1,),
            },
        ],
    )
    output = np.empty(1, dtype=np.float32)
    trigger = np.array([2.0], dtype=np.float32)

    kernel.compute_into(
        trigger,
        output,
        {
            "gain": np.array([0.0], dtype=np.float32),
            "override_enabled": np.array([0.0], dtype=np.float32),
        },
    )
    np.testing.assert_allclose(output, np.array([1.0], dtype=np.float32))

    kernel = _instantiate_kernel(
        LeakyIntegratorCpuKernel,
        input_shape=(1,),
        output_shape=(1,),
        parameters={"leak": 0.0, "gain": 0.5},
        auxiliary=[
            {"name": "gain_stream", "alias": "gain", "shape": (1,)},
            {
                "name": "override_stream",
                "alias": "override_enabled",
                "shape": (1,),
            },
        ],
    )
    kernel.compute_into(
        trigger,
        output,
        {
            "gain": np.array([0.0], dtype=np.float32),
            "override_enabled": np.array([1.0], dtype=np.float32),
        },
    )
    np.testing.assert_allclose(output, np.array([0.0], dtype=np.float32))


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


def test_spot_centroid_kernel_computes_single_centroid():
    kernel = _instantiate_kernel(
        SpotCentroidCpuKernel,
        input_shape=(5, 5),
        output_shape=(2,),
        parameters={"threshold": 0.0, "background": 0.0},
    )
    output = np.empty(2, dtype=np.float32)
    image = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    kernel.compute_into(image, output, {})

    expected = np.array([0.0, 0.75], dtype=np.float32)
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


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
@pytest.mark.parametrize(
    ("kernel_cls", "parameters", "payload", "expected"),
    [
        (
            CopyGpuKernel,
            {},
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        ),
        (
            ScaleGpuKernel,
            {"factor": 2.5},
            np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32),
            np.array([2.5, -5.0, 7.5, -10.0], dtype=np.float32),
        ),
        (
            AddConstantGpuKernel,
            {"constant": 3.0},
            np.array([0.5, 1.5, -2.0, 4.0], dtype=np.float32),
            np.array([3.5, 4.5, 1.0, 7.0], dtype=np.float32),
        ),
    ],
)
def test_gpu_unary_kernels_match_expected_values(
    kernel_cls,
    parameters,
    payload,
    expected,
):
    kernel = _instantiate_kernel(
        kernel_cls,
        input_shape=payload.shape,
        output_shape=payload.shape,
        parameters=parameters,
        storage="gpu",
    )
    output = _empty_output(payload.shape, storage="gpu")

    kernel.compute_into(_device_array(payload, storage="gpu"), output, {})

    _assert_allclose(output, expected)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
@pytest.mark.parametrize(
    ("kernel_cls", "lhs", "rhs", "expected"),
    [
        (
            ElementwiseAddGpuKernel,
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            np.array([0.5, -1.0, 2.0, 1.5], dtype=np.float32),
            np.array([1.5, 1.0, 5.0, 5.5], dtype=np.float32),
        ),
        (
            ElementwiseSubtractGpuKernel,
            np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
            np.array([1.0, 0.5, 2.0, 4.0], dtype=np.float32),
            np.array([4.0, 5.5, 5.0, 4.0], dtype=np.float32),
        ),
        (
            ElementwiseMultiplyGpuKernel,
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            np.array([2.0, 0.5, -1.0, 3.0], dtype=np.float32),
            np.array([2.0, 1.0, -3.0, 12.0], dtype=np.float32),
        ),
        (
            ElementwiseDivideGpuKernel,
            np.array([2.0, 6.0, 8.0, 9.0], dtype=np.float32),
            np.array([2.0, 3.0, 4.0, 1.5], dtype=np.float32),
            np.array([1.0, 2.0, 2.0, 6.0], dtype=np.float32),
        ),
    ],
)
def test_gpu_binary_kernels_match_expected_values(
    kernel_cls, lhs, rhs, expected
):
    kernel = _instantiate_kernel(
        kernel_cls,
        input_shape=lhs.shape,
        output_shape=lhs.shape,
        auxiliary=[{"name": "aux", "shape": lhs.shape}],
        storage="gpu",
    )
    output = _empty_output(lhs.shape, storage="gpu")

    kernel.compute_into(
        _device_array(lhs, storage="gpu"),
        output,
        {"aux": _device_array(rhs, storage="gpu")},
    )

    _assert_allclose(output, expected)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_scale_offset_gpu_kernel_applies_gain_and_offset():
    kernel = _instantiate_kernel(
        ScaleOffsetGpuKernel,
        input_shape=(4,),
        output_shape=(4,),
        auxiliary=[{"name": "offset", "shape": (4,)}],
        parameters={"gain": 1.75},
        storage="gpu",
    )
    output = _empty_output((4,), storage="gpu")
    payload = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    offset = np.array([0.5, 1.0, -1.0, 2.0], dtype=np.float32)

    kernel.compute_into(
        _device_array(payload, storage="gpu"),
        output,
        {"offset": _device_array(offset, storage="gpu")},
    )

    _assert_allclose(output, 1.75 * payload - offset)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_flatten_gpu_kernel_flattens_input():
    kernel = _instantiate_kernel(
        FlattenGpuKernel,
        input_shape=(2, 3),
        output_shape=(6,),
        storage="gpu",
    )
    output = _empty_output((6,), storage="gpu")
    payload = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    kernel.compute_into(_device_array(payload, storage="gpu"), output, {})

    _assert_allclose(output, payload.ravel())


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_affine_transform_gpu_kernel_applies_matrix_and_offset():
    kernel = _instantiate_kernel(
        AffineTransformGpuKernel,
        input_shape=(3,),
        output_shape=(2,),
        auxiliary=[
            {"name": "matrix", "shape": (2, 3)},
            {"name": "offset", "shape": (2,)},
        ],
        storage="gpu",
    )
    output = _empty_output((2,), storage="gpu")
    vector = np.array([2.0, -1.0, 4.0], dtype=np.float32)
    matrix = np.array([[1.0, 2.0, -1.0], [0.5, 0.0, 3.0]], dtype=np.float32)
    offset = np.array([1.0, -2.0], dtype=np.float32)

    kernel.compute_into(
        _device_array(vector, storage="gpu"),
        output,
        {
            "matrix": _device_array(matrix, storage="gpu"),
            "offset": _device_array(offset, storage="gpu"),
        },
    )

    _assert_allclose(output, matrix @ vector + offset)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_leaky_integrator_gpu_kernel_updates_state():
    kernel = _instantiate_kernel(
        LeakyIntegratorGpuKernel,
        input_shape=(3,),
        output_shape=(3,),
        parameters={"leak": 0.9, "gain": 0.5},
        storage="gpu",
    )
    first_output = _empty_output((3,), storage="gpu")
    second_output = _empty_output((3,), storage="gpu")
    first_input = np.array([2.0, -2.0, 4.0], dtype=np.float32)
    second_input = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    kernel.compute_into(
        _device_array(first_input, storage="gpu"), first_output, {}
    )
    kernel.compute_into(
        _device_array(second_input, storage="gpu"), second_output, {}
    )

    expected_first = 0.5 * first_input
    expected_second = 0.9 * expected_first + 0.5 * second_input
    _assert_allclose(first_output, expected_first)
    _assert_allclose(second_output, expected_second)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_centroid_gpu_kernel_computes_tile_centroids():
    kernel = _instantiate_kernel(
        ShackHartmannCentroidGpuKernel,
        input_shape=(4, 4),
        output_shape=(2, 2, 2),
        parameters={"tile_size": 2},
        storage="gpu",
    )
    output = _empty_output((2, 2, 2), storage="gpu")
    image = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 4.0, 2.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )

    kernel.compute_into(_device_array(image, storage="gpu"), output, {})

    expected = np.array(
        [
            [[0.5, 0.5], [0.16666667, -0.16666667]],
            [[-0.5, -0.5], [0.5, 0.5]],
        ],
        dtype=np.float32,
    )
    _assert_allclose(output, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_custom_operation_gpu_kernel_supports_min_max_aliases():
    kernel = _instantiate_kernel(
        CustomOperationGpuKernel,
        input_shape=(2, 2),
        output_shape=(2, 2),
        auxiliary=[
            {"name": "dark_frame", "shape": (2, 2), "alias": "dark"},
            {"name": "flat_field", "shape": (2, 2), "alias": "flat"},
            {"name": "high_frame", "shape": (2, 2), "alias": "high"},
        ],
        operation="max(input, dark) - min(flat, high)",
        storage="gpu",
    )
    output = _empty_output((2, 2), storage="gpu")
    image = np.array([[1.0, 5.0], [2.0, 8.0]], dtype=np.float32)
    dark = np.array([[3.0, 4.0], [1.0, 10.0]], dtype=np.float32)
    flat = np.array([[2.0, 9.0], [6.0, 2.0]], dtype=np.float32)
    high = np.array([[4.0, 3.0], [5.0, 5.0]], dtype=np.float32)

    kernel.compute_into(
        _device_array(image, storage="gpu"),
        output,
        {
            "dark": _device_array(dark, storage="gpu"),
            "flat": _device_array(flat, storage="gpu"),
            "high": _device_array(high, storage="gpu"),
        },
    )

    expected = np.maximum(image, dark) - np.minimum(flat, high)
    _assert_allclose(output, expected)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_raise_error_gpu_kernel_raises_configured_message():
    kernel = _instantiate_kernel(
        RaiseErrorGpuKernel,
        input_shape=(1,),
        output_shape=(1,),
        parameters={"message": "boom"},
        storage="gpu",
    )

    with pytest.raises(RuntimeError, match="boom"):
        kernel.compute_into(
            _device_array(np.array([1.0], dtype=np.float32), storage="gpu"),
            _empty_output((1,), storage="gpu"),
            {},
        )


# ---------------------------------------------------------------------------
# cpu.reduce kernel
# ---------------------------------------------------------------------------


def test_reduce_kernel_is_registered_in_default_registry():
    from shmpipeline.registry import get_default_registry

    assert "cpu.reduce" in get_default_registry().kinds()


@pytest.mark.parametrize(
    "operation,expected",
    [
        ("sum", 28.0),
        ("mean", 3.5),
        ("max", 7.0),
        ("min", 0.0),
    ],
)
def test_reduce_kernel_operations(operation, expected):
    from shmpipeline.kernels.cpu.reduce import ReduceCpuKernel

    kernel = _instantiate_kernel(
        ReduceCpuKernel,
        input_shape=(8,),
        output_shape=(1,),
        parameters={"operation": operation},
    )
    output = np.empty(1, dtype=np.float32)
    payload = np.arange(8, dtype=np.float32)
    kernel.compute_into(payload, output, {})
    np.testing.assert_allclose(output, [expected], rtol=1e-6)


def test_reduce_kernel_rejects_invalid_operation():
    from shmpipeline.kernels.cpu.reduce import ReduceCpuKernel

    with pytest.raises(
        ConfigValidationError, match="unsupported reduce operation"
    ):
        _instantiate_kernel(
            ReduceCpuKernel,
            input_shape=(8,),
            output_shape=(1,),
            parameters={"operation": "product"},
        )


def test_reduce_kernel_rejects_non_scalar_output():
    from shmpipeline.kernels.cpu.reduce import ReduceCpuKernel

    with pytest.raises(ConfigValidationError, match="scalar"):
        _instantiate_kernel(
            ReduceCpuKernel,
            input_shape=(8,),
            output_shape=(4,),
            parameters={"operation": "mean"},
        )


# ---------------------------------------------------------------------------
# CPU kernels do not allocate an unused output_buffer (GPU kernels do)
# ---------------------------------------------------------------------------


def test_cpu_kernel_has_no_output_buffer_attribute():
    kernel = _instantiate_kernel(
        ScaleCpuKernel,
        input_shape=(4,),
        output_shape=(4,),
        parameters={"factor": 1.0},
    )
    assert not hasattr(kernel, "output_buffer")


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
def test_gpu_kernel_still_has_output_buffer():
    from shmpipeline.kernels.gpu.scale import ScaleGpuKernel

    kernel = _instantiate_kernel(
        ScaleGpuKernel,
        input_shape=(4,),
        output_shape=(4,),
        parameters={"factor": 1.0},
        storage="gpu",
    )
    assert hasattr(kernel, "output_buffer")


# ---------------------------------------------------------------------------
# Multi-output kernel ABC behaviour
# ---------------------------------------------------------------------------


def test_compute_into_multiple_defaults_to_single_output():
    """The default compute_into_multiple forwards the first output buffer."""
    kernel = _instantiate_kernel(
        ScaleCpuKernel,
        input_shape=(4,),
        output_shape=(4,),
        parameters={"factor": 2.0},
    )
    output = np.empty(4, dtype=np.float32)
    payload = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    kernel.compute_into_multiple(payload, [output], {})
    np.testing.assert_allclose(output, payload * 2.0)


def test_multi_output_kernel_writes_all_outputs():
    from shmpipeline.kernel import Kernel

    class _Splitter(Kernel):
        kind = "test.splitter_unit"
        storage = "cpu"
        auxiliary_arity = 0
        output_arity = 2

        def compute_into_multiple(self, trigger_input, outputs, aux):
            outputs[0][...] = trigger_input * 2.0
            outputs[1][...] = trigger_input + 1.0

    config = KernelConfig.from_dict(
        {
            "name": "split",
            "kind": "test.splitter_unit",
            "input": "input",
            "outputs": ["out_a", "out_b"],
        }
    )
    shared_memory = _make_shared_memory(
        [
            {"name": "input", "shape": [3], "dtype": "float32"},
            {"name": "out_a", "shape": [3], "dtype": "float32"},
            {"name": "out_b", "shape": [3], "dtype": "float32"},
        ]
    )
    _Splitter.validate_config(config, shared_memory)
    kernel = _Splitter(
        KernelContext(config=config, shared_memory=shared_memory)
    )
    payload = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out_a = np.empty(3, dtype=np.float32)
    out_b = np.empty(3, dtype=np.float32)
    kernel.compute_into_multiple(payload, [out_a, out_b], {})
    np.testing.assert_allclose(out_a, payload * 2.0)
    np.testing.assert_allclose(out_b, payload + 1.0)


def test_output_arity_mismatch_is_rejected():
    from shmpipeline.kernel import Kernel

    class _DualOut(Kernel):
        kind = "test.dual_out_unit"
        storage = "cpu"
        auxiliary_arity = 0
        output_arity = 2

        def compute_into_multiple(self, trigger_input, outputs, aux):
            for out in outputs:
                out[...] = trigger_input

    config = KernelConfig.from_dict(
        {
            "name": "dual",
            "kind": "test.dual_out_unit",
            "input": "input",
            "output": "only_one",
        }
    )
    shared_memory = _make_shared_memory(
        [
            {"name": "input", "shape": [3], "dtype": "float32"},
            {"name": "only_one", "shape": [3], "dtype": "float32"},
        ]
    )
    with pytest.raises(ConfigValidationError, match="output stream"):
        _DualOut.validate_config(config, shared_memory)
