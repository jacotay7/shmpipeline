from __future__ import annotations

import pytest

from shmpipeline.registry import LEGACY_AO_KERNEL_KINDS, get_default_registry


def test_legacy_ao_kind_warns_but_remains_available():
    registry = get_default_registry()
    with pytest.warns(DeprecationWarning, match="shmpipeline-ao"):
        kernel = registry.get("cpu.shack_hartmann_centroid")
    assert kernel.kind == "cpu.shack_hartmann_centroid"


def test_generic_array_kinds_are_not_classified_as_ao():
    assert {
        "cpu.copy",
        "cpu.flatten",
        "cpu.affine_transform",
        "cpu.scale",
    }.isdisjoint(LEGACY_AO_KERNEL_KINDS)
