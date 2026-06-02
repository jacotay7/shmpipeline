from __future__ import annotations

import pytest

from shmpipeline import registry as registry_module
from shmpipeline.source import Source

pytestmark = pytest.mark.unit


def test_discover_entry_point_loaders_registers_source_plugins(monkeypatch):
    class _EntryPointSource(Source):
        kind = "test.entry_source"
        storage = "cpu"

        def read(self):
            return None

    class _FakeEntryPoint:
        group = "shmpipeline.sources"
        name = "test.entry_source"

        def load(self):
            return _EntryPointSource

    monkeypatch.setattr(
        registry_module,
        "_entry_points_for_group",
        lambda group: (
            (_FakeEntryPoint(),) if group == "shmpipeline.sources" else ()
        ),
    )

    loaders = registry_module._discover_entry_point_loaders(
        "shmpipeline.sources",
        validator=registry_module._source_kind,
        existing_kinds=set(),
    )

    assert set(loaders) == {"test.entry_source"}
    assert loaders["test.entry_source"]() is _EntryPointSource


# ---------------------------------------------------------------------------
# Registry picklability and lazy CPU kernel loading
# ---------------------------------------------------------------------------


def test_default_registry_is_picklable():
    """The default registry must be picklable for worker spawning."""
    import pickle

    from shmpipeline.registry import get_default_registry

    registry = get_default_registry()
    restored = pickle.loads(pickle.dumps(registry))
    assert set(restored.kinds()) == set(registry.kinds())


def test_extended_registry_with_class_is_picklable():
    import pickle

    from shmpipeline.kernels.cpu.scale import ScaleCpuKernel
    from shmpipeline.registry import get_default_registry

    registry = get_default_registry().extended(ScaleCpuKernel, replace=True)
    restored = pickle.loads(pickle.dumps(registry))
    assert "cpu.scale" in restored.kinds()


def test_registry_lazy_cpu_kernel_loads_on_first_access():
    from shmpipeline.kernels.cpu.scale import ScaleCpuKernel
    from shmpipeline.registry import get_default_registry

    registry = get_default_registry()
    assert "cpu.scale" in registry.kinds()
    assert registry.get("cpu.scale") is ScaleCpuKernel


def test_registry_lazy_cpu_reduce_is_registered():
    from shmpipeline.kernels.cpu.reduce import ReduceCpuKernel
    from shmpipeline.registry import get_default_registry

    registry = get_default_registry()
    assert "cpu.reduce" in registry.kinds()
    assert registry.get("cpu.reduce") is ReduceCpuKernel
