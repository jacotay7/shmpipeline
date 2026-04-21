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
