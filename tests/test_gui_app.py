from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
except Exception:  # pragma: no cover - exercised when Qt is unavailable
    Qt = None
    QApplication = None

from shmpipeline.gui import viewers as viewers_module
from shmpipeline.gui.app import MainWindow

pytestmark = pytest.mark.unit


class _FakeStream:
    def __init__(self) -> None:
        self.name = "demo"
        self._count = 1

    @property
    def count(self) -> int:
        return self._count

    def read(self):
        return np.arange(16, dtype=np.float32).reshape(4, 4)

    def close(self) -> None:
        return None


@pytest.fixture
def qapp():
    if QApplication is None:
        pytest.skip("PySide6 is not available")
    app = QApplication.instance() or QApplication([])
    yield app


def test_main_window_can_start_in_light_theme(qapp):
    window = MainWindow(theme_name="light")
    try:
        assert window.current_theme_name == "light"
    finally:
        window.close()


def test_main_window_can_switch_themes(qapp):
    window = MainWindow(theme_name="light")
    try:
        window._apply_theme("dark", persist=False)
        assert window.current_theme_name == "dark"
    finally:
        window.close()


def test_main_window_stacks_kernels_under_shared_memory(qapp):
    window = MainWindow(theme_name="light")
    try:
        assert window._editor_splitter.orientation() == Qt.Vertical
        assert window._main_splitter.orientation() == Qt.Horizontal
    finally:
        window.close()


def test_main_window_uses_rolling_metric_columns(qapp):
    window = MainWindow(theme_name="light")
    try:
        labels = [
            window._worker_table.horizontalHeaderItem(index).text()
            for index in range(window._worker_table.columnCount())
        ]
        assert "Avg us" in labels
        assert "Jitter us RMS" in labels
        assert "Hz" in labels
    finally:
        window.close()


def test_shared_memory_viewer_initializes_without_imageview_dependency(
    qapp,
    monkeypatch,
):
    fake_stream = _FakeStream()
    monkeypatch.setattr(
        viewers_module.pyshmem, "open", lambda *args, **kwargs: fake_stream
    )

    viewer = viewers_module.SharedMemoryViewer(
        {"name": "demo", "storage": "cpu"}
    )
    try:
        viewer.refresh()
        assert "Shape: (4, 4)" in viewer._status_label.text()
    finally:
        viewer.close()
