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

GUI_IMPORT_ERROR: Exception | None = None

try:
    from shmpipeline.gui import viewers as viewers_module
    from shmpipeline.gui.app import MainWindow, SyntheticInputDialog
except Exception as exc:  # pragma: no cover - GUI stack unavailable
    viewers_module = None
    MainWindow = None
    SyntheticInputDialog = None
    GUI_IMPORT_ERROR = exc

pytestmark = pytest.mark.unit


class _FakeStream:
    def __init__(self) -> None:
        self.name = "demo"
        self._count = 1
        self._write_time = 1000.0

    @property
    def count(self) -> int:
        return self._count

    @property
    def write_time(self) -> float:
        return self._write_time

    def read(self, safe: bool = False):
        self._count += 1
        self._write_time += 0.01
        return np.arange(16, dtype=np.float32).reshape(4, 4)

    def close(self) -> None:
        return None


@pytest.fixture
def qapp():
    if QApplication is None:
        pytest.skip("PySide6 is not available")
    if GUI_IMPORT_ERROR is not None:
        pytest.skip(f"GUI stack is unavailable: {GUI_IMPORT_ERROR}")
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


def test_main_window_exposes_stream_actions_menu(qapp):
    window = MainWindow(theme_name="light")
    try:
        labels = [action.text() for action in window.menuBar().actions()]
        assert "Streams" in labels
    finally:
        window.close()


def test_synthetic_input_dialog_defaults_to_500_hz(qapp):
    dialog = SyntheticInputDialog("demo")
    try:
        assert dialog.rate_edit.text() == "500.0"
    finally:
        dialog.close()


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
        assert "Stream Hz avg" in viewer._status_label.text()
        assert "Viewer Hz avg" in viewer._status_label.text()
    finally:
        viewer.close()


def test_gpu_viewer_can_open_without_cpu_mirror(qapp, monkeypatch):
    fake_stream = _FakeStream()
    open_calls: list[tuple[str, dict[str, object]]] = []

    def fake_open(name: str, **kwargs):
        open_calls.append((name, kwargs))
        return fake_stream

    monkeypatch.setattr(viewers_module.pyshmem, "open", fake_open)

    viewer = viewers_module.SharedMemoryViewer(
        {"name": "demo", "storage": "gpu", "gpu_device": "cuda:0"}
    )
    try:
        assert open_calls == [("demo", {"gpu_device": "cuda:0"})]
        assert "Mode: passive-gpu" in viewer._status_label.text()
    finally:
        viewer.close()
