from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from shmpipeline.control.discovery import LocalControlServerRecord

os.environ.setdefault(
    "QT_QPA_PLATFORM",
    "offscreen" if sys.platform.startswith("linux") else "minimal",
)

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QWidget
except Exception:  # pragma: no cover - exercised when Qt is unavailable
    Qt = None
    QApplication = None
    QWidget = None

GUI_IMPORT_ERROR: Exception | None = None

try:
    from shmpipeline.gui import themes as themes_module
    from shmpipeline.gui import viewers as viewers_module
    from shmpipeline.gui.app import (
        KernelDialog,
        MainWindow,
        SharedMemoryDialog,
        SinkDialog,
        SourceDialog,
        SyntheticInputDialog,
    )
    from shmpipeline.gui.app import (
        main as gui_main,
    )
    from shmpipeline.gui.control import ControlWindow
    from shmpipeline.gui.model import default_document, save_document
except Exception as exc:  # pragma: no cover - GUI stack unavailable
    viewers_module = None
    themes_module = None
    MainWindow = None
    KernelDialog = None
    SharedMemoryDialog = None
    SinkDialog = None
    SourceDialog = None
    SyntheticInputDialog = None
    ControlWindow = None
    gui_main = None
    default_document = None
    save_document = None
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


_QT_WIDGET_BASE = QWidget if QWidget is not None else object


class _FakePlotCurve:
    def __init__(self, pen=None) -> None:
        self.pen = pen
        self.data = None

    def setData(self, data) -> None:
        self.data = data

    def setPen(self, pen) -> None:
        self.pen = pen


class _FakePlotWidget(_QT_WIDGET_BASE):
    def __init__(self) -> None:
        if QWidget is not None:
            super().__init__()
        self.background = None
        self.curve = _FakePlotCurve()
        self.auto_range_calls = 0

    def plot(self, pen=None):
        self.curve = _FakePlotCurve(pen=pen)
        return self.curve

    def setBackground(self, background) -> None:
        self.background = background

    def autoRange(self) -> None:
        self.auto_range_calls += 1


class _FakeImagePlot:
    def __init__(self) -> None:
        self.auto_range_calls = 0

    def hideAxis(self, *_args) -> None:
        return None

    def setAspectLocked(self, *_args) -> None:
        return None

    def addItem(self, _item) -> None:
        return None

    def autoRange(self) -> None:
        self.auto_range_calls += 1


class _FakeGraphicsLayoutWidget(_QT_WIDGET_BASE):
    def __init__(self) -> None:
        if QWidget is not None:
            super().__init__()
        self.background = None
        self.plot = _FakeImagePlot()

    def addPlot(self):
        return self.plot

    def setBackground(self, background) -> None:
        self.background = background


class _FakeImageItem:
    def __init__(self) -> None:
        self.image = None

    def setImage(self, image, autoLevels=True) -> None:
        self.image = image


def _patch_viewer_pyqtgraph(monkeypatch) -> None:
    monkeypatch.setattr(viewers_module.pg, "PlotWidget", _FakePlotWidget)
    monkeypatch.setattr(
        viewers_module.pg,
        "GraphicsLayoutWidget",
        _FakeGraphicsLayoutWidget,
    )
    monkeypatch.setattr(viewers_module.pg, "ImageItem", _FakeImageItem)
    monkeypatch.setattr(
        viewers_module.pg,
        "mkPen",
        lambda *args, **kwargs: {"args": args, "kwargs": kwargs},
    )


@pytest.fixture
def qapp():
    if QApplication is None:
        pytest.skip("PySide6 is not available")
    if GUI_IMPORT_ERROR is not None:
        pytest.skip(f"GUI stack is unavailable: {GUI_IMPORT_ERROR}")
    app = QApplication.instance() or QApplication([])
    yield app
    app.closeAllWindows()
    QApplication.sendPostedEvents()
    app.processEvents()
    for widget in tuple(QApplication.topLevelWidgets()):
        try:
            widget.hide()
        except Exception:
            continue
        try:
            widget.deleteLater()
        except Exception:
            continue
    QApplication.sendPostedEvents()
    app.processEvents()


def test_main_window_can_start_in_light_theme(qapp):
    window = MainWindow(theme_name="light")
    try:
        assert window.current_theme_name == "light"
    finally:
        window.close()


def test_kernel_dialog_preserves_frame_id_and_advanced_wiring(qapp):
    initial = {
        "name": "control",
        "kind": "cpu.copy",
        "inputs": ["delta", "gain"],
        "outputs": ["command", "telemetry"],
        "trigger_policy": "all_new",
        "synchronization": {
            "mode": "matching_frame_id",
            "max_skew_generations": 4,
            "on_skew": "drop_older",
        },
        "propagate_frame_id": True,
        "poll_interval": 0.001,
        "auxiliary": {"reconstructor": "reconstructor"},
        "parameters": {"gain": 0.1, "leak": 0.99},
    }
    dialog = KernelDialog(
        ["delta", "gain", "command", "telemetry"],
        ["cpu.copy"],
        initial=initial,
    )
    try:
        value = dialog.value()
        assert value["propagate_frame_id"] is True
        assert value["inputs"] == ["delta", "gain"]
        assert value["outputs"] == ["command", "telemetry"]
        assert "input" not in value
        assert "output" not in value
        assert value["trigger_policy"] == "all_new"
        assert value["synchronization"] == initial["synchronization"]
        assert value["poll_interval"] == 0.001
        assert value["auxiliary"] == initial["auxiliary"]
        assert value["parameters"] == initial["parameters"]
    finally:
        dialog.close()


def test_gui_row_dialogs_preserve_unrepresented_config_fields(qapp):
    shared = SharedMemoryDialog(
        initial={
            "name": "matrix",
            "shape": [2, 2],
            "dtype": "float64",
            "notify": True,
            "mode": "replace",
            "initial": {"pattern": "identity", "scale": 2.0},
        }
    )
    source = SourceDialog(
        ["a", "b", "ready"],
        ["test.source"],
        initial={
            "name": "source",
            "kind": "test.source",
            "streams": ["a", "b"],
            "auxiliary": {"ready": "ready"},
            "parameters": {"path": "asset.npy"},
            "read_timeout": 0.5,
        },
    )
    sink = SinkDialog(
        ["a", "b", "ready"],
        ["test.sink"],
        initial={
            "name": "sink",
            "kind": "test.sink",
            "stream": "a",
            "outputs": ["b"],
            "auxiliary": {"ready": "ready"},
            "parameters": {"path": "asset.npy"},
            "consume_timeout": 0.5,
        },
    )
    try:
        shared_value = shared.value()
        assert shared_value["notify"] is True
        assert shared_value["mode"] == "replace"
        assert shared_value["initial"] == {
            "pattern": "identity",
            "scale": 2.0,
        }

        source_value = source.value()
        assert source_value["streams"] == ["a", "b"]
        assert "stream" not in source_value
        assert source_value["auxiliary"] == {"ready": "ready"}
        assert source_value["parameters"] == {"path": "asset.npy"}
        assert source_value["read_timeout"] == 0.5

        sink_value = sink.value()
        assert sink_value["outputs"] == ["b"]
        assert sink_value["auxiliary"] == {"ready": "ready"}
        assert sink_value["parameters"] == {"path": "asset.npy"}
        assert sink_value["consume_timeout"] == 0.5
    finally:
        shared.close()
        source.close()
        sink.close()


def test_main_window_can_switch_themes(qapp):
    window = MainWindow(theme_name="light")
    try:
        window._apply_theme("dark", persist=False)
        assert window.current_theme_name == "dark"
    finally:
        window.close()


def test_gui_main_loads_initial_document_argument(qapp, monkeypatch, tmp_path):
    path = tmp_path / "pipeline.yaml"
    document = default_document()
    document["shared_memory"] = [
        {
            "name": "camera_frame",
            "shape": [4, 4],
            "dtype": "float32",
            "storage": "cpu",
        }
    ]
    document["sources"] = [
        {
            "name": "simulated_camera",
            "kind": "example.simulated_camera",
            "stream": "camera_frame",
        }
    ]
    save_document(path, document)

    windows = []

    def _show(window) -> None:
        windows.append(window)

    monkeypatch.setattr(MainWindow, "show", _show)
    monkeypatch.setattr(QApplication, "exec", lambda self: 0)

    assert gui_main([str(path)]) == 0
    assert len(windows) == 1

    window = windows[0]
    try:
        assert window._current_path == path
        assert window._source_table.rowCount() == 1
        assert window.windowTitle().endswith(str(path))
    finally:
        window.close()


def test_main_window_stacks_kernels_under_shared_memory(qapp):
    window = MainWindow(theme_name="light")
    try:
        assert window._editor_splitter.orientation() == Qt.Vertical
        assert window._main_splitter.orientation() == Qt.Horizontal
        assert [
            window._pipeline_editor_tabs.tabText(index)
            for index in range(window._pipeline_editor_tabs.count())
        ] == ["Shared Memory", "Kernels"]
        assert [
            window._endpoint_editor_tabs.tabText(index)
            for index in range(window._endpoint_editor_tabs.count())
        ] == ["Sources", "Sinks"]
        assert window._pipeline_editor_tabs.currentIndex() == 0
        assert window._endpoint_editor_tabs.currentIndex() == 0
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


def test_main_window_exposes_runtime_tabs_for_all_component_types(qapp):
    window = MainWindow(theme_name="light")
    try:
        labels = [
            window._runtime_tabs.tabText(index)
            for index in range(window._runtime_tabs.count())
        ]
        assert labels == ["Workers", "Sources", "Sinks", "Synthetic"]
    finally:
        window.close()


def test_main_window_exposes_stream_actions_menu(qapp):
    window = MainWindow(theme_name="light")
    try:
        labels = [action.text() for action in window.menuBar().actions()]
        assert "Streams" in labels
        assert "Server" in labels
    finally:
        window.close()


def test_control_window_can_start_in_light_theme(qapp):
    window = ControlWindow(theme_name="light")
    try:
        assert window.current_theme_name == "light"
        assert window._tabs.tabText(0) == "State"
        assert window._tabs.tabText(1) == "Server"
        assert window._url_edit.minimumWidth() == window._FIELD_MIN_WIDTH
        assert window._token_edit.minimumWidth() == window._FIELD_MIN_WIDTH
        assert (
            window._config_path_edit.minimumWidth() == window._FIELD_MIN_WIDTH
        )
    finally:
        window.close()


def test_control_window_can_list_discovered_servers(qapp, monkeypatch):
    record = LocalControlServerRecord(
        pid=123,
        host="127.0.0.1",
        port=8765,
        token_required=False,
    )
    monkeypatch.setattr(
        "shmpipeline.gui.control.discover_local_servers",
        lambda: [record],
    )
    monkeypatch.setattr(
        ControlWindow,
        "_server_info_for_record",
        lambda self, candidate: {
            "state": "initialized",
            "config_path": "/tmp/pipeline.yaml",
        },
    )

    window = ControlWindow(theme_name="light")
    try:
        assert window._discovered_combo.count() == 1
        label = window._discovered_combo.itemText(0)
        assert "127.0.0.1:8765" in label
        assert "INITIALIZED" in label
        assert "pipeline.yaml" in label
    finally:
        window.close()


def test_control_window_updates_primary_button_and_badge(qapp):
    window = ControlWindow(theme_name="light")
    try:
        window._set_connection_status("connected")
        assert window._server_status_badge.text() == "CONNECTED"
        assert (
            window._theme.success in window._server_status_badge.styleSheet()
        )

        window._set_state_display("paused")
        assert window._primary_button.text() == "RESUME"
        assert window._state_badge.text() == "PAUSED"

        window._set_state_display("running")
        assert window._primary_button.text() == "PAUSE"
        assert window._state_badge.text() == "RUNNING"

        window._set_state_display("initialized")
        assert window._primary_button.text() == "START"
        assert window._state_badge.text() == "INITIALIZED"
    finally:
        window.close()


def test_main_window_build_auto_launches_local_server(qapp, monkeypatch):
    window = MainWindow(theme_name="light")

    class _FakeSession:
        def __init__(self) -> None:
            self.connection = type(
                "Conn",
                (),
                {
                    "display_name": "127.0.0.1:9000",
                    "base_url": "http://127.0.0.1:9000",
                    "is_local": True,
                },
            )()
            self.build_called = False

        def status(self):
            return {
                "state": "initialized",
                "placement_policy": "round-robin",
                "summary": {},
                "workers": {},
                "failures": [],
                "synthetic_sources": {},
            }

        def build(self):
            self.build_called = True
            return {"state": "built"}

        def close(self):
            return None

    fake_session = _FakeSession()

    def fake_start_local_server(*, show_feedback: bool) -> bool:
        window._manager = fake_session
        return True

    def fake_push_document_to_server(*, show_feedback: bool) -> bool:
        window._manager_dirty = False
        return True

    monkeypatch.setattr(window, "_start_local_server", fake_start_local_server)
    monkeypatch.setattr(
        window,
        "_push_document_to_server",
        fake_push_document_to_server,
    )
    monkeypatch.setattr(window, "_ensure_document_valid", lambda: True)

    try:
        window.build_pipeline()
        assert fake_session.build_called is True
    finally:
        window.close()


def test_main_window_formats_local_connection_errors(qapp):
    window = MainWindow(theme_name="light")
    try:
        message = window._format_connection_error(
            type(
                "Conn",
                (),
                {"base_url": "http://127.0.0.1:8765", "is_local": True},
            )(),
            RuntimeError("connection refused"),
        )
        assert "auto-launch a local server" in message
        assert "127.0.0.1:8765" in message
    finally:
        window.close()


def test_main_window_logs_errors_to_gui_output(qapp, monkeypatch):
    window = MainWindow(theme_name="light")
    monkeypatch.setattr(
        "shmpipeline.gui.app.QMessageBox.critical",
        lambda *args, **kwargs: None,
    )
    try:
        window._show_error("Build Failed", "example failure")
        assert "ERROR: Build Failed: example failure" in (
            window._validation_output.toPlainText()
        )
    finally:
        window.close()


def test_main_window_local_server_failure_message_includes_server_log(
    qapp,
    tmp_path,
):
    window = MainWindow(theme_name="light")
    log_path = tmp_path / "server.log"
    log_path.write_text("Traceback\nRuntimeError: boom", encoding="utf-8")
    window._managed_server_log_path = log_path
    try:
        message = window._format_local_server_failure(
            "The GUI-launched local control server exited before it became ready.",
            last_error="connection refused",
        )
        assert "Last connection error: connection refused" in message
        assert "RuntimeError: boom" in message
    finally:
        window.close()


def test_light_theme_styles_toolbar_explicitly(qapp):
    theme = themes_module.resolve_theme("light")
    stylesheet = themes_module.build_stylesheet(theme)

    assert "QToolBar" in stylesheet
    assert "QToolButton:pressed, QToolButton:checked" in stylesheet
    assert theme.alternate_base in stylesheet
    assert theme.highlight in stylesheet


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
    _patch_viewer_pyqtgraph(monkeypatch)
    fake_stream = _FakeStream()
    monkeypatch.setattr(
        viewers_module.pyshmem, "open", lambda *args, **kwargs: fake_stream
    )

    viewer = viewers_module.SharedMemoryViewer(
        {
            "name": "demo",
            "shape": [4, 4],
            "dtype": "float32",
            "storage": "cpu",
        }
    )
    try:
        viewer._timer.stop()
        assert viewer._image_plot.auto_range_calls == 1
        viewer.refresh()
        assert viewer._image_plot.auto_range_calls == 1
        assert viewer._stat_values["count"].text() == "2"
        assert viewer._stat_values["stream_avg"].text().endswith(" Hz")
        assert viewer._stat_values["stream_p99"].text().endswith(" Hz")
        assert viewer._stat_values["viewer_avg"].text().endswith(" Hz")
        assert viewer._status_label.isHidden()

        assert viewer._stats_action.isChecked()
        assert viewer._stats_action.shortcut().toString() == "Ctrl+I"
        assert viewer._details_action.shortcut().toString() == "Ctrl+,"
        assert viewer._fit_action.shortcut().toString() == "F"
        assert viewer._close_action.shortcut().toString() == "Ctrl+W"
        viewer._stats_action.setChecked(False)
        assert viewer._stats_group.isHidden()
        viewer._stats_action.setChecked(True)
        assert not viewer._stats_group.isHidden()

        viewer.show()
        qapp.processEvents()
        viewer._fit_action.trigger()
        assert viewer._image_plot.auto_range_calls == 2

        viewer._show_stream_details()
        details_dialog = viewer._stream_details_dialog
        assert details_dialog is not None
        assert details_dialog._value_labels["shape"].text() == "[4, 4]"
        assert details_dialog._value_labels["dtype"].text() == "float32"
        assert not details_dialog.isHidden()
        details_dialog.close()

        viewer._render_array(np.zeros((8, 6), dtype=np.float32))
        assert viewer._image_plot.auto_range_calls == 3
    finally:
        viewer.close()


def test_gpu_viewer_can_open_without_cpu_mirror(qapp, monkeypatch):
    _patch_viewer_pyqtgraph(monkeypatch)
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
        assert open_calls == [
            ("demo", {"gpu_device": "cuda:0", "readonly": True})
        ]
        viewer._show_stream_details()
        details_dialog = viewer._stream_details_dialog
        assert details_dialog is not None
        assert details_dialog._value_labels["gpu_device"].text() == "cuda:0"
        details_dialog.close()
    finally:
        viewer.close()


def test_main_window_runtime_status_text_includes_worker_health(qapp):
    window = MainWindow(theme_name="light")
    try:
        text = window._format_runtime_status_text(
            {
                "placement_policy": "round-robin",
                "summary": {
                    "active_workers": 1,
                    "idle_workers": 1,
                    "waiting_workers": 0,
                    "paused_workers": 0,
                    "failed_workers": 0,
                },
                "workers": {
                    "stage": {
                        "health": "active",
                        "idle_s": 0.12,
                        "last_metric_age_s": 0.05,
                        "avg_exec_us": 123.4,
                        "jitter_us_rms": 12.5,
                        "throughput_hz": 456.7,
                        "metrics_window": 42,
                    }
                },
                "failures": [],
                "synthetic_sources": {},
            }
        )
        assert "Workers: active=1 idle=1 waiting=0 paused=0 failed=0" in text
        assert "stage health=active" in text
        assert "idle_s=0.12" in text
        assert "metric_age_s=0.05" in text
    finally:
        window.close()


def test_main_window_prefers_remote_plugin_kinds(qapp):
    window = MainWindow(theme_name="light")
    try:
        window._server_info = {
            "source_kinds": ["remote.source"],
            "sink_kinds": ["remote.sink"],
            "kernel_kinds": ["remote.kernel"],
        }

        assert window._available_kinds(
            remote_key="source_kinds",
            fallback=lambda: ("local.source",),
        ) == ["remote.source"]
        assert window._available_kinds(
            remote_key="sink_kinds",
            fallback=lambda: ("local.sink",),
        ) == ["remote.sink"]
        assert window._available_kinds(
            remote_key="kernel_kinds",
            fallback=lambda: ("local.kernel",),
        ) == ["remote.kernel"]
    finally:
        window.close()


def test_main_window_refreshes_source_and_sink_tables(qapp):
    window = MainWindow(theme_name="light")
    try:
        window._document = {
            "shared_memory": [
                {
                    "name": "input_frame",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
                {
                    "name": "output_frame",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": "cpu",
                },
            ],
            "sources": [
                {
                    "name": "camera",
                    "kind": "example.camera",
                    "stream": "input_frame",
                    "poll_interval": 0.01,
                }
            ],
            "kernels": [
                {
                    "name": "copy_stage",
                    "kind": "cpu.copy",
                    "input": "input_frame",
                    "output": "output_frame",
                }
            ],
            "sinks": [
                {
                    "name": "display",
                    "kind": "example.display",
                    "stream": "output_frame",
                    "read_timeout": 0.1,
                    "pause_sleep": 0.01,
                }
            ],
        }

        window._refresh_all()

        assert window._source_table.rowCount() == 1
        assert window._sink_table.rowCount() == 1
        assert window._source_runtime_table.rowCount() == 1
        assert window._sink_runtime_table.rowCount() == 1
    finally:
        window.close()


def test_main_window_populates_source_and_sink_runtime_tables(qapp):
    window = MainWindow(theme_name="light")

    class _FakeSession:
        connection = type(
            "Conn",
            (),
            {
                "display_name": "127.0.0.1:9000",
                "base_url": "",
                "is_local": True,
            },
        )()

        def status(self):
            return {
                "state": "running",
                "placement_policy": "round-robin",
                "summary": {
                    "active_workers": 0,
                    "idle_workers": 0,
                    "waiting_workers": 0,
                    "paused_workers": 0,
                    "failed_workers": 0,
                    "active_sources": 1,
                    "failed_sources": 0,
                    "active_sinks": 1,
                    "failed_sinks": 0,
                },
                "workers": {},
                "sources": {
                    "camera": {
                        "alive": True,
                        "stream": "input_frame",
                        "frames_written": 12,
                        "effective_rate_hz": 48.5,
                        "last_error": None,
                    }
                },
                "sinks": {
                    "display": {
                        "alive": True,
                        "stream": "output_frame",
                        "frames_consumed": 11,
                        "effective_rate_hz": 47.2,
                        "last_error": None,
                    }
                },
                "failures": [],
                "synthetic_sources": {},
            }

        def close(self):
            return None

    try:
        window._document = {
            "shared_memory": [],
            "sources": [
                {
                    "name": "camera",
                    "kind": "example.camera",
                    "stream": "input_frame",
                }
            ],
            "kernels": [],
            "sinks": [
                {
                    "name": "display",
                    "kind": "example.display",
                    "stream": "output_frame",
                }
            ],
        }
        window._manager = _FakeSession()

        window._refresh_runtime_status()

        assert window._source_runtime_table.item(0, 3).text() == "True"
        assert window._source_runtime_table.item(0, 4).text() == "12"
        assert window._sink_runtime_table.item(0, 3).text() == "True"
        assert window._sink_runtime_table.item(0, 4).text() == "11"
    finally:
        window.close()


def test_main_window_shows_synthetic_inputs_in_source_runtime_table(qapp):
    window = MainWindow(theme_name="light")

    class _FakeSession:
        connection = type(
            "Conn",
            (),
            {
                "display_name": "127.0.0.1:9000",
                "base_url": "",
                "is_local": True,
            },
        )()

        def status(self):
            return {
                "state": "running",
                "placement_policy": "round-robin",
                "summary": {
                    "active_workers": 0,
                    "idle_workers": 0,
                    "waiting_workers": 0,
                    "paused_workers": 0,
                    "failed_workers": 0,
                    "active_sources": 0,
                    "failed_sources": 0,
                    "active_sinks": 0,
                    "failed_sinks": 0,
                },
                "workers": {},
                "sources": {},
                "sinks": {},
                "failures": [],
                "synthetic_sources": {
                    "input_frame": {
                        "pattern": "sine",
                        "alive": True,
                        "frames_written": 7,
                        "effective_rate_hz": 15.5,
                        "last_error": None,
                    }
                },
            }

        def close(self):
            return None

    try:
        window._document = {
            "shared_memory": [
                {
                    "name": "input_frame",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": "cpu",
                }
            ],
            "sources": [],
            "kernels": [],
            "sinks": [],
        }
        window._manager = _FakeSession()

        window._refresh_runtime_status()

        assert window._source_runtime_table.rowCount() == 1
        assert window._source_runtime_table.item(0, 0).text() == (
            "synthetic:input_frame"
        )
        assert window._source_runtime_table.item(0, 1).text() == (
            "synthetic.sine"
        )
        assert window._source_runtime_table.item(0, 2).text() == "input_frame"
        assert window._source_runtime_table.item(0, 3).text() == "True"
        assert window._source_runtime_table.item(0, 4).text() == "7"
        assert window._synthetic_table.rowCount() == 1
        assert "Sources A/F: 1/0" in window._status_label.text()
        assert "synthetic:input_frame" in window._runtime_output.toPlainText()
    finally:
        window.close()


def test_main_window_shows_synthetic_inputs_in_source_editor_table(qapp):
    window = MainWindow(theme_name="light")

    class _FakeSession:
        connection = type(
            "Conn",
            (),
            {
                "display_name": "127.0.0.1:9000",
                "base_url": "",
                "is_local": True,
            },
        )()

        def status(self):
            return {
                "state": "running",
                "placement_policy": "round-robin",
                "summary": {
                    "active_workers": 0,
                    "idle_workers": 0,
                    "waiting_workers": 0,
                    "paused_workers": 0,
                    "failed_workers": 0,
                    "active_sources": 0,
                    "failed_sources": 0,
                    "active_sinks": 0,
                    "failed_sinks": 0,
                },
                "workers": {},
                "sources": {},
                "sinks": {},
                "failures": [],
                "synthetic_sources": {
                    "input_frame": {
                        "pattern": "sine",
                        "alive": True,
                        "frames_written": 7,
                        "effective_rate_hz": 15.5,
                        "requested_rate_hz": 15.5,
                        "last_error": None,
                    }
                },
            }

        def synthetic_input_status(self):
            return self.status()["synthetic_sources"]

        def close(self):
            return None

    try:
        window._document = {
            "shared_memory": [
                {
                    "name": "input_frame",
                    "shape": [4],
                    "dtype": "float32",
                    "storage": "cpu",
                }
            ],
            "sources": [],
            "kernels": [],
            "sinks": [],
        }
        window._manager = _FakeSession()

        window._refresh_runtime_status()

        assert window._source_table.rowCount() == 1
        assert (
            window._source_table.item(0, 0).text() == "synthetic:input_frame"
        )
        assert window._source_table.item(0, 1).text() == "synthetic.sine"
        assert window._source_table.item(0, 2).text() == "input_frame"
        assert window._source_table.item(0, 3).text() == "15.50 Hz"
    finally:
        window.close()


def test_gui_local_launch_config_stays_beside_selected_yaml(
    qapp, tmp_path, monkeypatch
):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    selected_path = config_dir / "pipeline.yaml"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    window = MainWindow(theme_name="light")
    generated = None
    try:
        window._document = default_document()
        window._current_path = selected_path
        generated = window._prepare_local_server_config()
        assert generated.parent == config_dir.resolve()
        assert generated.is_file()
    finally:
        window.close()
        if generated is not None:
            generated.unlink(missing_ok=True)
