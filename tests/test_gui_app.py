from __future__ import annotations

import os

import numpy as np
import pytest

from shmpipeline.control.discovery import LocalControlServerRecord

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
except Exception:  # pragma: no cover - exercised when Qt is unavailable
    Qt = None
    QApplication = None

GUI_IMPORT_ERROR: Exception | None = None

try:
    from shmpipeline.gui import themes as themes_module
    from shmpipeline.gui import viewers as viewers_module
    from shmpipeline.gui.app import MainWindow, SyntheticInputDialog
    from shmpipeline.gui.control import ControlWindow
except Exception as exc:  # pragma: no cover - GUI stack unavailable
    viewers_module = None
    themes_module = None
    MainWindow = None
    SyntheticInputDialog = None
    ControlWindow = None
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
    app.closeAllWindows()
    app.processEvents()
    for widget in QApplication.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    app.processEvents()


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
