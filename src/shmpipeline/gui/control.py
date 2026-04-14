"""Minimal server and state-machine control GUI for shmpipeline."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from shmpipeline.control.discovery import (
    LocalControlServerRecord,
    discover_local_servers,
    terminate_local_server,
)
from shmpipeline.gui.remote import RemotePipelineSession, ServerConnection
from shmpipeline.gui.themes import apply_application_theme, resolve_theme


class ControlWindow(QMainWindow):
    """Minimal window for server handling and pipeline state-machine control."""

    _FIELD_MIN_WIDTH = 300

    def __init__(self, *, theme_name: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("shmpipeline Control")
        self.resize(600, 400)
        self.menuBar().setNativeMenuBar(False)
        self._settings = QSettings("shmpipeline", "control-gui")
        shared_gui_settings = QSettings("shmpipeline", "gui")
        default_theme = shared_gui_settings.value("theme", "light", str)
        self._theme = resolve_theme(
            theme_name or self._settings.value("theme", default_theme, str)
        )
        self._session: RemotePipelineSession | None = None
        self._reported_failures: set[tuple[str, str]] = set()
        self._discovered_records: list[LocalControlServerRecord] = []
        self._last_refresh_error: str | None = None
        self._launch_log_path: Path | None = None
        self._state_value = "DISCONNECTED"
        self._connection_state = "DISCONNECTED"

        self._url_edit = QLineEdit(
            self._settings.value("server_url", "http://127.0.0.1:8765", str)
        )
        self._url_edit.setMinimumWidth(self._FIELD_MIN_WIDTH)
        self._token_edit = QLineEdit(
            self._settings.value("server_token", "", str)
        )
        self._token_edit.setEchoMode(QLineEdit.Password)
        self._token_edit.setMinimumWidth(self._FIELD_MIN_WIDTH)
        self._config_path_edit = QLineEdit(
            self._settings.value("control_config_path", "", str)
        )
        self._config_path_edit.setPlaceholderText("/path/to/pipeline.yaml")
        self._config_path_edit.setMinimumWidth(self._FIELD_MIN_WIDTH)
        self._discovered_combo = QComboBox()
        self._discovered_combo.setMinimumContentsLength(28)

        self._server_value_label = QLabel("127.0.0.1:8765")
        self._server_status_badge = QLabel("DISCONNECTED")
        self._server_status_badge.setAlignment(Qt.AlignCenter)
        self._config_value_label = QLabel("N/A")
        self._config_value_label.setWordWrap(True)
        self._state_badge = QLabel("DISCONNECTED")
        self._state_badge.setAlignment(Qt.AlignCenter)
        self._state_badge.setMinimumHeight(96)
        self._state_hint_label = QLabel(
            "Use START, PAUSE, STOP, or TEARDOWN to drive the pipeline."
        )
        self._state_hint_label.setAlignment(Qt.AlignCenter)
        self._log_output = QPlainTextEdit()
        self._log_output.setReadOnly(True)

        self._primary_button = QPushButton("START")
        self._primary_button.clicked.connect(self.advance_state_machine)
        self._stop_button = QPushButton("STOP")
        self._stop_button.clicked.connect(self.stop_pipeline)
        self._teardown_button = QPushButton("TEARDOWN")
        self._teardown_button.clicked.connect(self.teardown_pipeline)

        self._build_ui()
        self._build_actions()
        self._apply_theme(self._theme.name, persist=False)
        self.discover_servers()
        self.refresh_status()

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self.refresh_status)
        self._status_timer.start()

    @property
    def current_theme_name(self) -> str:
        """Return the active theme name."""
        return self._theme.name

    def _build_ui(self) -> None:
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_state_tab(), "State")
        self._tabs.addTab(self._build_server_tab(), "Server")

        log_box = QWidget()
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QLabel("Session Log"))
        log_layout.addWidget(self._log_output)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self._tabs, 0)
        layout.addWidget(log_box, 1)
        self.setCentralWidget(central)

    def _build_server_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        discovery_row = QHBoxLayout()
        discovery_row.addWidget(self._discovered_combo, 1)
        for label, handler in [
            ("Discover", self.discover_servers),
            ("Connect", self.connect_selected_server),
            ("Kill", self.kill_selected_server),
        ]:
            button = QPushButton(label)
            button.clicked.connect(handler)
            discovery_row.addWidget(button)
        layout.addLayout(discovery_row)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.addRow("Token", self._token_edit)
        form.addRow(
            "Server URL",
            self._row_widget(
                self._url_edit,
                [
                    ("Connect", self.connect_from_fields),
                    ("Launch", self.launch_local_server),
                    ("Disconnect", self.disconnect_from_server),
                ],
            ),
        )
        form.addRow(
            "Config",
            self._row_widget(
                self._config_path_edit,
                [
                    ("Browse", self.browse_config_path),
                    ("Use On Server", self.change_server_config),
                ],
            ),
        )
        layout.addLayout(form)
        layout.addStretch(1)
        return tab

    def _build_state_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        server_row = QHBoxLayout()
        server_label = QLabel("Server:")
        server_label.setMinimumWidth(server_label.sizeHint().width())
        status_label = QLabel("Status:")
        status_label.setMinimumWidth(status_label.sizeHint().width())
        server_row.addWidget(server_label)
        server_row.addWidget(self._server_value_label, 1)
        server_row.addWidget(status_label)
        server_row.addWidget(self._server_status_badge)
        layout.addLayout(server_row)

        config_row = QHBoxLayout()
        config_label = QLabel("Config:")
        config_label.setMinimumWidth(server_label.sizeHint().width())
        config_row.addWidget(config_label)
        config_row.addWidget(self._config_value_label, 1)
        layout.addLayout(config_row)

        layout.addSpacing(8)
        layout.addWidget(self._state_badge)
        layout.addWidget(self._state_hint_label)
        command_row = QHBoxLayout()
        command_row.addWidget(self._primary_button)
        command_row.addWidget(self._stop_button)
        command_row.addWidget(self._teardown_button)
        layout.addLayout(command_row)
        layout.addStretch(1)
        return tab

    def _row_widget(
        self, field: QWidget, actions: list[tuple[str, Any]]
    ) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(field, 1)
        for label, handler in actions:
            button = QPushButton(label)
            button.clicked.connect(handler)
            row.addWidget(button)
        return container

    def _build_actions(self) -> None:
        view_menu = self.menuBar().addMenu("View")
        theme_menu = view_menu.addMenu("Theme")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        for theme_name, label in (("light", "Light"), ("dark", "Dark")):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(self._theme.name == theme_name)
            action.triggered.connect(
                lambda checked, selected=theme_name: (
                    checked and self._apply_theme(selected)
                )
            )
            theme_group.addAction(action)
            theme_menu.addAction(action)

    def _apply_theme(self, theme_name: str, *, persist: bool = True) -> None:
        app = QApplication.instance()
        if app is None:  # pragma: no cover - GUI runtime only
            return
        self._theme = apply_application_theme(app, theme_name)
        if persist:
            self._settings.setValue("theme", self._theme.name)
        self._apply_connection_badge_style(self._connection_state)
        self._apply_state_badge_style(self._state_value)

    def _configured_server_name(self) -> str:
        try:
            return self._build_connection().display_name
        except Exception:
            return self._url_edit.text().strip() or "N/A"

    def _set_connection_status(self, status: str) -> None:
        self._connection_state = str(status).upper()
        self._server_status_badge.setText(self._connection_state)
        self._apply_connection_badge_style(self._connection_state)

    def _apply_connection_badge_style(self, status: str) -> None:
        palette = {
            "CONNECTED": (self._theme.success_bg, self._theme.success),
            "ERROR": (self._theme.error_bg, self._theme.error),
            "DISCONNECTED": (self._theme.button, self._theme.muted_text),
        }
        background, color = palette.get(
            status,
            (self._theme.alternate_base, self._theme.text),
        )
        self._server_status_badge.setStyleSheet(
            "QLabel {"
            f" background-color: {background};"
            f" color: {color};"
            f" border: 1px solid {self._theme.border};"
            " border-radius: 8px;"
            " font-size: 12px;"
            " font-weight: 700;"
            " letter-spacing: 0.5px;"
            " padding: 4px 10px;"
            "}"
        )

    def _append_log_message(self, level: str, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {level}: {message}"
        self._log_output.moveCursor(QTextCursor.End)
        if self._log_output.toPlainText():
            self._log_output.insertPlainText("\n")
        self._log_output.insertPlainText(entry)
        self._log_output.moveCursor(QTextCursor.End)
        self._log_output.ensureCursorVisible()

    def _log_info(self, message: str) -> None:
        self._append_log_message("INFO", message)

    def _log_error(self, title: str, error: Exception | str) -> None:
        self._append_log_message("ERROR", f"{title}: {error}")

    def _show_error(self, title: str, error: Exception | str) -> None:
        self._log_error(title, error)
        QMessageBox.critical(self, title, str(error))

    def _selected_discovered_record(self) -> LocalControlServerRecord | None:
        index = self._discovered_combo.currentIndex()
        if index < 0 or index >= len(self._discovered_records):
            return None
        return self._discovered_records[index]

    def _build_connection(self) -> ServerConnection:
        return ServerConnection.from_values(
            self._url_edit.text(),
            self._token_edit.text(),
        )

    def _connect(
        self, connection: ServerConnection, *, announce: bool = True
    ) -> bool:
        session = None
        try:
            session = RemotePipelineSession(connection)
            session.info()
        except Exception as exc:
            if session is not None:
                session.close()
            self._show_error("Connection Failed", exc)
            return False

        self.disconnect_from_server(announce=False)
        self._session = session
        self._settings.setValue("server_url", connection.base_url)
        self._settings.setValue("server_token", connection.token or "")
        if announce:
            self._log_info(f"Connected to {connection.display_name}.")
        self.refresh_status()
        return True

    def connect_from_fields(self) -> None:
        self._connect(self._build_connection())

    def browse_config_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pipeline Config",
            str(Path.cwd()),
            "YAML Files (*.yaml *.yml)",
        )
        if not path:
            return
        self._config_path_edit.setText(path)
        self._settings.setValue("control_config_path", path)
        self._log_info(f"Selected config file {path}.")

    def _server_info_for_record(
        self,
        record: LocalControlServerRecord,
    ) -> dict[str, Any] | None:
        token = self._token_edit.text().strip() or None
        if record.token_required and token is None:
            return None
        session = None
        try:
            session = RemotePipelineSession(
                ServerConnection.from_values(record.base_url, token),
                timeout=0.5,
            )
            return session.info()
        except Exception:
            return None
        finally:
            if session is not None:
                session.close()

    def _describe_discovered_record(
        self,
        record: LocalControlServerRecord,
    ) -> str:
        parts = [record.base_url, f"pid {record.pid}"]
        if record.token_required:
            parts.append("token")
        info = self._server_info_for_record(record)
        if info is not None:
            parts.append(str(info.get("state", "unknown")).upper())
            config_path = info.get("config_path")
            if config_path:
                parts.append(Path(config_path).name)
        return " | ".join(parts)

    def discover_servers(self) -> None:
        self._discovered_records = discover_local_servers()
        current_base_url = (
            self._session.connection.base_url
            if self._session is not None
            else None
        )
        self._discovered_combo.clear()
        if not self._discovered_records:
            self._discovered_combo.addItem("No local servers discovered")
            self._discovered_combo.setEnabled(False)
            self._log_info("Discovered 0 local servers.")
            return
        self._discovered_combo.setEnabled(True)
        selected_index = 0
        for index, record in enumerate(self._discovered_records):
            self._discovered_combo.addItem(
                self._describe_discovered_record(record)
            )
            if current_base_url == record.base_url:
                selected_index = index
        self._discovered_combo.setCurrentIndex(selected_index)
        self._log_info(
            f"Discovered {len(self._discovered_records)} local servers."
        )

    def connect_selected_server(self) -> None:
        record = self._selected_discovered_record()
        if record is None:
            self._show_error(
                "Connect Failed", "Select a discovered server first."
            )
            return
        self._url_edit.setText(record.base_url)
        if record.token_required and not self._token_edit.text().strip():
            self._show_error(
                "Connect Failed",
                "The selected server requires a bearer token.",
            )
            return
        self._connect(
            ServerConnection.from_values(
                record.base_url, self._token_edit.text()
            )
        )

    def _read_launch_log(self) -> str:
        if self._launch_log_path is None:
            return ""
        try:
            text = self._launch_log_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).strip()
        except Exception:
            return ""
        if len(text) <= 8000:
            return text
        return "...\n" + text[-8000:]

    def launch_local_server(self) -> None:
        config_path = self._config_path_edit.text().strip()
        if not config_path:
            self._show_error("Launch Failed", "Choose a config file first.")
            return
        config_file = Path(config_path).expanduser().resolve()
        if not config_file.is_file():
            self._show_error(
                "Launch Failed", f"Config file not found: {config_file}"
            )
            return

        connection = self._build_connection()
        if not connection.is_local:
            self._show_error(
                "Launch Failed",
                "The minimal control GUI only launches local servers.",
            )
            return

        parsed = urlparse(connection.base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 8765
        self._settings.setValue("control_config_path", str(config_file))

        with tempfile.NamedTemporaryFile(
            prefix="shmpipeline-control-gui-",
            suffix=".log",
            delete=False,
        ) as handle:
            self._launch_log_path = Path(handle.name)
        command = [
            sys.executable,
            "-m",
            "shmpipeline.cli",
            "--log-level",
            "ERROR",
            "serve",
            str(config_file),
            "--host",
            host,
            "--port",
            str(port),
        ]
        token = connection.token
        if token is not None:
            command.extend(["--token", token])

        self._log_info(
            f"Launching local server at http://{host}:{port} using {config_file}."
        )
        try:
            with self._launch_log_path.open(
                "w", encoding="utf-8"
            ) as log_handle:
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        except Exception as exc:
            self._show_error("Launch Failed", exc)
            return

        deadline = time.monotonic() + 5.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                log_text = self._read_launch_log()
                message = "The launched local control server exited before it became ready."
                if log_text:
                    message += f"\n\nServer log:\n{log_text}"
                self._show_error("Launch Failed", message)
                self.discover_servers()
                return
            session = None
            try:
                session = RemotePipelineSession(connection, timeout=0.5)
                session.info()
            except Exception as exc:
                last_error = exc
                if session is not None:
                    session.close()
                time.sleep(0.1)
                continue
            else:
                session.close()
                self.discover_servers()
                self._connect(connection, announce=False)
                self._log_info(
                    f"Local server is ready at {connection.display_name}."
                )
                return

        process.terminate()
        try:
            process.wait(timeout=2.0)
        except Exception:
            process.kill()
            process.wait(timeout=2.0)
        log_text = self._read_launch_log()
        message = "Timed out waiting for the launched local control server to accept connections."
        if last_error is not None:
            message += f"\n\nLast connection error: {last_error}"
        if log_text:
            message += f"\n\nServer log:\n{log_text}"
        self._show_error("Launch Failed", message)
        self.discover_servers()

    def kill_selected_server(self) -> None:
        record = self._selected_discovered_record()
        if record is None:
            self._show_error(
                "Kill Failed", "Select a discovered server first."
            )
            return
        try:
            terminate_local_server(record)
        except Exception as exc:
            self._show_error("Kill Failed", exc)
            return
        if (
            self._session is not None
            and self._session.connection.base_url == record.base_url
        ):
            self.disconnect_from_server(announce=False)
        self._log_info(
            f"Sent termination signal to {record.base_url} (pid {record.pid})."
        )
        QTimer.singleShot(500, self.discover_servers)

    def disconnect_from_server(self, *, announce: bool = True) -> None:
        if self._session is None:
            self._server_value_label.setText(self._configured_server_name())
            self._config_value_label.setText("N/A")
            self._set_connection_status("DISCONNECTED")
            self._set_state_display("DISCONNECTED")
            return
        base_url = self._session.connection.base_url
        try:
            self._session.close()
        finally:
            self._session = None
            self._reported_failures.clear()
            self._last_refresh_error = None
            self._server_value_label.setText(self._configured_server_name())
            self._config_value_label.setText("N/A")
            self._set_connection_status("DISCONNECTED")
            self._set_state_display("DISCONNECTED")
            if announce:
                self._log_info(f"Disconnected from {base_url}.")

    def change_server_config(self) -> None:
        config_path = self._config_path_edit.text().strip()
        if not config_path:
            self._show_error(
                "Change Config Failed", "Choose a config file first."
            )
            return
        config_file = Path(config_path).expanduser().resolve()
        if not config_file.is_file():
            self._show_error(
                "Change Config Failed",
                f"Config file not found: {config_file}",
            )
            return
        if self._session is None and not self._connect(
            self._build_connection(), announce=False
        ):
            return
        try:
            assert self._session is not None
            payload = self._session.load_document_path(str(config_file))
        except Exception as exc:
            self._show_error("Change Config Failed", exc)
            return
        self._settings.setValue("control_config_path", str(config_file))
        self._log_info(
            "Server config changed to "
            f"{payload.get('config_path', str(config_file))}."
        )
        self.discover_servers()
        self.refresh_status()

    def _set_state_display(self, state: str) -> None:
        self._state_value = str(state).upper()
        self._state_badge.setText(self._state_value)
        self._apply_state_badge_style(self._state_value)
        if self._state_value == "RUNNING":
            self._primary_button.setText("PAUSE")
            self._primary_button.setEnabled(True)
        elif self._state_value == "PAUSED":
            self._primary_button.setText("RESUME")
            self._primary_button.setEnabled(True)
        elif self._state_value in {"FAILED", "CONNECTION ERROR"}:
            self._primary_button.setText("START")
            self._primary_button.setEnabled(False)
        else:
            self._primary_button.setText("START")
            self._primary_button.setEnabled(True)
        connected = self._state_value not in {
            "DISCONNECTED",
            "CONNECTION ERROR",
        }
        self._stop_button.setEnabled(connected)
        self._teardown_button.setEnabled(connected)

    def _apply_state_badge_style(self, state: str) -> None:
        palette = {
            "RUNNING": (self._theme.success_bg, self._theme.success),
            "PAUSED": (self._theme.highlight, self._theme.highlight_text),
            "FAILED": (self._theme.error_bg, self._theme.error),
            "CONNECTION ERROR": (self._theme.error_bg, self._theme.error),
            "INITIALIZED": (self._theme.alternate_base, self._theme.accent),
            "BUILT": (self._theme.alternate_base, self._theme.accent),
            "STOPPED": (self._theme.button, self._theme.muted_text),
            "DISCONNECTED": (self._theme.button, self._theme.muted_text),
        }
        background, color = palette.get(
            state,
            (self._theme.alternate_base, self._theme.text),
        )
        self._state_badge.setStyleSheet(
            "QLabel {"
            f" background-color: {background};"
            f" color: {color};"
            f" border: 1px solid {self._theme.border};"
            " border-radius: 10px;"
            " font-size: 28px;"
            " font-weight: 700;"
            " letter-spacing: 1px;"
            " padding: 12px;"
            "}"
        )
        self._state_hint_label.setStyleSheet(
            f"color: {self._theme.muted_text};"
        )

    def _report_failures(self, status: dict[str, Any]) -> None:
        active_failures: set[tuple[str, str]] = set()
        for failure in status.get("failures", []):
            kernel = str(failure.get("kernel") or "unknown")
            error = str(failure.get("error") or "unknown error")
            key = (kernel, error)
            active_failures.add(key)
            if key not in self._reported_failures:
                self._show_error("Pipeline Error", f"{kernel}: {error}")
        self._reported_failures = active_failures

    def refresh_status(self) -> None:
        if self._session is None:
            self._server_value_label.setText(self._configured_server_name())
            self._config_value_label.setText("N/A")
            self._set_connection_status("DISCONNECTED")
            self._set_state_display("DISCONNECTED")
            return
        try:
            status = self._session.status()
        except Exception as exc:
            message = str(exc)
            if message != self._last_refresh_error:
                self._log_error("Status Refresh Failed", message)
                self._last_refresh_error = message
            self._server_value_label.setText(
                self._session.connection.display_name
            )
            self._config_value_label.setText("UNAVAILABLE")
            self._set_connection_status("ERROR")
            self._set_state_display("CONNECTION ERROR")
            return

        self._last_refresh_error = None
        self._report_failures(status)
        state = str(status.get("state", "unknown")).upper()
        config_path = str(status.get("config_path") or "N/A")
        self._server_value_label.setText(self._session.connection.display_name)
        self._config_value_label.setText(config_path)
        self._set_connection_status("CONNECTED")
        self._set_state_display(state)

    def _ensure_session(self) -> bool:
        if self._session is not None:
            return True
        return self._connect(self._build_connection(), announce=False)

    def advance_state_machine(self) -> None:
        if not self._ensure_session():
            return
        try:
            assert self._session is not None
            state = self._session.state.value.upper()
            if state == "RUNNING":
                self._session.pause()
                self._log_info("Pipeline paused.")
            elif state == "PAUSED":
                self._session.resume()
                self._log_info("Pipeline resumed.")
            else:
                self._session.start()
                self._log_info("Pipeline started.")
        except Exception as exc:
            self._show_error("State Change Failed", exc)
            return
        self.refresh_status()

    def stop_pipeline(self) -> None:
        if not self._ensure_session():
            return
        try:
            assert self._session is not None
            self._session.stop(force=True)
        except Exception as exc:
            self._show_error("Stop Failed", exc)
            return
        self._log_info("Pipeline stopped.")
        self.refresh_status()

    def teardown_pipeline(self) -> None:
        if not self._ensure_session():
            return
        try:
            assert self._session is not None
            self._session.shutdown(force=True)
        except Exception as exc:
            self._show_error("Teardown Failed", exc)
            return
        self._log_info("Pipeline torn down.")
        self.refresh_status()

    def closeEvent(
        self,
        event: QCloseEvent,
    ) -> None:  # pragma: no cover - GUI runtime only
        self._status_timer.stop()
        self.disconnect_from_server(announce=False)
        super().closeEvent(event)


def main() -> int:
    """Launch the minimal control GUI."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("shmpipeline Control GUI")
    app.setOrganizationName("shmpipeline")
    window = ControlWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - GUI entry point
    raise SystemExit(main())
