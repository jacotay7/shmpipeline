"""Lightweight remote-control GUI for one shmpipeline server."""

from __future__ import annotations

import sys
from typing import Any

from PySide6.QtCore import QSettings, QTimer
from PySide6.QtGui import QColor, QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from shmpipeline.gui.remote import RemotePipelineSession, ServerConnection
from shmpipeline.gui.themes import apply_application_theme, resolve_theme


class ControlWindow(QMainWindow):
    """Small control-surface window for one remote pipeline server."""

    def __init__(self, *, theme_name: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("shmpipeline Control")
        self.resize(1100, 720)
        self._settings = QSettings("shmpipeline", "control-gui")
        self._theme = resolve_theme(
            theme_name or self._settings.value("theme", "light", str)
        )
        self._session: RemotePipelineSession | None = None
        self._reported_failures: set[tuple[str, str]] = set()

        self._url_edit = QLineEdit(
            self._settings.value(
                "server_url", "http://127.0.0.1:8765", str
            )
        )
        self._token_edit = QLineEdit(
            self._settings.value("server_token", "", str)
        )
        self._token_edit.setEchoMode(QLineEdit.Password)
        self._status_label = QLabel("Server: disconnected")
        self._runtime_output = QPlainTextEdit()
        self._runtime_output.setReadOnly(True)

        self._worker_table = QTableWidget(0, 7)
        self._worker_table.setHorizontalHeaderLabels(
            [
                "Kernel",
                "Health",
                "PID",
                "Frames",
                "Avg us",
                "Hz",
                "Exit Code",
            ]
        )

        self._build_ui()
        self._apply_theme(self._theme.name, persist=False)
        self.refresh_status()

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(300)
        self._status_timer.timeout.connect(self.refresh_status)
        self._status_timer.start()

    @property
    def current_theme_name(self) -> str:
        """Return the active theme name."""
        return self._theme.name

    def _build_ui(self) -> None:
        top_form = QFormLayout()
        top_form.addRow("Server URL", self._url_edit)
        top_form.addRow("Bearer token", self._token_edit)

        connection_row = QHBoxLayout()
        for label, handler in [
            ("Connect", self.connect_to_server),
            ("Disconnect", self.disconnect_from_server),
            ("Refresh", self.refresh_status),
        ]:
            button = QPushButton(label)
            button.clicked.connect(handler)
            connection_row.addWidget(button)

        command_row = QHBoxLayout()
        for label, handler in [
            ("Build", self.build_pipeline),
            ("Start", self.start_pipeline),
            ("Pause", self.pause_pipeline),
            ("Resume", self.resume_pipeline),
            ("Stop", self.stop_pipeline),
            ("Shutdown", self.shutdown_pipeline),
        ]:
            button = QPushButton(label)
            button.clicked.connect(handler)
            command_row.addWidget(button)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(top_form)
        layout.addLayout(connection_row)
        layout.addLayout(command_row)
        layout.addWidget(self._status_label)
        layout.addWidget(self._worker_table, 3)
        layout.addWidget(self._runtime_output, 2)
        self.setCentralWidget(central)

    def _apply_theme(self, theme_name: str, *, persist: bool = True) -> None:
        app = QApplication.instance()
        if app is None:  # pragma: no cover - GUI runtime only
            return
        self._theme = apply_application_theme(app, theme_name)
        if persist:
            self._settings.setValue("theme", self._theme.name)

    def _show_error(self, title: str, error: Exception | str) -> None:
        QMessageBox.critical(self, title, str(error))

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

    def _format_float(self, value: Any) -> str:
        if value is None or value == "":
            return ""
        try:
            return f"{float(value):.2f}"
        except Exception:
            return str(value)

    def _format_microseconds(self, value: Any) -> str:
        if value is None or value == "":
            return ""
        try:
            return str(int(round(float(value))))
        except Exception:
            return str(value)

    def _format_runtime_status_text(
        self,
        status: dict[str, Any] | None,
    ) -> str:
        if status is None:
            return "Runtime: disconnected"
        lines = [
            f"State: {status.get('state', 'unknown')}",
            f"Placement: {status.get('placement_policy', 'n/a')}",
        ]
        summary = status.get("summary", {})
        lines.append(
            "Workers: "
            f"active={summary.get('active_workers', 0)} "
            f"idle={summary.get('idle_workers', 0)} "
            f"waiting={summary.get('waiting_workers', 0)} "
            f"paused={summary.get('paused_workers', 0)} "
            f"failed={summary.get('failed_workers', 0)}"
        )
        failures = status.get("failures", [])
        if failures:
            lines.append("Failures:")
            for failure in failures:
                lines.append(
                    f"- {failure.get('kernel')}: {failure.get('error')}"
                )
        else:
            lines.append("Failures: none")
        return "\n".join(lines)

    def _run_command(
        self,
        command,
        *,
        title: str,
        success_message: str,
    ) -> None:
        if self._session is None:
            self._show_error(title, "Connect to a control server first.")
            return
        try:
            command()
        except Exception as exc:
            self._show_error(title, exc)
            return
        self._runtime_output.setPlainText(success_message)
        self.refresh_status()

    def connect_to_server(self) -> bool:
        session = None
        try:
            connection = ServerConnection.from_values(
                self._url_edit.text(),
                self._token_edit.text(),
            )
            session = RemotePipelineSession(connection)
            session.info()
        except Exception as exc:
            if session is not None:
                session.close()
            self._show_error("Connection Failed", exc)
            return False

        self.disconnect_from_server()
        self._session = session
        self._settings.setValue("server_url", connection.base_url)
        self._settings.setValue("server_token", connection.token or "")
        self.refresh_status()
        return True

    def disconnect_from_server(self) -> None:
        if self._session is None:
            self._status_label.setText("Server: disconnected")
            return
        try:
            self._session.close()
        finally:
            self._session = None
            self._reported_failures.clear()
            self._worker_table.setRowCount(0)
            self._runtime_output.setPlainText("Runtime: disconnected")
            self._status_label.setText("Server: disconnected")

    def refresh_status(self) -> None:
        if self._session is None:
            self._worker_table.setRowCount(0)
            self._runtime_output.setPlainText("Runtime: disconnected")
            self._status_label.setText("Server: disconnected")
            return
        try:
            status = self._session.status()
        except Exception as exc:
            self._runtime_output.setPlainText(
                f"Runtime status failed: {exc}"
            )
            self._status_label.setText(
                f"Server: {self._session.connection.display_name} | connection error"
            )
            return

        self._report_failures(status)
        workers = status.get("workers", {})
        self._worker_table.setRowCount(len(workers))
        for row, (kernel_name, worker) in enumerate(sorted(workers.items())):
            values = [
                kernel_name,
                str(worker.get("health", "")),
                str(worker.get("pid", "")),
                str(worker.get("frames_processed", 0)),
                self._format_microseconds(worker.get("avg_exec_us")),
                self._format_float(worker.get("throughput_hz")),
                str(worker.get("exitcode", "")),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 1 and value in {"active", "idle", "waiting"}:
                    item.setBackground(QColor(self._theme.success_bg))
                if column == 6 and value not in {"", "None", "0"}:
                    item.setBackground(QColor(self._theme.error_bg))
                self._worker_table.setItem(row, column, item)
        self._worker_table.resizeColumnsToContents()

        summary = status.get("summary", {})
        self._status_label.setText(
            f"Server: {self._session.connection.display_name} | "
            f"State: {status.get('state', 'unknown')} | "
            f"Active: {summary.get('active_workers', 0)} | "
            f"Failed: {summary.get('failed_workers', 0)}"
        )
        self._runtime_output.setPlainText(
            self._format_runtime_status_text(status)
        )

    def build_pipeline(self) -> None:
        self._run_command(
            self._session.build if self._session is not None else lambda: None,
            title="Build Failed",
            success_message="Pipeline built.",
        )

    def start_pipeline(self) -> None:
        self._run_command(
            self._session.start if self._session is not None else lambda: None,
            title="Start Failed",
            success_message="Pipeline started.",
        )

    def pause_pipeline(self) -> None:
        self._run_command(
            self._session.pause if self._session is not None else lambda: None,
            title="Pause Failed",
            success_message="Pipeline paused.",
        )

    def resume_pipeline(self) -> None:
        self._run_command(
            self._session.resume if self._session is not None else lambda: None,
            title="Resume Failed",
            success_message="Pipeline resumed.",
        )

    def stop_pipeline(self) -> None:
        self._run_command(
            (
                (lambda: self._session.stop(force=True))
                if self._session is not None
                else lambda: None
            ),
            title="Stop Failed",
            success_message="Pipeline stopped.",
        )

    def shutdown_pipeline(self) -> None:
        self._run_command(
            (
                (lambda: self._session.shutdown(force=True))
                if self._session is not None
                else lambda: None
            ),
            title="Shutdown Failed",
            success_message="Pipeline shut down.",
        )

    def closeEvent(
        self,
        event: QCloseEvent,
    ) -> None:  # pragma: no cover - GUI runtime only
        self.disconnect_from_server()
        super().closeEvent(event)


def main() -> int:
    """Launch the lightweight remote control GUI."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("shmpipeline Control GUI")
    app.setOrganizationName("shmpipeline")
    window = ControlWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - GUI entry point
    raise SystemExit(main())
