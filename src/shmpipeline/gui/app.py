"""Qt desktop GUI for editing and running shmpipeline configurations."""

from __future__ import annotations

import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QColor,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from shmpipeline.config import PipelineConfig
from shmpipeline.errors import ConfigValidationError
from shmpipeline.graph import PipelineGraph
from shmpipeline.gui.model import (
    available_kernel_kinds,
    default_document,
    document_to_yaml,
    load_document,
    parse_inline_yaml,
    save_document,
    validate_document,
)
from shmpipeline.gui.remote import (
    RemotePipelineSession,
    ServerConnection,
)
from shmpipeline.gui.themes import apply_application_theme, resolve_theme
from shmpipeline.gui.viewers import SharedMemoryViewer, launch_viewer_process
from shmpipeline.state import PipelineState
from shmpipeline.synthetic import (
    SyntheticInputConfig,
    available_synthetic_patterns,
)


@dataclass
class SharedMemoryRecord:
    """Editable shared-memory row values."""

    name: str
    shape: str
    dtype: str
    storage: str
    gpu_device: str
    cpu_mirror: bool


class SharedMemoryDialog(QDialog):
    """Edit one shared-memory definition."""

    def __init__(
        self,
        parent: QWidget | None = None,
        initial: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Shared Memory")
        initial = dict(initial or {})

        self.name_edit = QLineEdit(initial.get("name", ""))
        self.shape_edit = QLineEdit(
            ", ".join(str(axis) for axis in initial.get("shape", []))
        )
        self.dtype_edit = QLineEdit(str(initial.get("dtype", "float32")))
        self.storage_combo = QComboBox()
        self.storage_combo.addItems(["cpu", "gpu"])
        self.storage_combo.setCurrentText(initial.get("storage", "cpu"))
        self.gpu_device_edit = QLineEdit(initial.get("gpu_device", "cuda:0"))
        self.cpu_mirror_check = QCheckBox("Enable CPU mirror")
        self.cpu_mirror_check.setChecked(
            bool(initial.get("cpu_mirror", False))
        )

        form = QFormLayout()
        form.addRow("Name", self.name_edit)
        form.addRow("Shape", self.shape_edit)
        form.addRow("Dtype", self.dtype_edit)
        form.addRow("Storage", self.storage_combo)
        form.addRow("GPU device", self.gpu_device_edit)
        form.addRow("", self.cpu_mirror_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self._update_storage_fields(self.storage_combo.currentText())
        self.storage_combo.currentTextChanged.connect(
            self._update_storage_fields
        )

    def _update_storage_fields(self, storage: str) -> None:
        gpu_enabled = storage == "gpu"
        self.gpu_device_edit.setEnabled(gpu_enabled)
        self.cpu_mirror_check.setEnabled(gpu_enabled)

    def value(self) -> dict[str, Any]:
        """Return the edited shared-memory document row."""
        shape = [
            int(part.strip())
            for part in self.shape_edit.text().split(",")
            if part.strip()
        ]
        record = {
            "name": self.name_edit.text().strip(),
            "shape": shape,
            "dtype": self.dtype_edit.text().strip(),
            "storage": self.storage_combo.currentText(),
        }
        if record["storage"] == "gpu":
            record["gpu_device"] = (
                self.gpu_device_edit.text().strip() or "cuda:0"
            )
            if self.cpu_mirror_check.isChecked():
                record["cpu_mirror"] = True
        return record


class KernelDialog(QDialog):
    """Edit one kernel definition."""

    def __init__(
        self,
        shared_names: list[str],
        parent: QWidget | None = None,
        initial: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Kernel")
        initial = dict(initial or {})

        self.name_edit = QLineEdit(initial.get("name", ""))
        self.kind_combo = QComboBox()
        self.kind_combo.setEditable(True)
        self.kind_combo.addItems(list(available_kernel_kinds()))
        self.kind_combo.setCurrentText(initial.get("kind", "cpu.copy"))
        self.input_combo = QComboBox()
        self.input_combo.setEditable(True)
        self.input_combo.addItems(shared_names)
        self.input_combo.setCurrentText(
            initial.get("input", shared_names[0] if shared_names else "")
        )
        self.output_combo = QComboBox()
        self.output_combo.setEditable(True)
        self.output_combo.addItems(shared_names)
        self.output_combo.setCurrentText(
            initial.get("output", shared_names[0] if shared_names else "")
        )
        self.auxiliary_edit = QPlainTextEdit()
        self.auxiliary_edit.setPlainText(
            ""
            if "auxiliary" not in initial
            else document_to_yaml({"auxiliary": initial.get("auxiliary")})
            .split("auxiliary:", 1)[1]
            .strip()
        )
        self.operation_edit = QLineEdit(initial.get("operation", ""))
        self.parameters_edit = QPlainTextEdit()
        self.parameters_edit.setPlainText(
            document_to_yaml(initial.get("parameters", {})).strip()
            if initial.get("parameters")
            else "{}"
        )
        self.read_timeout_edit = QLineEdit(
            str(initial.get("read_timeout", 1.0))
        )
        self.pause_sleep_edit = QLineEdit(
            str(initial.get("pause_sleep", 0.01))
        )

        form = QFormLayout()
        form.addRow("Name", self.name_edit)
        form.addRow("Kind", self.kind_combo)
        form.addRow("Input", self.input_combo)
        form.addRow("Output", self.output_combo)
        form.addRow("Auxiliary YAML", self.auxiliary_edit)
        form.addRow("Operation", self.operation_edit)
        form.addRow("Parameters YAML", self.parameters_edit)
        form.addRow("Read timeout", self.read_timeout_edit)
        form.addRow("Pause sleep", self.pause_sleep_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def value(self) -> dict[str, Any]:
        """Return the edited kernel document row."""
        auxiliary = parse_inline_yaml(
            self.auxiliary_edit.toPlainText(), fallback=[]
        )
        parameters = parse_inline_yaml(
            self.parameters_edit.toPlainText(), fallback={}
        )
        record: dict[str, Any] = {
            "name": self.name_edit.text().strip(),
            "kind": self.kind_combo.currentText().strip(),
            "input": self.input_combo.currentText().strip(),
            "output": self.output_combo.currentText().strip(),
            "auxiliary": auxiliary,
            "parameters": parameters,
            "read_timeout": float(self.read_timeout_edit.text().strip()),
            "pause_sleep": float(self.pause_sleep_edit.text().strip()),
        }
        operation = self.operation_edit.text().strip()
        if operation:
            record["operation"] = operation
        return record


class SyntheticInputDialog(QDialog):
    """Configure one synthetic input source."""

    def __init__(
        self,
        stream_name: str,
        parent: QWidget | None = None,
        initial: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Synthetic Input")
        initial = dict(initial or {})

        self._stream_name = stream_name
        self._stream_label = QLabel(stream_name)
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(list(available_synthetic_patterns()))
        self.pattern_combo.setCurrentText(initial.get("pattern", "random"))
        self.seed_edit = QLineEdit(str(initial.get("seed", 0)))
        default_rate_hz = initial.get("rate_hz", 500.0)
        self.rate_edit = QLineEdit(
            "" if default_rate_hz is None else str(default_rate_hz)
        )
        self.amplitude_edit = QLineEdit(str(initial.get("amplitude", 1.0)))
        self.offset_edit = QLineEdit(str(initial.get("offset", 0.0)))
        self.period_edit = QLineEdit(str(initial.get("period", 64.0)))
        self.constant_edit = QLineEdit(str(initial.get("constant", 0.0)))
        self.impulse_interval_edit = QLineEdit(
            str(initial.get("impulse_interval", 64))
        )

        form = QFormLayout()
        form.addRow("Target stream", self._stream_label)
        form.addRow("Pattern", self.pattern_combo)
        form.addRow("Seed", self.seed_edit)
        form.addRow("Rate Hz (clear=max)", self.rate_edit)
        form.addRow("Amplitude", self.amplitude_edit)
        form.addRow("Offset", self.offset_edit)
        form.addRow("Period", self.period_edit)
        form.addRow("Constant", self.constant_edit)
        form.addRow("Impulse interval", self.impulse_interval_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def value(self) -> SyntheticInputConfig:
        """Return the configured synthetic input spec."""
        rate_text = self.rate_edit.text().strip()
        return SyntheticInputConfig(
            stream_name=self._stream_name,
            pattern=self.pattern_combo.currentText().strip(),
            seed=int(self.seed_edit.text().strip()),
            rate_hz=None if not rate_text else float(rate_text),
            amplitude=float(self.amplitude_edit.text().strip()),
            offset=float(self.offset_edit.text().strip()),
            period=float(self.period_edit.text().strip()),
            constant=float(self.constant_edit.text().strip()),
            impulse_interval=int(self.impulse_interval_edit.text().strip()),
        )


class ConnectionDialog(QDialog):
    """Collect remote control-server connection details."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        base_url: str = "",
        token: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connect to Server")

        self.base_url_edit = QLineEdit(base_url)
        self.base_url_edit.setPlaceholderText("http://127.0.0.1:8765")
        self.token_edit = QLineEdit(token)
        self.token_edit.setEchoMode(QLineEdit.Password)
        self.token_edit.setPlaceholderText("optional bearer token")

        form = QFormLayout()
        form.addRow("Server URL", self.base_url_edit)
        form.addRow("Bearer token", self.token_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def value(self) -> tuple[str, str]:
        """Return the edited connection fields."""
        return (
            self.base_url_edit.text().strip(),
            self.token_edit.text().strip(),
        )


class MainWindow(QMainWindow):
    """Main GUI window for configuration editing and pipeline control.

    The window combines document editing, validation, runtime controls,
    synthetic-input management, and viewer launch actions for one pipeline
    document.
    """

    def __init__(self, *, theme_name: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("shmpipeline GUI")
        self.resize(1600, 980)
        self._settings = QSettings("shmpipeline", "gui")
        self._theme = resolve_theme(
            theme_name or self._settings.value("theme", "light", str)
        )
        self._document = default_document()
        self._current_path: Path | None = None
        self._manager = None
        self._manager_dirty = True
        self._managed_server_process: subprocess.Popen[str] | None = None
        self._managed_server_url: str | None = None
        self._managed_server_config_path: Path | None = None
        self._managed_server_log_path: Path | None = None
        self._managed_server_exit_reported = False
        self._viewers: list[SharedMemoryViewer] = []
        self._viewer_processes: list[Any] = []
        self._validation_state = "idle"
        self._reported_failures: set[tuple[str, str]] = set()

        self._status_label = QLabel("Server: disconnected")
        self._validation_label = QLabel("Validation: not run")

        self._shared_table = QTableWidget(0, 6)
        self._shared_table.setHorizontalHeaderLabels(
            ["Name", "Storage", "Shape", "Dtype", "GPU Device", "CPU Mirror"]
        )
        self._shared_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._shared_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._shared_table.itemDoubleClicked.connect(
            lambda *_: self.edit_shared_memory()
        )

        self._kernel_table = QTableWidget(0, 6)
        self._kernel_table.setHorizontalHeaderLabels(
            ["Name", "Kind", "Input", "Output", "Auxiliary", "Operation"]
        )
        self._kernel_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._kernel_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._kernel_table.itemDoubleClicked.connect(
            lambda *_: self.edit_kernel()
        )

        self._worker_table = QTableWidget(0, 11)
        self._worker_table.setHorizontalHeaderLabels(
            [
                "Kernel",
                "Kind",
                "PID",
                "CPU Slot",
                "Alive",
                "Exit Code",
                "Frames",
                "Avg us",
                "Jitter us RMS",
                "Hz",
                "Input -> Output",
            ]
        )
        self._worker_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._worker_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self._synthetic_table = QTableWidget(0, 6)
        self._synthetic_table.setHorizontalHeaderLabels(
            ["Stream", "Pattern", "Alive", "Frames", "Rate Hz", "Last Error"]
        )
        self._synthetic_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._synthetic_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self._yaml_preview = QPlainTextEdit()
        self._yaml_preview.setReadOnly(True)
        self._graph_preview = QPlainTextEdit()
        self._graph_preview.setReadOnly(True)
        self._runtime_output = QTextEdit()
        self._runtime_output.setReadOnly(True)
        self._validation_output = QTextEdit()
        self._validation_output.setReadOnly(True)

        self._build_ui()
        self._build_actions()
        self._apply_theme(self._theme.name, persist=False)
        self._set_validation_status("idle", "Validation: not run")
        self._refresh_all()

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(300)
        self._status_timer.timeout.connect(self._refresh_runtime_status)
        self._status_timer.start()

    def _build_ui(self) -> None:
        top_bar = QHBoxLayout()
        top_bar.addWidget(self._status_label, 1)
        top_bar.addWidget(self._validation_label, 1)

        shared_box = QGroupBox("Shared Memory")
        shared_layout = QVBoxLayout(shared_box)
        shared_layout.addWidget(self._shared_table)
        shared_button_row = QHBoxLayout()
        for label, handler in [
            ("Add", self.add_shared_memory),
            ("Edit", self.edit_shared_memory),
            ("Remove", self.remove_shared_memory),
            ("Open Viewer", self.open_viewer),
            ("Start/Reconfigure Test Input", self.start_synthetic_input),
            ("Stop Test Input", self.stop_synthetic_input),
        ]:
            button = QPushButton(label)
            button.clicked.connect(handler)
            shared_button_row.addWidget(button)
        shared_layout.addLayout(shared_button_row)

        kernel_box = QGroupBox("Kernels")
        kernel_layout = QVBoxLayout(kernel_box)
        kernel_layout.addWidget(self._kernel_table)
        kernel_button_row = QHBoxLayout()
        for label, handler in [
            ("Add", self.add_kernel),
            ("Edit", self.edit_kernel),
            ("Remove", self.remove_kernel),
        ]:
            button = QPushButton(label)
            button.clicked.connect(handler)
            kernel_button_row.addWidget(button)
        kernel_layout.addLayout(kernel_button_row)

        runtime_box = QGroupBox("Runtime")
        runtime_layout = QVBoxLayout(runtime_box)
        runtime_layout.addWidget(self._worker_table)
        runtime_layout.addWidget(self._synthetic_table)
        runtime_layout.addWidget(self._runtime_output)
        runtime_layout.addWidget(self._validation_output)

        self._editor_splitter = QSplitter(Qt.Vertical)
        self._editor_splitter.addWidget(shared_box)
        self._editor_splitter.addWidget(kernel_box)
        self._editor_splitter.setStretchFactor(0, 5)
        self._editor_splitter.setStretchFactor(1, 3)

        self._main_splitter = QSplitter(Qt.Horizontal)
        self._main_splitter.addWidget(self._editor_splitter)
        self._main_splitter.addWidget(runtime_box)
        self._main_splitter.setStretchFactor(0, 3)
        self._main_splitter.setStretchFactor(1, 5)

        yaml_box = QGroupBox("YAML Preview")
        yaml_layout = QVBoxLayout(yaml_box)
        yaml_layout.addWidget(self._yaml_preview)

        graph_box = QGroupBox("Graph Summary")
        graph_layout = QVBoxLayout(graph_box)
        graph_layout.addWidget(self._graph_preview)

        tabs = QTabWidget()
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        config_layout.addWidget(self._main_splitter)
        tabs.addTab(config_tab, "Editor")
        tabs.addTab(yaml_box, "YAML")
        tabs.addTab(graph_box, "Graph")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(top_bar)
        layout.addWidget(tabs)
        self.setCentralWidget(central)

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        server_menu = self.menuBar().addMenu("Server")
        stream_menu = self.menuBar().addMenu("Streams")
        control_menu = self.menuBar().addMenu("Pipeline")
        view_menu = self.menuBar().addMenu("View")

        actions = [
            (file_menu, "New", self.new_document),
            (file_menu, "Load...", self.load_document_from_disk),
            (file_menu, "Save", self.save_document_to_disk),
            (file_menu, "Save As...", self.save_document_as),
            (server_menu, "Launch Local Server", self.launch_local_server),
            (server_menu, "Stop Local Server", self.stop_local_server),
            (server_menu, "Connect...", self.connect_to_server),
            (server_menu, "Disconnect", self.disconnect_from_server),
            (server_menu, "Pull Config", self.pull_document_from_server),
            (server_menu, "Push Config", self.push_document_to_server),
            (control_menu, "Validate", self.validate_current_document),
            (control_menu, "Build", self.build_pipeline),
            (control_menu, "Start", self.start_pipeline),
            (control_menu, "Pause", self.pause_pipeline),
            (control_menu, "Resume", self.resume_pipeline),
            (control_menu, "Stop", self.stop_pipeline),
            (control_menu, "Shutdown", self.shutdown_pipeline),
        ]
        toolbar = self.addToolBar("Main")
        for menu, label, handler in actions:
            action = QAction(label, self)
            action.triggered.connect(handler)
            menu.addAction(action)
            toolbar.addAction(action)

        for label, handler in [
            ("Open Viewer", self.open_viewer),
            ("Start/Reconfigure Test Input", self.start_synthetic_input),
            ("Stop Test Input", self.stop_synthetic_input),
        ]:
            action = QAction(label, self)
            action.triggered.connect(handler)
            stream_menu.addAction(action)

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

    @property
    def current_theme_name(self) -> str:
        """Return the currently active GUI theme name."""
        return self._theme.name

    def _apply_theme(self, theme_name: str, *, persist: bool = True) -> None:
        app = QApplication.instance()
        if app is None:  # pragma: no cover - GUI runtime only
            return
        self._theme = apply_application_theme(app, theme_name)
        if persist:
            self._settings.setValue("theme", self._theme.name)
        self._set_validation_status(
            self._validation_state, self._validation_label.text()
        )
        self._refresh_runtime_status()
        for viewer in list(self._viewers):
            viewer.apply_theme(self._theme)

    def _set_validation_status(self, state: str, text: str) -> None:
        self._validation_state = state
        self._validation_label.setText(text)
        if state == "passed":
            color = self._theme.success
        elif state == "failed":
            color = self._theme.error
        else:
            color = self._theme.muted_text
        self._validation_label.setStyleSheet(f"color: {color};")

    def _dispose_manager(self) -> None:
        if self._manager is None:
            return
        try:
            self._manager.close()
        except Exception:
            pass
        self._manager = None
        self._reported_failures.clear()

    def _selected_row(self, table: QTableWidget) -> int | None:
        selected = table.selectionModel().selectedRows()
        if not selected:
            return None
        return selected[0].row()

    def _set_document(
        self, document: dict[str, Any], *, path: Path | None = None
    ) -> None:
        self._close_viewers()
        self._document = document
        self._current_path = path
        self._manager_dirty = True
        self._refresh_all()

    def _refresh_all(self) -> None:
        self._refresh_shared_table()
        self._refresh_kernel_table()
        self._refresh_graph_preview()
        self._refresh_runtime_status()
        self._yaml_preview.setPlainText(document_to_yaml(self._document))
        self._update_window_title()

    def _update_window_title(self) -> None:
        suffix = (
            str(self._current_path)
            if self._current_path is not None
            else "untitled"
        )
        self.setWindowTitle(f"shmpipeline GUI - {suffix}")

    def _refresh_shared_table(self) -> None:
        rows = self._document.get("shared_memory", [])
        self._shared_table.setRowCount(len(rows))
        for row, spec in enumerate(rows):
            values = [
                spec.get("name", ""),
                spec.get("storage", ""),
                str(tuple(spec.get("shape", []))),
                str(spec.get("dtype", "")),
                spec.get("gpu_device", ""),
                str(spec.get("cpu_mirror", False)),
            ]
            for column, value in enumerate(values):
                self._shared_table.setItem(
                    row, column, QTableWidgetItem(value)
                )
        self._shared_table.resizeColumnsToContents()

    def _refresh_kernel_table(self) -> None:
        rows = self._document.get("kernels", [])
        self._kernel_table.setRowCount(len(rows))
        for row, spec in enumerate(rows):
            values = [
                spec.get("name", ""),
                spec.get("kind", ""),
                spec.get("input", ""),
                spec.get("output", ""),
                repr(spec.get("auxiliary", [])),
                spec.get("operation", ""),
            ]
            for column, value in enumerate(values):
                self._kernel_table.setItem(
                    row, column, QTableWidgetItem(value)
                )
        self._kernel_table.resizeColumnsToContents()

    def _refresh_graph_preview(self) -> None:
        try:
            config = PipelineConfig.from_dict(self._document)
        except ConfigValidationError as exc:
            self._graph_preview.setPlainText(f"Graph unavailable: {exc}")
            return
        graph = PipelineGraph.from_config(config)
        self._graph_preview.setPlainText(graph.describe())

    def _refresh_runtime_status(self) -> None:
        self._handle_managed_server_exit()
        kernels = self._document.get("kernels", [])
        status = None
        if self._manager is not None:
            try:
                status = self._manager.status()
            except Exception as exc:
                self._runtime_output.setPlainText(
                    f"Runtime status failed: {exc}"
                )
        if status is not None:
            self._report_remote_failures(status)

        self._worker_table.setRowCount(len(kernels))
        for row, kernel in enumerate(kernels):
            worker = (
                (status or {})
                .get("workers", {})
                .get(kernel.get("name", ""), {})
            )
            values = [
                kernel.get("name", ""),
                kernel.get("kind", ""),
                str(worker.get("pid", "")),
                str(worker.get("cpu_slot", "")),
                str(worker.get("alive", False)),
                str(worker.get("exitcode", "")),
                str(worker.get("frames_processed", 0)),
                self._format_microseconds(worker.get("avg_exec_us")),
                self._format_microseconds(worker.get("jitter_us_rms")),
                self._format_float(worker.get("throughput_hz")),
                f"{kernel.get('input', '')} -> {kernel.get('output', '')}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 4 and value == "True":
                    item.setBackground(QColor(self._theme.success_bg))
                if column == 5 and value not in {"", "None", "0"}:
                    item.setBackground(QColor(self._theme.error_bg))
                self._worker_table.setItem(row, column, item)
        self._worker_table.resizeColumnsToContents()

        synthetic_sources = (status or {}).get("synthetic_sources", {})
        self._synthetic_table.setRowCount(len(synthetic_sources))
        for row, (stream_name, source_status) in enumerate(
            sorted(synthetic_sources.items())
        ):
            values = [
                stream_name,
                str(source_status.get("pattern", "")),
                str(source_status.get("alive", False)),
                str(source_status.get("frames_written", 0)),
                self._format_float(source_status.get("effective_rate_hz")),
                str(source_status.get("last_error") or ""),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 2 and value == "True":
                    item.setBackground(QColor(self._theme.success_bg))
                if column == 5 and value:
                    item.setBackground(QColor(self._theme.error_bg))
                self._synthetic_table.setItem(row, column, item)
        self._synthetic_table.resizeColumnsToContents()

        state = (
            status.get("state", "connection error")
            if status
            else (
                "disconnected" if self._manager is None else "connection error"
            )
        )
        dirty = "dirty" if self._manager_dirty else "synced"
        placement = (status or {}).get("placement_policy", "n/a")
        synthetic_count = len(synthetic_sources)
        summary = (status or {}).get("summary", {})
        active_workers = summary.get("active_workers", 0)
        idle_workers = summary.get("idle_workers", 0)
        waiting_workers = summary.get("waiting_workers", 0)
        failed_workers = summary.get("failed_workers", 0)
        server = (
            self._manager.connection.display_name
            if self._manager is not None
            else "disconnected"
        )
        self._status_label.setText(
            "Server: "
            f"{server} | State: {state} | Config: {dirty} | "
            f"Placement: {placement} | "
            f"Synthetic: {synthetic_count} | Active: {active_workers} | "
            f"Idle: {idle_workers} | Waiting: {waiting_workers} | "
            f"Failed: {failed_workers}"
        )

        self._runtime_output.setPlainText(
            self._format_runtime_status_text(status)
        )

    def _format_runtime_status_text(
        self, status: dict[str, Any] | None
    ) -> str:
        if status is None:
            return "Runtime: server not connected"
        lines = [f"Placement policy: {status.get('placement_policy', 'n/a')}"]
        summary = status.get("summary", {})
        if summary:
            lines.append(
                "Workers: "
                f"active={summary.get('active_workers', 0)} "
                f"idle={summary.get('idle_workers', 0)} "
                f"waiting={summary.get('waiting_workers', 0)} "
                f"paused={summary.get('paused_workers', 0)} "
                f"failed={summary.get('failed_workers', 0)}"
            )
        workers = status.get("workers", {})
        if workers:
            lines.append("Worker timing:")
            for kernel_name, worker_metrics in sorted(workers.items()):
                lines.append(
                    "- "
                    f"{kernel_name} health={worker_metrics.get('health', 'n/a')} "
                    f"idle_s={self._format_float(worker_metrics.get('idle_s'))} "
                    f"metric_age_s={self._format_float(worker_metrics.get('last_metric_age_s'))} "
                    f"avg_us={self._format_microseconds(worker_metrics.get('avg_exec_us'))} "
                    f"jitter_us_rms={self._format_microseconds(worker_metrics.get('jitter_us_rms'))} "
                    f"hz={self._format_float(worker_metrics.get('throughput_hz'))} "
                    f"window={worker_metrics.get('metrics_window', '')}"
                )
        else:
            lines.append("Workers: none")
        failures = status.get("failures", [])
        if failures:
            lines.append("Failures:")
            for failure in failures:
                lines.append(
                    f"- {failure.get('kernel')}: {failure.get('error')}"
                )
        else:
            lines.append("Failures: none")

        synthetic_sources = status.get("synthetic_sources", {})
        if synthetic_sources:
            lines.append("Synthetic inputs:")
            for stream_name, source_status in sorted(
                synthetic_sources.items()
            ):
                lines.append(
                    "- "
                    f"{stream_name} pattern={source_status.get('pattern')} "
                    f"frames={source_status.get('frames_written')} "
                    f"hz={self._format_float(source_status.get('effective_rate_hz'))}"
                )
        else:
            lines.append("Synthetic inputs: none")
        return "\n".join(lines)

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

    def _append_log_message(self, level: str, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {level}: {message}"
        self._validation_output.moveCursor(QTextCursor.End)
        if self._validation_output.toPlainText():
            self._validation_output.insertPlainText("\n")
        self._validation_output.insertPlainText(entry)
        self._validation_output.moveCursor(QTextCursor.End)
        self._validation_output.ensureCursorVisible()

    def _log_info(self, message: str) -> None:
        self._append_log_message("INFO", message)

    def _log_error(self, title: str, error: Exception | str) -> str:
        message = f"{title}: {error}"
        self._append_log_message("ERROR", message)
        return message

    def _show_error(self, title: str, error: Exception | str) -> None:
        self._log_error(title, error)
        QMessageBox.critical(self, title, str(error))

    def _show_info(self, title: str, message: str) -> None:
        self._log_info(f"{title}: {message}")
        QMessageBox.information(self, title, message)

    def _format_connection_error(
        self,
        connection: ServerConnection,
        error: Exception | str,
    ) -> str:
        if connection.is_local:
            return (
                f"Could not reach the control server at {connection.base_url}.\n\n"
                "Click Build or Start again to auto-launch a local server, "
                "use Server > Launch Local Server, or connect to another "
                "running server.\n\n"
                f"Original error: {error}"
            )
        return (
            f"Could not reach the control server at {connection.base_url}.\n\n"
            "Verify the server URL, bearer token, and remote server process.\n\n"
            f"Original error: {error}"
        )

    def _ensure_document_valid(self) -> bool:
        errors = validate_document(self._document)
        if errors:
            self._set_validation_status("failed", "Validation: failed")
            self._log_error("Validation Failed", "\n".join(errors))
            return False
        self._set_validation_status("passed", "Validation: passed")
        return True

    def _ensure_local_server_support(self) -> bool:
        try:
            from shmpipeline.control import api as _control_api  # noqa: F401
        except ImportError as exc:
            self._show_error(
                "Local Server Unavailable",
                "Launching a local control server from the GUI requires the "
                "control-plane dependencies. Install the updated GUI extra or "
                'install "shmpipeline[control]" alongside the GUI.\n\n'
                f"Original error: {exc}",
            )
            return False
        return True

    def _allocate_local_server_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _prepare_local_server_config(self) -> Path:
        if self._managed_server_config_path is None:
            with tempfile.NamedTemporaryFile(
                prefix="shmpipeline-gui-",
                suffix=".yaml",
                delete=False,
            ) as handle:
                self._managed_server_config_path = Path(handle.name)
        save_document(self._managed_server_config_path, self._document)
        return self._managed_server_config_path

    def _prepare_local_server_log(self) -> Path:
        if self._managed_server_log_path is not None:
            try:
                self._managed_server_log_path.unlink(missing_ok=True)
            except Exception:
                pass
        with tempfile.NamedTemporaryFile(
            prefix="shmpipeline-gui-",
            suffix=".log",
            delete=False,
        ) as handle:
            self._managed_server_log_path = Path(handle.name)
        return self._managed_server_log_path

    def _read_managed_server_log_tail(self, *, max_chars: int = 8000) -> str:
        if self._managed_server_log_path is None:
            return ""
        try:
            text = self._managed_server_log_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).strip()
        except Exception:
            return ""
        if len(text) <= max_chars:
            return text
        return "...\n" + text[-max_chars:]

    def _format_local_server_failure(
        self,
        summary: str,
        *,
        last_error: Exception | str | None = None,
    ) -> str:
        parts = [summary]
        if last_error is not None:
            parts.append(f"Last connection error: {last_error}")
        server_log = self._read_managed_server_log_tail()
        if server_log:
            parts.append(f"Server log:\n{server_log}")
        return "\n\n".join(parts)

    def _handle_managed_server_exit(self) -> None:
        process = self._managed_server_process
        if process is None or self._managed_server_exit_reported:
            return
        exit_code = process.poll()
        if exit_code is None:
            return
        self._managed_server_exit_reported = True
        message = self._format_local_server_failure(
            f"The GUI-launched local control server exited unexpectedly with code {exit_code}."
        )
        if (
            self._manager is not None
            and self._managed_server_url is not None
            and self._manager.connection.base_url == self._managed_server_url
        ):
            self._dispose_manager()
            self._refresh_runtime_status()
        self._show_error("Local Server Failed", message)

    def _connect_to_connection(
        self,
        connection: ServerConnection,
        *,
        show_feedback: bool = True,
    ) -> bool:
        session = None
        try:
            session = RemotePipelineSession(connection)
            session.info()
        except Exception as exc:
            if session is not None:
                session.close()
            self._show_error(
                "Connection Failed",
                self._format_connection_error(connection, exc),
            )
            return False

        self._dispose_manager()
        self._manager = session
        self._settings.setValue("server_url", connection.base_url)
        self._settings.setValue("server_token", connection.token or "")
        if show_feedback:
            self._log_info(f"Connected to {connection.display_name}.")
        self._refresh_runtime_status()
        return True

    def _start_local_server(self, *, show_feedback: bool = True) -> bool:
        if not self._ensure_local_server_support():
            return False
        if (
            self._managed_server_process is not None
            and self._managed_server_process.poll() is None
            and self._managed_server_url is not None
        ):
            return self._connect_to_connection(
                ServerConnection.from_values(self._managed_server_url),
                show_feedback=show_feedback,
            )

        config_path = self._prepare_local_server_config()
        log_path = self._prepare_local_server_log()
        port = self._allocate_local_server_port()
        base_url = f"http://127.0.0.1:{port}"
        command = [
            sys.executable,
            "-m",
            "shmpipeline.cli",
            "--log-level",
            "ERROR",
            "serve",
            str(config_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        try:
            with log_path.open("w", encoding="utf-8") as log_handle:
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        except Exception as exc:
            self._show_error("Local Server Failed", exc)
            return False

        deadline = time.monotonic() + 5.0
        last_error: Exception | None = None
        self._managed_server_exit_reported = False
        while time.monotonic() < deadline:
            if process.poll() is not None:
                self._show_error(
                    "Local Server Failed",
                    self._format_local_server_failure(
                        "The GUI-launched local control server exited before it became ready.",
                        last_error=last_error,
                    ),
                )
                return False
            session = None
            try:
                connection = ServerConnection.from_values(base_url)
                session = RemotePipelineSession(connection, timeout=0.5)
                session.info()
            except Exception as exc:
                last_error = exc
                if session is not None:
                    session.close()
                time.sleep(0.1)
                continue

            self._managed_server_process = process
            self._managed_server_url = base_url
            self._dispose_manager()
            self._manager = session
            self._settings.setValue("server_url", connection.base_url)
            self._settings.setValue("server_token", "")
            if show_feedback:
                self._log_info(
                    "Launched a local control server and connected to "
                    f"{connection.display_name}."
                )
            self._refresh_runtime_status()
            return True

        process.terminate()
        try:
            process.wait(timeout=2.0)
        except Exception:
            process.kill()
            process.wait(timeout=2.0)
        self._show_error(
            "Local Server Failed",
            self._format_local_server_failure(
                "Timed out waiting for the GUI-launched local control server to accept connections.",
                last_error=last_error,
            ),
        )
        return False

    def _stop_managed_server(self) -> None:
        process = self._managed_server_process
        self._managed_server_process = None
        self._managed_server_url = None
        self._managed_server_exit_reported = True
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except Exception:
                process.kill()
                process.wait(timeout=2.0)

    def _cleanup_managed_server_config(self) -> None:
        if self._managed_server_config_path is None:
            return
        try:
            self._managed_server_config_path.unlink(missing_ok=True)
        except Exception:
            pass
        self._managed_server_config_path = None

    def _cleanup_managed_server_log(self) -> None:
        if self._managed_server_log_path is None:
            return
        try:
            self._managed_server_log_path.unlink(missing_ok=True)
        except Exception:
            pass
        self._managed_server_log_path = None

    def _pipeline_state(self) -> PipelineState | None:
        if self._manager is None:
            return None
        try:
            status = self._manager.status()
        except Exception as exc:
            self._show_error("Server Communication Failed", exc)
            return None
        return PipelineState(status["state"])

    def _report_remote_failures(self, status: dict[str, Any]) -> None:
        active_failures: set[tuple[str, str]] = set()
        for failure in status.get("failures", []):
            kernel = str(failure.get("kernel") or "unknown")
            error = str(failure.get("error") or "unknown error")
            key = (kernel, error)
            active_failures.add(key)
            if key not in self._reported_failures:
                self._show_error("Pipeline Error", f"{kernel}: {error}")
        self._reported_failures = active_failures

    def _close_viewers(self) -> None:
        for viewer in list(self._viewers):
            viewer.close()
        self._viewers.clear()
        for process in list(self._viewer_processes):
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=2.0)
        self._viewer_processes.clear()

    def _selected_stream_spec(
        self, *, show_message: bool = False
    ) -> dict[str, Any] | None:
        row = self._selected_row(self._shared_table)
        if row is None:
            if show_message:
                self._show_info(
                    "Stream Selection",
                    "Select a shared memory row first.",
                )
            return None
        return dict(self._document["shared_memory"][row])

    def new_document(self) -> None:
        self._set_document(default_document())
        self._log_info("New empty document")

    def launch_local_server(self) -> None:
        if not self._ensure_document_valid():
            return
        self._start_local_server(show_feedback=True)

    def stop_local_server(self) -> None:
        was_connected = (
            self._manager is not None
            and self._managed_server_url is not None
            and self._manager.connection.base_url == self._managed_server_url
        )
        if was_connected:
            self._dispose_manager()
        self._stop_managed_server()
        if was_connected:
            self._refresh_runtime_status()
        self._log_info("Local control server stopped.")

    def connect_to_server(self) -> bool:
        initial_url = self._settings.value(
            "server_url", "http://127.0.0.1:8765", str
        )
        initial_token = self._settings.value("server_token", "", str)
        if self._manager is not None:
            initial_url = self._manager.connection.base_url
            initial_token = self._manager.connection.token or ""

        dialog = ConnectionDialog(
            self,
            base_url=initial_url,
            token=initial_token,
        )
        if dialog.exec() != QDialog.Accepted:
            return False

        base_url, token = dialog.value()
        return self._connect_to_connection(
            ServerConnection.from_values(base_url, token)
        )

    def disconnect_from_server(self) -> None:
        if self._manager is None:
            return
        self._close_viewers()
        server_name = self._manager.connection.display_name
        self._dispose_manager()
        self._log_info(f"Disconnected from {server_name}.")
        self._refresh_runtime_status()

    def pull_document_from_server(self) -> None:
        if self._manager is None and not self.connect_to_server():
            return
        try:
            assert self._manager is not None
            payload = self._manager.document()
        except Exception as exc:
            self._show_error("Pull Failed", exc)
            return
        self._set_document(payload["document"])
        self._manager_dirty = False
        self._log_info(
            f"Pulled config revision {payload.get('revision', '?')} from server."
        )
        self._refresh_all()

    def push_document_to_server(self) -> None:
        if not self._ensure_document_valid():
            return
        if self._manager is None and not self._start_local_server(
            show_feedback=False
        ):
            return
        if self._push_document_to_server(show_feedback=True):
            self._refresh_runtime_status()

    def load_document_from_disk(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Pipeline Config",
            str(Path.cwd()),
            "YAML Files (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            document = load_document(path)
        except Exception as exc:
            self._show_error("Load Failed", exc)
            return
        self._close_viewers()
        self._set_document(document, path=Path(path))
        self._log_info(f"Loaded {path}")

    def save_document_to_disk(self) -> None:
        if self._current_path is None:
            self.save_document_as()
            return
        try:
            save_document(self._current_path, self._document)
        except Exception as exc:
            self._show_error("Save Failed", exc)
            return
        self._log_info(f"Saved {self._current_path}")

    def save_document_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Pipeline Config",
            str(Path.cwd() / "pipeline.yaml"),
            "YAML Files (*.yaml *.yml)",
        )
        if not path:
            return
        self._current_path = Path(path)
        self.save_document_to_disk()
        self._update_window_title()

    def add_shared_memory(self) -> None:
        dialog = SharedMemoryDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            value = dialog.value()
        except Exception as exc:
            self._show_error("Invalid Shared Memory", exc)
            return
        self._document.setdefault("shared_memory", []).append(value)
        self._manager_dirty = True
        self._refresh_all()

    def edit_shared_memory(self) -> None:
        row = self._selected_row(self._shared_table)
        if row is None:
            return
        dialog = SharedMemoryDialog(self, self._document["shared_memory"][row])
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            value = dialog.value()
        except Exception as exc:
            self._show_error("Invalid Shared Memory", exc)
            return
        self._document["shared_memory"][row] = value
        self._manager_dirty = True
        self._refresh_all()

    def remove_shared_memory(self) -> None:
        row = self._selected_row(self._shared_table)
        if row is None:
            return
        del self._document["shared_memory"][row]
        self._manager_dirty = True
        self._refresh_all()

    def add_kernel(self) -> None:
        shared_names = [
            spec.get("name", "")
            for spec in self._document.get("shared_memory", [])
        ]
        dialog = KernelDialog(shared_names, self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            value = dialog.value()
        except Exception as exc:
            self._show_error("Invalid Kernel", exc)
            return
        self._document.setdefault("kernels", []).append(value)
        self._manager_dirty = True
        self._refresh_all()

    def edit_kernel(self) -> None:
        row = self._selected_row(self._kernel_table)
        if row is None:
            return
        shared_names = [
            spec.get("name", "")
            for spec in self._document.get("shared_memory", [])
        ]
        dialog = KernelDialog(
            shared_names, self, self._document["kernels"][row]
        )
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            value = dialog.value()
        except Exception as exc:
            self._show_error("Invalid Kernel", exc)
            return
        self._document["kernels"][row] = value
        self._manager_dirty = True
        self._refresh_all()

    def remove_kernel(self) -> None:
        row = self._selected_row(self._kernel_table)
        if row is None:
            return
        del self._document["kernels"][row]
        self._manager_dirty = True
        self._refresh_all()

    def validate_current_document(self) -> None:
        if not self._ensure_document_valid():
            return
        self._log_info("Configuration is valid.")

    def _push_document_to_server(self, *, show_feedback: bool) -> bool:
        if not self._ensure_document_valid():
            return False
        try:
            assert self._manager is not None
            payload = self._manager.update_document(self._document)
        except Exception as exc:
            self._show_error("Push Failed", exc)
            return False
        self._manager_dirty = False
        self._set_validation_status("passed", "Validation: passed")
        if show_feedback:
            self._log_info(
                "Pushed config to server at revision "
                f"{payload.get('revision', '?')}."
            )
        return True

    def _ensure_manager_ready(self) -> bool:
        if not self._ensure_document_valid():
            return False
        if self._manager is None and not self._start_local_server(
            show_feedback=False
        ):
            return False
        if self._manager_dirty:
            self._close_viewers()
            if not self._push_document_to_server(show_feedback=False):
                return False
        return True

    def build_pipeline(self) -> None:
        if not self._ensure_manager_ready():
            return
        try:
            assert self._manager is not None
            self._manager.build()
        except Exception as exc:
            self._show_error("Build Failed", exc)
            return
        self._log_info("Pipeline built.")
        self._refresh_all()

    def start_pipeline(self) -> None:
        if not self._ensure_manager_ready():
            return
        try:
            assert self._manager is not None
            self._manager.start()
        except Exception as exc:
            self._show_error("Start Failed", exc)
            return
        self._log_info("Pipeline started.")
        self._refresh_all()

    def pause_pipeline(self) -> None:
        if self._manager is None:
            return
        try:
            self._manager.pause()
        except Exception as exc:
            self._show_error("Pause Failed", exc)
            return
        self._log_info("Pipeline paused.")
        self._refresh_all()

    def resume_pipeline(self) -> None:
        if self._manager is None:
            return
        try:
            self._manager.resume()
        except Exception as exc:
            self._show_error("Resume Failed", exc)
            return
        self._log_info("Pipeline resumed.")
        self._refresh_all()

    def stop_pipeline(self) -> None:
        if self._manager is None:
            return
        try:
            self._manager.stop(force=True)
        except Exception as exc:
            self._show_error("Stop Failed", exc)
            return
        self._log_info("Pipeline stopped.")
        self._refresh_all()

    def shutdown_pipeline(self) -> None:
        if self._manager is None:
            return
        try:
            self._close_viewers()
            self._manager.shutdown(force=True)
        except Exception as exc:
            self._show_error("Shutdown Failed", exc)
            return
        self._log_info("Pipeline shut down.")
        self._refresh_all()

    def start_synthetic_input(self) -> None:
        state = self._pipeline_state()
        if self._manager is None or state not in {
            PipelineState.BUILT,
            PipelineState.RUNNING,
            PipelineState.PAUSED,
            PipelineState.FAILED,
        }:
            self._show_info(
                "Synthetic Input",
                "Build the pipeline before starting a synthetic input.",
            )
            return
        spec = self._selected_stream_spec(show_message=True)
        if spec is None:
            return
        stream_name = spec.get("name", "")
        initial = self._manager.synthetic_input_status().get(stream_name, {})
        dialog = SyntheticInputDialog(stream_name, self, initial=initial)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            synthetic_spec = dialog.value()
            self._manager.start_synthetic_input(synthetic_spec)
        except Exception as exc:
            self._show_error("Synthetic Input Failed", exc)
            return
        self._log_info(f"Synthetic input started for {stream_name}.")
        self._refresh_runtime_status()

    def stop_synthetic_input(self) -> None:
        if self._manager is None:
            return
        spec = self._selected_stream_spec(show_message=True)
        if spec is None:
            return
        stream_name = spec.get("name", "")
        try:
            self._manager.stop_synthetic_input(stream_name)
        except Exception as exc:
            self._show_error("Synthetic Input Failed", exc)
            return
        self._log_info(f"Synthetic input stopped for {stream_name}.")
        self._refresh_runtime_status()

    def open_viewer(self) -> None:
        state = self._pipeline_state()
        if self._manager is None or state not in {
            PipelineState.BUILT,
            PipelineState.RUNNING,
            PipelineState.PAUSED,
            PipelineState.FAILED,
        }:
            self._show_info(
                "Viewer", "Build the pipeline before opening viewers."
            )
            return
        if self._manager_dirty:
            self._show_info(
                "Viewer",
                "Push the current config to the server before opening a viewer.",
            )
            return
        if not self._manager.is_local:
            self._show_info(
                "Viewer",
                "Shared-memory viewers are only available when connected to a local server.",
            )
            return
        spec = self._selected_stream_spec(show_message=True)
        if spec is None:
            return
        try:
            process = launch_viewer_process(spec, self._theme.name)
        except Exception as exc:
            self._show_error("Viewer Failed", exc)
            return
        self._viewer_processes.append(process)

    def closeEvent(
        self, event: QCloseEvent
    ) -> None:  # pragma: no cover - GUI runtime only
        self._status_timer.stop()
        self._close_viewers()
        self._dispose_manager()
        self._stop_managed_server()
        self._cleanup_managed_server_config()
        self._cleanup_managed_server_log()
        super().closeEvent(event)


def main() -> int:
    """Launch the shmpipeline desktop GUI."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("shmpipeline GUI")
    app.setOrganizationName("shmpipeline")
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - GUI entry point
    raise SystemExit(main())
