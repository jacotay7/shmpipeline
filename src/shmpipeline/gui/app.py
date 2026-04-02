"""Qt desktop GUI for editing and running shmpipeline configurations."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QColor
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
    create_manager,
    default_document,
    document_to_yaml,
    load_document,
    parse_inline_yaml,
    save_document,
    validate_document,
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


class MainWindow(QMainWindow):
    """Main GUI window for configuration editing and pipeline control."""

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
        self._viewers: list[SharedMemoryViewer] = []
        self._viewer_processes: list[Any] = []
        self._validation_state = "idle"

        self._status_label = QLabel("No config loaded")
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
        stream_menu = self.menuBar().addMenu("Streams")
        control_menu = self.menuBar().addMenu("Pipeline")
        view_menu = self.menuBar().addMenu("View")

        actions = [
            (file_menu, "New", self.new_document),
            (file_menu, "Load...", self.load_document_from_disk),
            (file_menu, "Save", self.save_document_to_disk),
            (file_menu, "Save As...", self.save_document_as),
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
            self._manager.shutdown(force=True)
        except Exception:
            pass
        self._manager = None

    def _selected_row(self, table: QTableWidget) -> int | None:
        selected = table.selectionModel().selectedRows()
        if not selected:
            return None
        return selected[0].row()

    def _set_document(
        self, document: dict[str, Any], *, path: Path | None = None
    ) -> None:
        self._close_viewers()
        self._dispose_manager()
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
        kernels = self._document.get("kernels", [])
        status = None
        if self._manager is not None:
            try:
                status = self._manager.status()
            except Exception as exc:
                self._runtime_output.setPlainText(
                    f"Runtime status failed: {exc}"
                )

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
            self._manager.state.value
            if self._manager is not None
            else "no manager"
        )
        dirty = "dirty" if self._manager_dirty else "synced"
        placement = (status or {}).get("placement_policy", "n/a")
        synthetic_count = len(synthetic_sources)
        self._status_label.setText(
            "State: "
            f"{state} | Config: {dirty} | Placement: {placement} | "
            f"Synthetic: {synthetic_count}"
        )

        self._runtime_output.setPlainText(
            self._format_runtime_status_text(status)
        )

    def _format_runtime_status_text(
        self, status: dict[str, Any] | None
    ) -> str:
        if status is None:
            return "Runtime: manager not built"
        lines = [f"Placement policy: {status.get('placement_policy', 'n/a')}"]
        metrics = status.get("metrics", {})
        if metrics:
            lines.append("Worker timing:")
            for kernel_name, worker_metrics in sorted(metrics.items()):
                lines.append(
                    "- "
                    f"{kernel_name} avg_us={self._format_microseconds(worker_metrics.get('avg_exec_us'))} "
                    f"jitter_us_rms={self._format_microseconds(worker_metrics.get('jitter_us_rms'))} "
                    f"hz={self._format_float(worker_metrics.get('throughput_hz'))} "
                    f"window={worker_metrics.get('metrics_window', '')}"
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

    def _show_error(self, title: str, error: Exception | str) -> None:
        QMessageBox.critical(self, title, str(error))

    def _show_info(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)

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
        self._validation_output.setPlainText("New empty document")

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
        self._validation_output.setPlainText(f"Loaded {path}")

    def save_document_to_disk(self) -> None:
        if self._current_path is None:
            self.save_document_as()
            return
        try:
            save_document(self._current_path, self._document)
        except Exception as exc:
            self._show_error("Save Failed", exc)
            return
        self._validation_output.setPlainText(f"Saved {self._current_path}")

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
        errors = validate_document(self._document)
        if errors:
            self._set_validation_status("failed", "Validation: failed")
            self._validation_output.setPlainText("\n".join(errors))
            return
        self._set_validation_status("passed", "Validation: passed")
        self._validation_output.setPlainText("Configuration is valid.")

    def _ensure_manager_ready(self) -> bool:
        errors = validate_document(self._document)
        if errors:
            self._set_validation_status("failed", "Validation: failed")
            self._validation_output.setPlainText("\n".join(errors))
            return False
        if self._manager is None or self._manager_dirty:
            if self._manager is not None:
                try:
                    self._close_viewers()
                    self._dispose_manager()
                except Exception:
                    pass
            try:
                self._manager = create_manager(self._document)
            except Exception as exc:
                self._show_error("Manager Creation Failed", exc)
                return False
            self._manager_dirty = False
        return True

    def build_pipeline(self) -> None:
        if not self._ensure_manager_ready():
            return
        try:
            assert self._manager is not None
            if self._manager.state == PipelineState.INITIALIZED:
                self._manager.build()
            elif self._manager.state == PipelineState.STOPPED:
                self._manager = create_manager(self._document)
                self._manager_dirty = False
                self._manager.build()
            else:
                self._show_info(
                    "Build",
                    "Build is not allowed while state is "
                    f"{self._manager.state.value!r}.",
                )
                return
        except Exception as exc:
            self._show_error("Build Failed", exc)
            return
        self._validation_output.setPlainText("Pipeline built.")
        self._refresh_all()

    def start_pipeline(self) -> None:
        if not self._ensure_manager_ready():
            return
        try:
            assert self._manager is not None
            if self._manager.state == PipelineState.INITIALIZED:
                self._manager.build()
            self._manager.start()
        except Exception as exc:
            self._show_error("Start Failed", exc)
            return
        self._validation_output.setPlainText("Pipeline started.")
        self._refresh_all()

    def pause_pipeline(self) -> None:
        if self._manager is None:
            return
        try:
            self._manager.pause()
        except Exception as exc:
            self._show_error("Pause Failed", exc)
            return
        self._validation_output.setPlainText("Pipeline paused.")
        self._refresh_all()

    def resume_pipeline(self) -> None:
        if self._manager is None:
            return
        try:
            self._manager.resume()
        except Exception as exc:
            self._show_error("Resume Failed", exc)
            return
        self._validation_output.setPlainText("Pipeline resumed.")
        self._refresh_all()

    def stop_pipeline(self) -> None:
        if self._manager is None:
            return
        try:
            self._manager.stop(force=True)
        except Exception as exc:
            self._show_error("Stop Failed", exc)
            return
        self._validation_output.setPlainText("Pipeline stopped.")
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
        self._validation_output.setPlainText("Pipeline shut down.")
        self._refresh_all()

    def start_synthetic_input(self) -> None:
        if self._manager is None or self._manager.state not in {
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
        self._validation_output.setPlainText(
            f"Synthetic input started for {stream_name}."
        )
        self._refresh_runtime_status()

    def stop_synthetic_input(self) -> None:
        if self._manager is None:
            return
        spec = self._selected_stream_spec(show_message=True)
        if spec is None:
            return
        stream_name = spec.get("name", "")
        self._manager.stop_synthetic_input(stream_name)
        self._validation_output.setPlainText(
            f"Synthetic input stopped for {stream_name}."
        )
        self._refresh_runtime_status()

    def open_viewer(self) -> None:
        if self._manager is None or self._manager.state not in {
            PipelineState.BUILT,
            PipelineState.RUNNING,
            PipelineState.PAUSED,
            PipelineState.FAILED,
        }:
            self._show_info(
                "Viewer", "Build the pipeline before opening viewers."
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
        self._close_viewers()
        self._dispose_manager()
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
