"""Qt desktop GUI for editing and running shmpipeline configurations."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
import pyshmem
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QColor, QCloseEvent
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

from shmpipeline.errors import ConfigValidationError, ShmPipelineError
from shmpipeline.gui.model import (
    available_kernel_kinds,
    create_manager,
    default_document,
    document_to_yaml,
    load_document,
    parse_inline_yaml,
    save_document,
    to_numpy,
    validate_document,
)
from shmpipeline.state import PipelineState


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

    def __init__(self, parent: QWidget | None = None, initial: dict[str, Any] | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Shared Memory")
        initial = dict(initial or {})

        self.name_edit = QLineEdit(initial.get("name", ""))
        self.shape_edit = QLineEdit(", ".join(str(axis) for axis in initial.get("shape", [])))
        self.dtype_edit = QLineEdit(str(initial.get("dtype", "float32")))
        self.storage_combo = QComboBox()
        self.storage_combo.addItems(["cpu", "gpu"])
        self.storage_combo.setCurrentText(initial.get("storage", "cpu"))
        self.gpu_device_edit = QLineEdit(initial.get("gpu_device", "cuda:0"))
        self.cpu_mirror_check = QCheckBox("Enable CPU mirror")
        self.cpu_mirror_check.setChecked(bool(initial.get("cpu_mirror", False)))

        form = QFormLayout()
        form.addRow("Name", self.name_edit)
        form.addRow("Shape", self.shape_edit)
        form.addRow("Dtype", self.dtype_edit)
        form.addRow("Storage", self.storage_combo)
        form.addRow("GPU device", self.gpu_device_edit)
        form.addRow("", self.cpu_mirror_check)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self._update_storage_fields(self.storage_combo.currentText())
        self.storage_combo.currentTextChanged.connect(self._update_storage_fields)

    def _update_storage_fields(self, storage: str) -> None:
        gpu_enabled = storage == "gpu"
        self.gpu_device_edit.setEnabled(gpu_enabled)
        self.cpu_mirror_check.setEnabled(gpu_enabled)

    def value(self) -> dict[str, Any]:
        """Return the edited shared-memory document row."""
        shape = [int(part.strip()) for part in self.shape_edit.text().split(",") if part.strip()]
        record = {
            "name": self.name_edit.text().strip(),
            "shape": shape,
            "dtype": self.dtype_edit.text().strip(),
            "storage": self.storage_combo.currentText(),
        }
        if record["storage"] == "gpu":
            record["gpu_device"] = self.gpu_device_edit.text().strip() or "cuda:0"
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
        self.input_combo.setCurrentText(initial.get("input", shared_names[0] if shared_names else ""))
        self.output_combo = QComboBox()
        self.output_combo.setEditable(True)
        self.output_combo.addItems(shared_names)
        self.output_combo.setCurrentText(initial.get("output", shared_names[0] if shared_names else ""))
        self.auxiliary_edit = QPlainTextEdit()
        self.auxiliary_edit.setPlainText(
            "" if "auxiliary" not in initial else document_to_yaml({"auxiliary": initial.get("auxiliary")}).split("auxiliary:", 1)[1].strip()
        )
        self.operation_edit = QLineEdit(initial.get("operation", ""))
        self.parameters_edit = QPlainTextEdit()
        self.parameters_edit.setPlainText(
            document_to_yaml(initial.get("parameters", {})).strip()
            if initial.get("parameters")
            else "{}"
        )
        self.read_timeout_edit = QLineEdit(str(initial.get("read_timeout", 1.0)))
        self.pause_sleep_edit = QLineEdit(str(initial.get("pause_sleep", 0.01)))

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

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def value(self) -> dict[str, Any]:
        """Return the edited kernel document row."""
        auxiliary = parse_inline_yaml(self.auxiliary_edit.toPlainText(), fallback=[])
        parameters = parse_inline_yaml(self.parameters_edit.toPlainText(), fallback={})
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


class SharedMemoryViewer(QMainWindow):
    """Passive 30 Hz viewer for one shared-memory stream."""

    def __init__(self, spec: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.spec = dict(spec)
        self.setWindowTitle(f"Viewer: {self.spec['name']}")
        self._stream = self._open_stream()
        self._last_count = -1
        self._slice_index = 0

        self._status_label = QLabel("Waiting for samples")
        self._plot_widget = pg.PlotWidget()
        self._plot_curve = self._plot_widget.plot(pen=pg.mkPen(color="#1f77b4", width=2))
        self._image_widget = pg.ImageView()
        self._text_widget = QTextEdit()
        self._text_widget.setReadOnly(True)
        self._slice_combo = QComboBox()
        self._slice_combo.currentIndexChanged.connect(self._set_slice_index)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self._status_label)
        layout.addWidget(self._slice_combo)
        layout.addWidget(self._plot_widget)
        layout.addWidget(self._image_widget)
        layout.addWidget(self._text_widget)
        self.setCentralWidget(central)

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    def _open_stream(self):
        if self.spec.get("storage") == "gpu":
            return pyshmem.open(self.spec["name"], gpu_device=self.spec.get("gpu_device"))
        return pyshmem.open(self.spec["name"])

    def _set_slice_index(self, index: int) -> None:
        self._slice_index = max(0, index)

    def refresh(self) -> None:
        """Read the latest payload and update the passive viewer."""
        try:
            payload = self._stream.read()
            count = self._stream.count
            array = np.asarray(to_numpy(payload))
        except Exception as exc:  # pragma: no cover - GUI runtime only
            self._status_label.setText(f"Viewer read failed: {exc}")
            return

        self._status_label.setText(
            f"Count: {count} | Shape: {tuple(array.shape)} | Dtype: {array.dtype}"
        )
        self._last_count = count
        self._render_array(array)

    def _render_array(self, array: np.ndarray) -> None:
        self._plot_widget.hide()
        self._image_widget.hide()
        self._text_widget.hide()
        self._slice_combo.hide()

        if array.ndim == 0:
            self._text_widget.setPlainText(repr(array.item()))
            self._text_widget.show()
            return
        if array.ndim == 1:
            self._plot_curve.setData(array.astype(np.float64, copy=False))
            self._plot_widget.show()
            return
        if array.ndim == 2:
            self._image_widget.setImage(array.astype(np.float32, copy=False), autoLevels=True)
            self._image_widget.show()
            return
        if array.ndim == 3 and array.shape[-1] <= 4:
            self._slice_combo.clear()
            self._slice_combo.addItems([str(index) for index in range(array.shape[-1])])
            self._slice_combo.setCurrentIndex(min(self._slice_index, array.shape[-1] - 1))
            self._slice_combo.show()
            self._image_widget.setImage(
                array[..., self._slice_combo.currentIndex()].astype(np.float32, copy=False),
                autoLevels=True,
            )
            self._image_widget.show()
            return
        self._text_widget.setPlainText(
            f"ndim={array.ndim}\nshape={tuple(array.shape)}\ndtype={array.dtype}\n\n"
            f"Preview:\n{np.array2string(array.reshape(-1)[: min(array.size, 256)], precision=4)}"
        )
        self._text_widget.show()

    def closeEvent(self, event: QCloseEvent) -> None:  # pragma: no cover - GUI runtime only
        self._timer.stop()
        try:
            self._stream.close()
        except Exception:
            pass
        super().closeEvent(event)


class MainWindow(QMainWindow):
    """Main GUI window for configuration editing and pipeline control."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("shmpipeline GUI")
        self.resize(1600, 980)
        self._document = default_document()
        self._current_path: Path | None = None
        self._manager = None
        self._manager_dirty = True
        self._viewers: list[SharedMemoryViewer] = []

        self._status_label = QLabel("No config loaded")
        self._validation_label = QLabel("Validation: not run")
        self._validation_label.setStyleSheet("color: #666666;")

        self._shared_table = QTableWidget(0, 6)
        self._shared_table.setHorizontalHeaderLabels(
            ["Name", "Storage", "Shape", "Dtype", "GPU Device", "CPU Mirror"]
        )
        self._shared_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._shared_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._shared_table.itemDoubleClicked.connect(lambda *_: self.edit_shared_memory())

        self._kernel_table = QTableWidget(0, 6)
        self._kernel_table.setHorizontalHeaderLabels(
            ["Name", "Kind", "Input", "Output", "Auxiliary", "Operation"]
        )
        self._kernel_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._kernel_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._kernel_table.itemDoubleClicked.connect(lambda *_: self.edit_kernel())

        self._worker_table = QTableWidget(0, 6)
        self._worker_table.setHorizontalHeaderLabels(
            ["Kernel", "Kind", "PID", "Alive", "Exit Code", "Input -> Output"]
        )
        self._worker_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._worker_table.setEditTriggers(QTableWidget.NoEditTriggers)

        self._yaml_preview = QPlainTextEdit()
        self._yaml_preview.setReadOnly(True)
        self._validation_output = QTextEdit()
        self._validation_output.setReadOnly(True)

        self._build_ui()
        self._build_actions()
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
        runtime_layout.addWidget(self._validation_output)

        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.addWidget(shared_box)
        top_splitter.addWidget(kernel_box)
        top_splitter.addWidget(runtime_box)
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 4)
        top_splitter.setStretchFactor(2, 4)

        yaml_box = QGroupBox("YAML Preview")
        yaml_layout = QVBoxLayout(yaml_box)
        yaml_layout.addWidget(self._yaml_preview)

        tabs = QTabWidget()
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        config_layout.addWidget(top_splitter)
        tabs.addTab(config_tab, "Editor")
        tabs.addTab(yaml_box, "YAML")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(top_bar)
        layout.addWidget(tabs)
        self.setCentralWidget(central)

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        control_menu = self.menuBar().addMenu("Pipeline")

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

    def _selected_row(self, table: QTableWidget) -> int | None:
        selected = table.selectionModel().selectedRows()
        if not selected:
            return None
        return selected[0].row()

    def _set_document(self, document: dict[str, Any], *, path: Path | None = None) -> None:
        self._document = document
        self._current_path = path
        self._manager_dirty = True
        self._refresh_all()

    def _refresh_all(self) -> None:
        self._refresh_shared_table()
        self._refresh_kernel_table()
        self._refresh_runtime_status()
        self._yaml_preview.setPlainText(document_to_yaml(self._document))
        self._update_window_title()

    def _update_window_title(self) -> None:
        suffix = str(self._current_path) if self._current_path is not None else "untitled"
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
                self._shared_table.setItem(row, column, QTableWidgetItem(value))
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
                self._kernel_table.setItem(row, column, QTableWidgetItem(value))
        self._kernel_table.resizeColumnsToContents()

    def _refresh_runtime_status(self) -> None:
        kernels = self._document.get("kernels", [])
        status = None
        if self._manager is not None:
            try:
                status = self._manager.status()
            except Exception as exc:
                self._validation_output.setPlainText(f"Runtime status failed: {exc}")

        self._worker_table.setRowCount(len(kernels))
        for row, kernel in enumerate(kernels):
            worker = (status or {}).get("workers", {}).get(kernel.get("name", ""), {})
            values = [
                kernel.get("name", ""),
                kernel.get("kind", ""),
                str(worker.get("pid", "")),
                str(worker.get("alive", False)),
                str(worker.get("exitcode", "")),
                f"{kernel.get('input', '')} -> {kernel.get('output', '')}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 3 and value == "True":
                    item.setBackground(QColor("#d8f5d0"))
                self._worker_table.setItem(row, column, item)
        self._worker_table.resizeColumnsToContents()

        state = self._manager.state.value if self._manager is not None else "no manager"
        dirty = "dirty" if self._manager_dirty else "synced"
        self._status_label.setText(f"State: {state} | Config: {dirty}")

        if status is not None and status.get("failures"):
            failure_lines = [
                f"{failure.get('kernel')}: {failure.get('error')}"
                for failure in status["failures"]
            ]
            self._validation_output.setPlainText("Failures:\n" + "\n".join(failure_lines))

    def _show_error(self, title: str, error: Exception | str) -> None:
        QMessageBox.critical(self, title, str(error))

    def _show_info(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)

    def _close_viewers(self) -> None:
        for viewer in list(self._viewers):
            viewer.close()
        self._viewers.clear()

    def new_document(self) -> None:
        self._close_viewers()
        self._document = default_document()
        self._current_path = None
        self._manager_dirty = True
        self._refresh_all()
        self._validation_output.setPlainText("New empty document")

    def load_document_from_disk(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Pipeline Config", str(Path.cwd()), "YAML Files (*.yaml *.yml)")
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
        path, _ = QFileDialog.getSaveFileName(self, "Save Pipeline Config", str(Path.cwd() / "pipeline.yaml"), "YAML Files (*.yaml *.yml)")
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
        shared_names = [spec.get("name", "") for spec in self._document.get("shared_memory", [])]
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
        shared_names = [spec.get("name", "") for spec in self._document.get("shared_memory", [])]
        dialog = KernelDialog(shared_names, self, self._document["kernels"][row])
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
            self._validation_label.setText("Validation: failed")
            self._validation_label.setStyleSheet("color: #a00000;")
            self._validation_output.setPlainText("\n".join(errors))
            return
        self._validation_label.setText("Validation: passed")
        self._validation_label.setStyleSheet("color: #006400;")
        self._validation_output.setPlainText("Configuration is valid.")

    def _ensure_manager_ready(self) -> bool:
        errors = validate_document(self._document)
        if errors:
            self._validation_label.setText("Validation: failed")
            self._validation_label.setStyleSheet("color: #a00000;")
            self._validation_output.setPlainText("\n".join(errors))
            return False
        if self._manager is None or self._manager_dirty:
            if self._manager is not None:
                try:
                    self._close_viewers()
                    self._manager.shutdown(force=True)
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
                self._show_info("Build", f"Build is not allowed while state is {self._manager.state.value!r}.")
                return
        except Exception as exc:
            self._show_error("Build Failed", exc)
            return
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
        self._refresh_all()

    def pause_pipeline(self) -> None:
        if self._manager is None:
            return
        try:
            self._manager.pause()
        except Exception as exc:
            self._show_error("Pause Failed", exc)
            return
        self._refresh_all()

    def resume_pipeline(self) -> None:
        if self._manager is None:
            return
        try:
            self._manager.resume()
        except Exception as exc:
            self._show_error("Resume Failed", exc)
            return
        self._refresh_all()

    def stop_pipeline(self) -> None:
        if self._manager is None:
            return
        try:
            self._manager.stop(force=True)
        except Exception as exc:
            self._show_error("Stop Failed", exc)
            return
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
        self._refresh_all()

    def open_viewer(self) -> None:
        if self._manager is None or self._manager.state not in {
            PipelineState.BUILT,
            PipelineState.RUNNING,
            PipelineState.PAUSED,
            PipelineState.FAILED,
        }:
            self._show_info("Viewer", "Build the pipeline before opening viewers.")
            return
        row = self._selected_row(self._shared_table)
        if row is None:
            return
        spec = dict(self._document["shared_memory"][row])
        viewer = SharedMemoryViewer(spec, self)
        viewer.show()
        self._viewers.append(viewer)

    def closeEvent(self, event: QCloseEvent) -> None:  # pragma: no cover - GUI runtime only
        self._close_viewers()
        if self._manager is not None:
            try:
                self._manager.shutdown(force=True)
            except Exception:
                pass
        super().closeEvent(event)


def main() -> int:
    """Launch the shmpipeline desktop GUI."""
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("shmpipeline GUI")
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover - GUI entry point
    raise SystemExit(main())