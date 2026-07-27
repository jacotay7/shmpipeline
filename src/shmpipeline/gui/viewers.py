"""Live shared-memory viewers used by the desktop GUI."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections import deque
from typing import Any

import numpy as np
import pyqtgraph as pg
import pyshmem
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from shmpipeline.gui.model import to_numpy
from shmpipeline.gui.themes import (
    ThemeDefinition,
    apply_application_theme,
    resolve_theme,
)

VIEWER_METRICS_WINDOW = 300
STREAM_DETAIL_LABELS = {
    "name": "Stream name",
    "shape": "Shape",
    "dtype": "Data type",
    "storage": "Storage",
    "gpu_device": "GPU device",
    "cpu_mirror": "CPU mirror",
    "initial": "Initializer",
}


def _compute_view_rate_hz(refresh_intervals_s: deque[float]) -> float:
    """Return rolling average viewer refresh rate in Hz."""
    if not refresh_intervals_s:
        return 0.0
    total_interval_s = sum(refresh_intervals_s)
    if total_interval_s <= 0.0:
        return 0.0
    return len(refresh_intervals_s) / total_interval_s


def _compute_stream_rate_metrics(
    stream_samples: deque[tuple[int, float]],
) -> tuple[float, float]:
    """Return average and p99 source rates from shared-memory metadata."""
    if len(stream_samples) < 2:
        return 0.0, 0.0

    total_count = stream_samples[-1][0] - stream_samples[0][0]
    total_duration_s = stream_samples[-1][1] - stream_samples[0][1]
    avg_hz = 0.0
    if total_count > 0 and total_duration_s > 0.0:
        avg_hz = total_count / total_duration_s

    burst_rates_hz: list[float] = []
    samples = list(stream_samples)
    for (previous_count, previous_write_time), (count, write_time) in zip(
        samples,
        samples[1:],
    ):
        count_delta = count - previous_count
        time_delta_s = write_time - previous_write_time
        if count_delta > 0 and time_delta_s > 0.0:
            burst_rates_hz.append(count_delta / time_delta_s)
    if not burst_rates_hz:
        return avg_hz, avg_hz
    return avg_hz, float(np.percentile(burst_rates_hz, 99))


def run_viewer_process(spec: dict[str, Any], theme_name: str | None) -> int:
    """Launch one viewer window in its own Python process."""
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("shmpipeline Viewer")
    app.setOrganizationName("shmpipeline")
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    theme = apply_application_theme(app, theme_name)
    viewer = SharedMemoryViewer(spec)
    viewer.apply_theme(theme)
    viewer.show()
    return app.exec()


def launch_viewer_process(
    spec: dict[str, Any], theme_name: str | None
) -> mp.Process:
    """Start a detached viewer process for one shared-memory stream."""
    context = mp.get_context("spawn")
    process = context.Process(
        target=run_viewer_process,
        args=(dict(spec), theme_name),
        name=f"shmpipeline-viewer-{spec['name']}",
        daemon=True,
    )
    process.start()
    return process


def _format_detail_value(value: Any) -> str:
    """Format one shared-memory configuration value for display."""
    if value is None or value == "":
        return "Not set"
    if isinstance(value, bool):
        return "Enabled" if value else "Disabled"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


class StreamDetailsDialog(QDialog):
    """Read-only shared-memory configuration shown on demand."""

    def __init__(self, spec: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.spec = dict(spec)
        stream_name = str(self.spec.get("name", "stream"))
        self.setWindowTitle(f"Shared Memory Details: {stream_name}")
        self.setMinimumWidth(420)

        description = QLabel(
            "Read-only configuration for this viewer. Edit these settings in "
            "the main shmpipeline window, then rebuild the pipeline."
        )
        description.setWordWrap(True)

        details_group = QGroupBox("Shared-memory configuration")
        details_form = QFormLayout(details_group)
        details_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        preferred_keys = (
            "name",
            "shape",
            "dtype",
            "storage",
            "gpu_device",
            "cpu_mirror",
        )
        ordered_keys = [
            key for key in preferred_keys if key in self.spec
        ] + sorted(key for key in self.spec if key not in preferred_keys)
        self._value_labels: dict[str, QLabel] = {}
        for key in ordered_keys:
            value_label = QLabel(_format_detail_value(self.spec[key]))
            value_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            value_label.setWordWrap(True)
            self._value_labels[key] = value_label
            details_form.addRow(
                f"{STREAM_DETAIL_LABELS.get(key, key.replace('_', ' ').title())}:",
                value_label,
            )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(description)
        layout.addWidget(details_group)
        layout.addWidget(buttons)


class SharedMemoryViewer(QMainWindow):
    """Live viewer for one shared-memory stream.

    The implementation avoids `pyqtgraph.ImageView` because its default
    histogram/color-menu path pulls in optional matplotlib integration and can
    trip over unrelated import machinery in GPU-heavy environments.

    GPU streams prefer their CPU mirror when available. Otherwise the viewer
    attaches to the GPU handle directly and takes safe cloned reads.
    """

    def __init__(self, spec: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.spec = dict(spec)
        self.setWindowTitle(f"Viewer: {self.spec['name']}")
        self.resize(960, 720)
        self.setMinimumSize(480, 360)
        self._stream = self._open_stream()
        self._slice_index = 0
        self._last_count = -1
        self._last_refresh_at: float | None = None
        self._cached_array: np.ndarray | None = None
        self._image_shape: tuple[int, int] | None = None
        self._refresh_intervals_s: deque[float] = deque(
            maxlen=VIEWER_METRICS_WINDOW
        )
        self._stream_samples: deque[tuple[int, float]] = deque(
            maxlen=VIEWER_METRICS_WINDOW
        )
        self._stream_details_dialog: StreamDetailsDialog | None = None

        self._build_menus()

        self._stats_group = QGroupBox("Frame statistics")
        self._stats_group.setObjectName("frameStats")
        stats_layout = QGridLayout(self._stats_group)
        stats_layout.setContentsMargins(10, 8, 10, 8)
        stats_layout.setHorizontalSpacing(12)
        stats_layout.setVerticalSpacing(3)
        self._stat_values: dict[str, QLabel] = {}
        stats = (
            ("Frame count", "count"),
            ("Stream average", "stream_avg"),
            ("Stream p99", "stream_p99"),
            ("Viewer refresh", "viewer_avg"),
        )
        for index, (caption, key) in enumerate(stats):
            row, pair_column = divmod(index, 2)
            caption_label = QLabel(caption)
            caption_label.setProperty("statRole", "caption")
            value_label = QLabel("—")
            value_label.setProperty("statRole", "value")
            value_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            value_label.setMinimumWidth(76)
            stats_layout.addWidget(caption_label, row, pair_column * 2)
            stats_layout.addWidget(value_label, row, pair_column * 2 + 1)
            self._stat_values[key] = value_label
        stats_layout.setColumnStretch(1, 1)
        stats_layout.setColumnStretch(3, 1)

        self._status_label = QLabel("Waiting for the first frame…")
        self._status_label.setObjectName("viewerMessage")
        self._status_label.setWordWrap(True)

        self._slice_controls = QWidget()
        slice_layout = QHBoxLayout(self._slice_controls)
        slice_layout.setContentsMargins(0, 0, 0, 0)
        slice_layout.addWidget(QLabel("Displayed slice"))
        self._slice_combo = QComboBox()
        self._slice_combo.setMinimumContentsLength(12)
        slice_layout.addWidget(self._slice_combo, 1)
        self._slice_controls.hide()
        self._slice_combo.currentIndexChanged.connect(self._set_slice_index)

        self._plot_widget = pg.PlotWidget()
        self._plot_curve = self._plot_widget.plot(pen=pg.mkPen(width=2))

        self._image_widget = pg.GraphicsLayoutWidget()
        self._image_plot = self._image_widget.addPlot()
        self._image_plot.hideAxis("left")
        self._image_plot.hideAxis("bottom")
        self._image_plot.setAspectLocked(True)
        self._image_item = pg.ImageItem()
        self._image_plot.addItem(self._image_item)

        self._text_widget = QTextEdit()
        self._text_widget.setReadOnly(True)
        self._plot_widget.hide()
        self._image_widget.hide()
        self._text_widget.hide()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(self._stats_group)
        layout.addWidget(self._status_label)
        layout.addWidget(self._slice_controls)
        layout.addWidget(self._plot_widget, 1)
        layout.addWidget(self._image_widget, 1)
        layout.addWidget(self._text_widget, 1)
        self.setCentralWidget(central)

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.apply_theme(resolve_theme("light"))
        self.refresh()

    def apply_theme(self, theme: ThemeDefinition) -> None:
        """Apply a GUI theme to plot and image surfaces."""
        self._plot_widget.setBackground(theme.plot_background)
        self._plot_curve.setPen(pg.mkPen(color=theme.accent, width=2))
        self._image_widget.setBackground(theme.plot_background)
        self._stats_group.setStyleSheet(
            "QGroupBox#frameStats {"
            f" background-color: {theme.alternate_base};"
            f" border-color: {theme.border};"
            "}"
            'QGroupBox#frameStats QLabel[statRole="caption"] {'
            f" color: {theme.muted_text};"
            "}"
            'QGroupBox#frameStats QLabel[statRole="value"] {'
            f" color: {theme.text};"
            " font-weight: 600;"
            "}"
        )
        self._status_label.setStyleSheet(
            f"color: {theme.muted_text}; padding: 2px 4px;"
        )

    def _build_menus(self) -> None:
        """Create viewer actions and their window-scoped shortcuts."""
        stream_menu = self.menuBar().addMenu("&Stream")
        self._details_action = QAction(
            "Shared Memory &Details…",
            self,
        )
        self._details_action.setShortcut(QKeySequence("Ctrl+,"))
        self._details_action.setStatusTip(
            "Show the shared-memory configuration"
        )
        self._details_action.triggered.connect(self._show_stream_details)
        stream_menu.addAction(self._details_action)
        stream_menu.addSeparator()

        self._close_action = QAction("&Close Viewer", self)
        self._close_action.setShortcut(QKeySequence("Ctrl+W"))
        self._close_action.triggered.connect(self.close)
        stream_menu.addAction(self._close_action)

        view_menu = self.menuBar().addMenu("&View")
        self._stats_action = QAction("&Frame Statistics", self)
        self._stats_action.setCheckable(True)
        self._stats_action.setChecked(True)
        self._stats_action.setShortcut(QKeySequence("Ctrl+I"))
        self._stats_action.setStatusTip("Show or hide live frame statistics")
        self._stats_action.toggled.connect(self._set_stats_visible)
        view_menu.addAction(self._stats_action)

        self._fit_action = QAction("&Fit Data to Window", self)
        self._fit_action.setShortcut(QKeySequence("F"))
        self._fit_action.setStatusTip("Fit the current image or plot")
        self._fit_action.triggered.connect(self._fit_current_view)
        view_menu.addAction(self._fit_action)

    def _set_stats_visible(self, visible: bool) -> None:
        self._stats_group.setVisible(visible)

    def _show_stream_details(self) -> None:
        if self._stream_details_dialog is None:
            self._stream_details_dialog = StreamDetailsDialog(self.spec, self)
        self._stream_details_dialog.show()
        self._stream_details_dialog.raise_()
        self._stream_details_dialog.activateWindow()

    def _fit_current_view(self) -> None:
        if self._image_widget.isVisible():
            self._image_plot.autoRange()
        elif self._plot_widget.isVisible():
            self._plot_widget.autoRange()

    def _open_stream(self):
        if self.spec.get("storage") == "gpu":
            if self.spec.get("cpu_mirror", False):
                return pyshmem.open(
                    self.spec["name"],
                    gpu_device=False,
                    readonly=True,
                )
            gpu_device = self.spec.get("gpu_device")
            if not gpu_device:
                raise RuntimeError(
                    "GPU viewers without cpu_mirror require a gpu_device so "
                    "they can attach directly to the CUDA handle"
                )
            return pyshmem.open(
                self.spec["name"], gpu_device=gpu_device, readonly=True
            )
        return pyshmem.open(self.spec["name"], readonly=True)

    def _record_stream_sample(self, count: int, write_time: float) -> None:
        """Track stream metadata samples for rate reporting."""
        if count < 0 or write_time <= 0.0:
            return
        if self._stream_samples:
            previous_count, previous_write_time = self._stream_samples[-1]
            if count <= previous_count or write_time <= previous_write_time:
                return
        self._stream_samples.append((count, write_time))

    def _set_slice_index(self, index: int) -> None:
        self._slice_index = max(0, index)

    def refresh(self) -> None:
        """Read the latest payload and refresh the active view."""
        now = time.monotonic()
        if self._last_refresh_at is not None:
            self._refresh_intervals_s.append(now - self._last_refresh_at)
        self._last_refresh_at = now
        avg_view_hz = _compute_view_rate_hz(self._refresh_intervals_s)

        try:
            count = self._stream.count
            write_time = self._stream.write_time
        except Exception as exc:  # pragma: no cover - GUI runtime only
            self._status_label.setText(f"Viewer read failed: {exc}")
            self._status_label.show()
            return

        self._record_stream_sample(count, write_time)
        avg_stream_hz, p99_stream_hz = _compute_stream_rate_metrics(
            self._stream_samples
        )

        if count == self._last_count and self._cached_array is not None:
            self._update_stats(
                count,
                avg_stream_hz=avg_stream_hz,
                p99_stream_hz=p99_stream_hz,
                avg_view_hz=avg_view_hz,
            )
            return

        try:
            payload = self._stream.read(safe=True)
            array = np.asarray(to_numpy(payload))
        except Exception as exc:  # pragma: no cover - GUI runtime only
            self._status_label.setText(f"Viewer read failed: {exc}")
            self._status_label.show()
            return

        self._cached_array = array
        self._last_count = count
        self._update_stats(
            count,
            avg_stream_hz=avg_stream_hz,
            p99_stream_hz=p99_stream_hz,
            avg_view_hz=avg_view_hz,
        )
        self._render_array(array)

    def _update_stats(
        self,
        count: int,
        *,
        avg_stream_hz: float,
        p99_stream_hz: float,
        avg_view_hz: float,
    ) -> None:
        self._stat_values["count"].setText(f"{count:,}")
        self._stat_values["stream_avg"].setText(f"{avg_stream_hz:.1f} Hz")
        self._stat_values["stream_p99"].setText(f"{p99_stream_hz:.1f} Hz")
        self._stat_values["viewer_avg"].setText(f"{avg_view_hz:.1f} Hz")
        self._status_label.hide()

    def _render_array(self, array: np.ndarray) -> None:
        self._plot_widget.hide()
        self._image_widget.hide()
        self._text_widget.hide()
        self._slice_controls.hide()

        if array.ndim == 0:
            self._text_widget.setPlainText(repr(array.item()))
            self._text_widget.show()
            return

        if array.ndim == 1:
            self._plot_curve.setData(array.astype(np.float64, copy=False))
            self._plot_widget.show()
            return

        if array.ndim == 2:
            self._render_image(
                np.ascontiguousarray(array.astype(np.float32, copy=False)),
            )
            return

        if array.ndim == 3 and array.shape[-1] in (3, 4):
            self._render_image(np.ascontiguousarray(array))
            return

        if array.ndim >= 3:
            slice_count = array.shape[0]
            self._slice_combo.blockSignals(True)
            if self._slice_combo.count() != slice_count:
                self._slice_combo.clear()
                self._slice_combo.addItems(
                    [f"Slice {index}" for index in range(slice_count)]
                )
            current_index = min(self._slice_index, slice_count - 1)
            self._slice_combo.setCurrentIndex(current_index)
            self._slice_combo.blockSignals(False)
            self._slice_controls.show()
            sliced = np.take(array, current_index, axis=0)
            if sliced.ndim == 1:
                self._plot_curve.setData(sliced.astype(np.float64, copy=False))
                self._plot_widget.show()
                return
            if sliced.ndim == 2 or (
                sliced.ndim == 3 and sliced.shape[-1] in (3, 4)
            ):
                self._render_image(np.ascontiguousarray(sliced))
                return

        preview = np.array2string(
            array.reshape(-1)[: min(array.size, 256)],
            precision=4,
        )
        self._text_widget.setPlainText(
            f"ndim={array.ndim}\nshape={tuple(array.shape)}\ndtype={array.dtype}\n\n"
            f"Preview:\n{preview}"
        )
        self._text_widget.show()

    def _render_image(self, image: np.ndarray) -> None:
        """Render an image without resetting an existing zoom or pan."""
        self._image_item.setImage(image, autoLevels=True)
        image_shape = (int(image.shape[0]), int(image.shape[1]))
        if image_shape != self._image_shape:
            self._image_shape = image_shape
            self._image_plot.autoRange()
        self._image_widget.show()

    def closeEvent(
        self, event: QCloseEvent
    ) -> None:  # pragma: no cover - GUI runtime only
        self._timer.stop()
        try:
            self._stream.close()
        except Exception:
            pass
        super().closeEvent(event)
