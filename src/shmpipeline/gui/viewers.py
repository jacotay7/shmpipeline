"""Live shared-memory viewers used by the desktop GUI."""

from __future__ import annotations

import multiprocessing as mp
import time
from collections import deque
from typing import Any

import numpy as np
import pyqtgraph as pg
import pyshmem
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QMainWindow,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from shmpipeline.gui.model import to_numpy
from shmpipeline.gui.themes import ThemeDefinition, resolve_theme

VIEWER_METRICS_WINDOW = 300


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
    viewer = SharedMemoryViewer(spec)
    viewer.apply_theme(resolve_theme(theme_name))
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
        self._stream = self._open_stream()
        self._slice_index = 0
        self._last_count = -1
        self._last_refresh_at: float | None = None
        self._cached_array: np.ndarray | None = None
        self._refresh_intervals_s: deque[float] = deque(
            maxlen=VIEWER_METRICS_WINDOW
        )
        self._stream_samples: deque[tuple[int, float]] = deque(
            maxlen=VIEWER_METRICS_WINDOW
        )

        self._status_label = QLabel("Waiting for samples")
        self._slice_combo = QComboBox()
        self._slice_combo.hide()
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
        self.apply_theme(resolve_theme("light"))
        self.refresh()

    def apply_theme(self, theme: ThemeDefinition) -> None:
        """Apply a GUI theme to plot and image surfaces."""
        self._plot_widget.setBackground(theme.plot_background)
        self._plot_curve.setPen(pg.mkPen(color=theme.accent, width=2))
        self._image_widget.setBackground(theme.plot_background)

    def _open_stream(self):
        if self.spec.get("storage") == "gpu":
            if self.spec.get("cpu_mirror", False):
                return pyshmem.open(self.spec["name"])
            gpu_device = self.spec.get("gpu_device")
            if not gpu_device:
                raise RuntimeError(
                    "GPU viewers without cpu_mirror require a gpu_device so "
                    "they can attach directly to the CUDA handle"
                )
            return pyshmem.open(self.spec["name"], gpu_device=gpu_device)
        return pyshmem.open(self.spec["name"])

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
            return

        self._record_stream_sample(count, write_time)
        avg_stream_hz, p99_stream_hz = _compute_stream_rate_metrics(
            self._stream_samples
        )

        if count == self._last_count and self._cached_array is not None:
            self._update_status(
                count,
                self._cached_array,
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
            return

        self._cached_array = array
        self._last_count = count
        self._update_status(
            count,
            array,
            avg_stream_hz=avg_stream_hz,
            p99_stream_hz=p99_stream_hz,
            avg_view_hz=avg_view_hz,
        )
        self._render_array(array)

    def _update_status(
        self,
        count: int,
        array: np.ndarray,
        *,
        avg_stream_hz: float,
        p99_stream_hz: float,
        avg_view_hz: float,
    ) -> None:
        if self.spec.get("storage") == "gpu":
            mode = (
                "passive-cpu-mirror"
                if self.spec.get("cpu_mirror", False)
                else "passive-gpu"
            )
        else:
            mode = "passive-cpu"
        self._status_label.setText(
            "Count: "
            f"{count} | Shape: {tuple(array.shape)} | Dtype: {array.dtype} "
            f"| Mode: {mode} | Stream Hz avg: {avg_stream_hz:.1f} "
            f"| Stream Hz p99: {p99_stream_hz:.1f} "
            f"| Viewer Hz avg: {avg_view_hz:.1f}"
        )

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
            self._image_item.setImage(
                np.ascontiguousarray(array.astype(np.float32, copy=False)),
                autoLevels=True,
            )
            self._image_plot.autoRange()
            self._image_widget.show()
            return

        if array.ndim == 3 and array.shape[-1] in (3, 4):
            self._image_item.setImage(
                np.ascontiguousarray(array), autoLevels=True
            )
            self._image_plot.autoRange()
            self._image_widget.show()
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
            self._slice_combo.show()
            sliced = np.take(array, current_index, axis=0)
            if sliced.ndim == 1:
                self._plot_curve.setData(sliced.astype(np.float64, copy=False))
                self._plot_widget.show()
                return
            if sliced.ndim == 2 or (
                sliced.ndim == 3 and sliced.shape[-1] in (3, 4)
            ):
                self._image_item.setImage(
                    np.ascontiguousarray(sliced), autoLevels=True
                )
                self._image_plot.autoRange()
                self._image_widget.show()
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

    def closeEvent(
        self, event: QCloseEvent
    ) -> None:  # pragma: no cover - GUI runtime only
        self._timer.stop()
        try:
            self._stream.close()
        except Exception:
            pass
        super().closeEvent(event)
