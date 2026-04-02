"""Live shared-memory viewers used by the desktop GUI."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pyqtgraph as pg
import pyshmem
from PySide6.QtCore import QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QMainWindow,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from shmpipeline.gui.model import to_numpy
from shmpipeline.gui.themes import ThemeDefinition, resolve_theme


class SharedMemoryViewer(QMainWindow):
    """Live viewer for one shared-memory stream.

    The implementation avoids `pyqtgraph.ImageView` because its default
    histogram/color-menu path pulls in optional matplotlib integration and can
    trip over unrelated import machinery in GPU-heavy environments.
    """

    def __init__(self, spec: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.spec = dict(spec)
        self.setWindowTitle(f"Viewer: {self.spec['name']}")
        self._stream = self._open_stream()
        self._slice_index = 0
        self._last_count = -1
        self._last_refresh_at: float | None = None
        self._observed_rate_hz = 0.0

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
            return pyshmem.open(
                self.spec["name"],
                gpu_device=self.spec.get("gpu_device"),
            )
        return pyshmem.open(self.spec["name"])

    def _set_slice_index(self, index: int) -> None:
        self._slice_index = max(0, index)

    def refresh(self) -> None:
        """Read the latest payload and refresh the active view."""
        try:
            payload = self._stream.read()
            count = self._stream.count
            array = np.asarray(to_numpy(payload))
        except Exception as exc:  # pragma: no cover - GUI runtime only
            self._status_label.setText(f"Viewer read failed: {exc}")
            return

        now = time.monotonic()
        if (
            self._last_count >= 0
            and count > self._last_count
            and self._last_refresh_at is not None
        ):
            delta_t = now - self._last_refresh_at
            if delta_t > 0.0:
                self._observed_rate_hz = (count - self._last_count) / delta_t
        self._last_count = count
        self._last_refresh_at = now
        self._status_label.setText(
            "Count: "
            f"{count} | Shape: {tuple(array.shape)} | Dtype: {array.dtype} "
            f"| Observed Hz: {self._observed_rate_hz:.1f}"
        )
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
