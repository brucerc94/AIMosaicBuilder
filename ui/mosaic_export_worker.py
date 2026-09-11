"""Background worker for rendering the final mosaic image."""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QRunnable, Signal

from engine.mosaic_image_export import render_mosaic_image
from engine.models import ImageRecord

logger = logging.getLogger("mosaic_export_worker")


class MosaicExportSignals(QObject):
    progress = Signal(str)
    finished = Signal(str)
    error = Signal(str)
    cancelled = Signal()


class MosaicExportWorker(QRunnable):
    """Render the full-resolution mosaic outside the UI thread."""

    def __init__(
        self,
        records: list[ImageRecord],
        canvas_size: tuple[int, int],
        padding_px: int,
        output_path: str,
    ) -> None:
        super().__init__()
        self._records = list(records)
        self._canvas_size = canvas_size
        self._padding_px = padding_px
        self._output_path = output_path
        self._cancelled = False
        self.signals = MosaicExportSignals()

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            if self._cancelled:
                self.signals.cancelled.emit()
                return
            self.signals.progress.emit("Rendering mosaic image…")
            output = render_mosaic_image(
                self._records,
                canvas_size=self._canvas_size,
                padding_px=self._padding_px,
                output_path=self._output_path,
            )
            if self._cancelled:
                self.signals.cancelled.emit()
                return
            self.signals.finished.emit(str(output))
        except Exception as exc:
            logger.error("Mosaic image export failed: %s", exc, exc_info=True)
            self.signals.error.emit(str(exc))
