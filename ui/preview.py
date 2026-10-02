"""Interactive preview window for the generated AI Mosaic Builder layout."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QDialog, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QPushButton, QToolBar, QVBoxLayout, QWidget

from engine.layout_optimizer import simulate_viewer_layout
from engine.models import ImageRecord


class MosaicGraphicsView(QGraphicsView):
    """Canvas with fit-to-view, wheel zoom and hand-pan navigation."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(Qt.black)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else (1.0 / 1.15)
        self.scale(factor, factor)

    def fit_canvas(self) -> None:
        scene = self.scene()
        if scene is None or scene.sceneRect().isNull():
            return
        self.resetTransform()
        self.fitInView(scene.sceneRect(), Qt.KeepAspectRatio)


class PreviewWindow(QDialog):
    """Show the generated mosaic using the same canvas coordinate system as ImageMosaicView."""

    def __init__(
        self,
        records: list[ImageRecord],
        canvas_size: tuple[int, int],
        padding_px: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Mosaic Preview")
        self.resize(1400, 900)
        self.setModal(False)
        self._records = list(records)
        self._canvas_w, self._canvas_h = map(int, canvas_size)
        self._padding_px = int(padding_px)
        self._occupied: list[tuple[int, int, int, int]] = []
        self._build_ui()
        self._render_mosaic()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        toolbar = QToolBar()
        toolbar.setMovable(False)
        fit_button = QPushButton("Fit")
        fit_button.clicked.connect(self._fit_canvas)
        toolbar.addWidget(fit_button)
        root.addWidget(toolbar)

        self._view = MosaicGraphicsView()
        self._scene = QGraphicsScene(self)
        self._scene.setSceneRect(0, 0, self._canvas_w, self._canvas_h)
        self._scene.setBackgroundBrush(Qt.black)
        self._view.setScene(self._scene)
        root.addWidget(self._view, 1)

    def _layout_records(self):
        """Use the canonical first-fit layout shared with export and generation."""
        return simulate_viewer_layout(
            self._records,
            canvas_size=(self._canvas_w, self._canvas_h),
            padding_px=self._padding_px,
            initial_zoom=0.5,
            min_zoom=0.1,
            zoom_decay=0.9,
            min_subject_px=0,
        )

    @staticmethod
    def _pixmap_from_crop(record: ImageRecord, crop, width: int, height: int) -> QPixmap | None:
        """Load one source image, crop it, resize it, and convert it to a Qt pixmap."""
        try:
            from PIL import Image

            with Image.open(record.path) as source:
                source = source.convert("RGB")
                cropped = source.crop((crop.x, crop.y, crop.x2, crop.y2))
                cropped = cropped.resize((width, height), Image.Resampling.LANCZOS)
                raw = cropped.tobytes("raw", "RGB")
                image = QImage(
                    raw,
                    width,
                    height,
                    width * 3,
                    QImage.Format_RGB888,
                ).copy()
            return QPixmap.fromImage(image)
        except Exception:
            return None

    def _render_mosaic(self) -> None:
        self._scene.clear()
        self._scene.setSceneRect(0, 0, self._canvas_w, self._canvas_h)

        for placement in self._layout_records():
            pixmap = self._pixmap_from_crop(
                placement.record,
                placement.crop_bbox,
                placement.width,
                placement.height,
            )
            if pixmap is None or pixmap.isNull():
                continue
            item = QGraphicsPixmapItem(pixmap)
            item.setPos(placement.x, placement.y)
            item.setToolTip(placement.record.filename)
            self._scene.addItem(item)

        self._fit_canvas()

    def _fit_canvas(self) -> None:
        self._view.fit_canvas()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_view"):
            self._fit_canvas()
