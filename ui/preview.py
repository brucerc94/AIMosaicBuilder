"""Interactive preview window for the generated AI Mosaic Builder layout."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QDialog, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QPushButton, QToolBar, QVBoxLayout, QWidget

from engine.models import ImageRecord, ImageStatus
from vision.cropper import compute_crop_box


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

    @staticmethod
    def _best_detection(record: ImageRecord):
        return max(record.detections, key=lambda d: (d.is_main, d.confidence, d.relative_size))

    def _find_position(self, width: int, height: int) -> tuple[int, int] | None:
        if width <= 0 or height <= 0 or width > self._canvas_w or height > self._canvas_h:
            return None
        for y in range(0, self._canvas_h - height + 1, 10):
            for x in range(0, self._canvas_w - width + 1, 10):
                if not any(
                    not (
                        x + width <= ox
                        or x >= ox + ow
                        or y + height <= oy
                        or y >= oy + oh
                    )
                    for ox, oy, ow, oh in self._occupied
                ):
                    return x, y
        return None

    def _layout_records(self) -> list[tuple[ImageRecord, object, float, int, int, int, int]]:
        selected = [
            record
            for record in self._records
            if record.status == ImageStatus.SELECTED and record.detections
        ]
        selected.sort(
            key=lambda record: (
                record.selection.slot_index if record.selection else 10**9,
                record.filename.lower(),
            )
        )

        self._occupied = []
        placements = []
        for record in selected:
            crop = record.selection.crop_bbox if record.selection and record.selection.crop_bbox else None
            if crop is None:
                detection = self._best_detection(record)
                crop = compute_crop_box(
                    detection.bbox,
                    record.width,
                    record.height,
                    self._padding_px,
                )
            zoom = float(record.selection.zoom) if record.selection else 0.5
            width = max(1, int(crop.width * zoom))
            height = max(1, int(crop.height * zoom))
            position = self._find_position(width, height)
            if position is None:
                continue
            x, y = position
            self._occupied.append((x, y, width, height))
            placements.append((record, crop, zoom, width, height, x, y))
        return placements

    @staticmethod
    def _pixmap_from_crop(record: ImageRecord, crop, width: int, height: int) -> QPixmap | None:
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

        for record, crop, _zoom, width, height, x, y in self._layout_records():
            pixmap = self._pixmap_from_crop(record, crop, width, height)
            if pixmap is None or pixmap.isNull():
                continue
            item = QGraphicsPixmapItem(pixmap)
            item.setPos(x, y)
            item.setToolTip(record.filename)
            self._scene.addItem(item)

        self._fit_canvas()

    def _fit_canvas(self) -> None:
        self._view.fit_canvas()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_view"):
            self._fit_canvas()
