"""
Image detail panel for AI Mosaic Builder.
Right-side panel showing:
  - Full preview of selected image
  - Bounding box overlay
  - Crop preview
  - Analysis scores
  - Person detection info
  - Manual override controls
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal, QRect, QPoint
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from engine.models import BoundingBox, ImageRecord, ImageStatus, PersonDetection

logger = logging.getLogger("ui.image_detail")

_PREVIEW_MAX_SIZE = (420, 300)
_CROP_PREVIEW_SIZE = (200, 200)


class BboxOverlayLabel(QLabel):
    """
    QLabel that draws the image with a bounding box overlay.
    """

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._base_pixmap: Optional[QPixmap] = None
        self._bbox: Optional[BoundingBox] = None
        self._img_w = 1
        self._img_h = 1
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #181825;")

    def set_image(self, pixmap: QPixmap, img_w: int, img_h: int) -> None:
        self._base_pixmap = pixmap
        self._img_w = img_w
        self._img_h = img_h
        self._redraw()

    def set_bbox(self, bbox: Optional[BoundingBox]) -> None:
        self._bbox = bbox
        self._redraw()

    def _redraw(self) -> None:
        if self._base_pixmap is None:
            self.setText("No image")
            return

        px = self._base_pixmap.copy()
        display_w = self.width() or _PREVIEW_MAX_SIZE[0]
        display_h = self.height() or _PREVIEW_MAX_SIZE[1]
        scaled = px.scaled(
            display_w, display_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        if self._bbox and self._bbox.is_valid() and self._img_w > 0 and self._img_h > 0:
            sx = scaled.width() / self._img_w
            sy = scaled.height() / self._img_h
            rx = int(self._bbox.x * sx)
            ry = int(self._bbox.y * sy)
            rw = int(self._bbox.width * sx)
            rh = int(self._bbox.height * sy)

            cx = (display_w - scaled.width()) // 2
            cy = (display_h - scaled.height()) // 2

            combined = QPixmap(display_w, display_h)
            combined.fill(QColor("#181825"))
            painter = QPainter(combined)
            painter.drawPixmap(cx, cy, scaled)
            pen = QPen(QColor("#cba6f7"), 3)
            painter.setPen(pen)
            painter.drawRect(cx + rx, cy + ry, rw, rh)
            painter.end()
            self.setPixmap(combined)
        else:
            self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._redraw()


class ImageDetailPanel(QWidget):
    """
    Right-side detail panel for a selected ImageRecord.
    Emits include_toggled / exclude_toggled when the user overrides AI selection.
    """

    include_toggled = Signal(object)
    exclude_toggled = Signal(object)
    padding_changed = Signal(object, int)

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._current_record: Optional[ImageRecord] = None
        self._build_ui()

    def _build_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        preview_grp = QGroupBox("Preview")
        preview_v = QVBoxLayout(preview_grp)

        self._preview_label = BboxOverlayLabel()
        self._preview_label.setFixedHeight(_PREVIEW_MAX_SIZE[1])
        self._preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        preview_v.addWidget(self._preview_label)
        layout.addWidget(preview_grp)

        scores_grp = QGroupBox("Scores")
        scores_form = QVBoxLayout(scores_grp)

        def score_row(label: str) -> tuple[QLabel, QLabel]:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #7f849c; font-size: 11px;")
            lbl.setMinimumWidth(120)
            val = QLabel("—")
            val.setStyleSheet("color: #cdd6f4; font-size: 11px; font-weight: bold;")
            row.addWidget(lbl)
            row.addWidget(val)
            row.addStretch()
            scores_form.addLayout(row)
            return lbl, val

        _, self._lbl_score = score_row("Final Score:")
        _, self._lbl_rank = score_row("Rank:")
        _, self._lbl_quality = score_row("Image Quality:")
        _, self._lbl_person_vis = score_row("Person Visibility:")
        _, self._lbl_composition = score_row("Composition:")
        _, self._lbl_blur = score_row("Blur:")
        _, self._lbl_person_cnt = score_row("Persons:")
        _, self._lbl_face = score_row("Face Visible:")
        _, self._lbl_body = score_row("Body Visible:")
        _, self._lbl_mosaic = score_row("Mosaic Value:")
        _, self._lbl_status = score_row("Status:")
        layout.addWidget(scores_grp)

        crop_grp = QGroupBox("Crop Preview")
        crop_v = QVBoxLayout(crop_grp)

        self._crop_label = QLabel()
        self._crop_label.setFixedSize(*_CROP_PREVIEW_SIZE)
        self._crop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._crop_label.setStyleSheet("background-color: #181825; border-radius: 4px;")
        self._crop_label.setText("Select an image")
        crop_v.addWidget(self._crop_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        pad_row = QHBoxLayout()
        pad_row.addWidget(QLabel("Padding:"))
        self._padding_spin = QSpinBox()
        self._padding_spin.setRange(0, 500)
        self._padding_spin.setSuffix(" px")
        self._padding_spin.setValue(40)
        self._padding_spin.valueChanged.connect(self._on_padding_changed)
        pad_row.addWidget(self._padding_spin)
        pad_row.addStretch()
        crop_v.addLayout(pad_row)
        layout.addWidget(crop_grp)

        notes_grp = QGroupBox("AI Notes")
        notes_v = QVBoxLayout(notes_grp)
        self._notes_label = QLabel("—")
        self._notes_label.setWordWrap(True)
        self._notes_label.setStyleSheet("color: #cdd6f4; font-size: 11px;")
        notes_v.addWidget(self._notes_label)
        layout.addWidget(notes_grp)

        override_grp = QGroupBox("Manual Override")
        override_v = QVBoxLayout(override_grp)

        self._btn_include = QPushButton("✓  Force Include")
        self._btn_include.setObjectName("success_btn")
        self._btn_include.clicked.connect(self._on_include)

        self._btn_exclude = QPushButton("✗  Force Exclude")
        self._btn_exclude.setObjectName("danger_btn")
        self._btn_exclude.clicked.connect(self._on_exclude)

        override_v.addWidget(self._btn_include)
        override_v.addWidget(self._btn_exclude)
        layout.addWidget(override_grp)

        layout.addStretch()
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def show_record(self, record: ImageRecord) -> None:
        self._current_record = record
        self._update_preview(record)
        self._update_scores(record)
        self._update_crop_preview(record)
        self._update_notes(record)
        self._update_buttons(record)

    def clear(self) -> None:
        self._current_record = None
        self._preview_label.setText("No image selected")
        self._preview_label.set_bbox(None)
        self._crop_label.setText("—")
        for lbl in [
            self._lbl_score,
            self._lbl_rank,
            self._lbl_quality,
            self._lbl_person_vis,
            self._lbl_composition,
            self._lbl_blur,
            self._lbl_person_cnt,
            self._lbl_face,
            self._lbl_body,
            self._lbl_mosaic,
            self._lbl_status,
        ]:
            lbl.setText("—")
        self._notes_label.setText("—")

    def _update_preview(self, record: ImageRecord) -> None:
        try:
            px = QPixmap(record.path)
            if px.isNull():
                self._preview_label.setText("Cannot load image")
                return
            self._preview_label.set_image(px, record.width, record.height)
            if record.detections:
                main = next((d for d in record.detections if d.is_main), record.detections[0])
                self._preview_label.set_bbox(main.bbox)
            else:
                self._preview_label.set_bbox(None)
        except Exception as e:
            logger.debug(f"[detail] Preview load failed: {e}")
            self._preview_label.setText("Preview error")

    def _update_scores(self, record: ImageRecord) -> None:
        a = record.analysis
        r = record.ranking

        self._lbl_score.setText(f"{r.final_score:.1f} / 100" if r else "—")
        self._lbl_rank.setText(f"#{r.rank}" if (r and r.rank > 0) else "—")
        self._lbl_quality.setText(f"{a.image_quality:.0%}" if a else "—")
        self._lbl_person_vis.setText(f"{a.person_visibility:.0%}" if a else "—")
        self._lbl_composition.setText(f"{a.composition:.0%}" if a else "—")
        self._lbl_blur.setText(f"{a.blur:.0%}" if a else "—")
        self._lbl_person_cnt.setText(str(a.person_count) if a else "—")
        self._lbl_face.setText(("Yes" if a.face_visible else "No") if a else "—")
        self._lbl_body.setText(("Yes" if a.body_visible else "No") if a else "—")
        self._lbl_mosaic.setText(f"{a.mosaic_value:.0%}" if a else "—")
        self._lbl_status.setText(record.status.value.upper())

    def _update_crop_preview(self, record: ImageRecord) -> None:
        try:
            if not record.detections:
                self._crop_label.setText("No person detected")
                return
            main = next((d for d in record.detections if d.is_main), record.detections[0])
            from vision.cropper import crop_image
            cropped = crop_image(
                record.path,
                main.bbox,
                padding_px=self._padding_spin.value(),
                output_size=_CROP_PREVIEW_SIZE,
            )
            if cropped is None:
                self._crop_label.setText("Crop error")
                return
            import io
            buf = io.BytesIO()
            cropped.save(buf, format="PNG")
            buf.seek(0)
            img = QImage()
            img.loadFromData(buf.getvalue())
            self._crop_label.setPixmap(QPixmap.fromImage(img))
        except Exception as e:
            logger.debug(f"[detail] Crop preview error: {e}")
            self._crop_label.setText("—")

    def _update_notes(self, record: ImageRecord) -> None:
        if record.analysis and record.analysis.notes:
            self._notes_label.setText(record.analysis.notes)
        elif record.error_message:
            self._notes_label.setText(f"Error: {record.error_message}")
        else:
            self._notes_label.setText("—")

    def _update_buttons(self, record: ImageRecord) -> None:
        if record.manually_included:
            self._btn_include.setText("✓  Included (click to undo)")
        else:
            self._btn_include.setText("✓  Force Include")

        if record.manually_excluded:
            self._btn_exclude.setText("✗  Excluded (click to undo)")
        else:
            self._btn_exclude.setText("✗  Force Exclude")

    def _on_include(self) -> None:
        if self._current_record:
            self.include_toggled.emit(self._current_record)

    def _on_exclude(self) -> None:
        if self._current_record:
            self.exclude_toggled.emit(self._current_record)

    def _on_padding_changed(self, value: int) -> None:
        if self._current_record:
            self._update_crop_preview(self._current_record)
            self.padding_changed.emit(self._current_record, value)
