"""
Image grid widget for AI Mosaic Builder.

Displays thumbnails in a scrollable grid. Each card shows:
  - thumbnail preview
  - filename
  - score (0-100)
  - rank
  - person detected indicator
  - selected / rejected badge
  - manual include/exclude toggle

Cards are loaded lazily (thumbnails generated in background workers).
"""

from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from engine.models import ImageRecord, ImageStatus, SortOrder
from ui.workers import ThumbnailPool

logger = logging.getLogger("ui.image_grid")

_CARD_W = 190
_CARD_H = 230
_THUMB_SIZE = (170, 150)

_BADGE_COLORS = {
    ImageStatus.SELECTED:    ("#a6e3a1", "#1e1e2e"),
    ImageStatus.REJECTED:    ("#f38ba8", "#1e1e2e"),
    ImageStatus.ANALYZING:   ("#fab387", "#1e1e2e"),
    ImageStatus.CACHED:      ("#89dceb", "#1e1e2e"),
    ImageStatus.ANALYZED:    ("#cba6f7", "#1e1e2e"),
    ImageStatus.PREFILTERED: ("#7f849c", "#cdd6f4"),
    ImageStatus.PENDING:     ("#313244", "#cdd6f4"),
    ImageStatus.ERROR:       ("#f38ba8", "#1e1e2e"),
}

_PLACEHOLDER_PIXMAP: Optional[QPixmap] = None


def _get_placeholder(w: int = _THUMB_SIZE[0], h: int = _THUMB_SIZE[1]) -> QPixmap:
    global _PLACEHOLDER_PIXMAP
    if _PLACEHOLDER_PIXMAP is None:
        img = QImage(w, h, QImage.Format.Format_RGB32)
        img.fill(QColor("#181825"))
        _PLACEHOLDER_PIXMAP = QPixmap.fromImage(img)
    return _PLACEHOLDER_PIXMAP


def _pixmap_from_bytes(data: bytes) -> Optional[QPixmap]:
    if not data:
        return None
    img = QImage()
    if img.loadFromData(data):
        return QPixmap.fromImage(img)
    return None


class ImageCard(QFrame):
    """Single image card in the grid."""

    clicked = Signal(object)
    include_toggled = Signal(object)
    exclude_toggled = Signal(object)

    def __init__(self, record: ImageRecord, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._record = record
        self._build_ui()
        self.update_from_record(record)
        self.setObjectName("image_card")
        self.setFixedSize(_CARD_W, _CARD_H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(3)

        self._thumb_label = QLabel()
        self._thumb_label.setFixedSize(*_THUMB_SIZE)
        self._thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb_label.setPixmap(_get_placeholder())
        self._thumb_label.setScaledContents(False)
        layout.addWidget(self._thumb_label)

        self._name_label = QLabel()
        self._name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._name_label.setWordWrap(False)
        self._name_label.setStyleSheet("font-size: 10px; color: #7f849c;")
        layout.addWidget(self._name_label)

        score_row = QHBoxLayout()
        score_row.setSpacing(4)
        self._score_label = QLabel("—")
        self._score_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #cba6f7;")
        self._rank_label = QLabel("")
        self._rank_label.setStyleSheet("font-size: 10px; color: #7f849c;")
        score_row.addWidget(self._score_label)
        score_row.addStretch()
        score_row.addWidget(self._rank_label)
        layout.addLayout(score_row)

        self._badge_label = QLabel("PENDING")
        self._badge_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge_label.setStyleSheet(
            "font-size: 9px; font-weight: bold; border-radius: 3px; padding: 2px 6px;"
        )
        layout.addWidget(self._badge_label)

    def set_thumbnail(self, data: bytes) -> None:
        px = _pixmap_from_bytes(data)
        if px is None:
            return
        scaled = px.scaled(
            *_THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumb_label.setPixmap(scaled)

    def update_from_record(self, record: ImageRecord) -> None:
        self._record = record

        name = record.filename
        if len(name) > 22:
            name = name[:10] + "…" + name[-10:]
        self._name_label.setText(name)
        self._name_label.setToolTip(record.filename)

        if record.ranking and record.ranking.final_score > 0:
            self._score_label.setText(f"{record.ranking.final_score:.0f}")
        else:
            self._score_label.setText("—")

        if record.ranking and record.ranking.rank > 0:
            self._rank_label.setText(f"#{record.ranking.rank}")
        else:
            self._rank_label.setText("")

        status = record.status
        colors = _BADGE_COLORS.get(status, ("#313244", "#cdd6f4"))
        badge_text = status.value.upper()
        if record.manually_included:
            badge_text = "✓ MANUAL"
            colors = ("#a6e3a1", "#1e1e2e")
        elif record.manually_excluded:
            badge_text = "✗ EXCLUDED"
            colors = ("#f38ba8", "#1e1e2e")

        self._badge_label.setText(badge_text)
        self._badge_label.setStyleSheet(
            f"font-size: 9px; font-weight: bold; border-radius: 3px; padding: 2px 6px; "
            f"background-color: {colors[0]}; color: {colors[1]};"
        )

        if status == ImageStatus.SELECTED or record.manually_included:
            border = "#a6e3a1"
        elif status == ImageStatus.REJECTED or record.manually_excluded:
            border = "#f38ba8"
        elif status == ImageStatus.ANALYZING:
            border = "#fab387"
        else:
            border = "#313244"

        self.setStyleSheet(
            f"QFrame#image_card {{ border: 2px solid {border}; border-radius: 8px; background-color: #181825; }}"
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._record)
        super().mousePressEvent(event)

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; }"
            "QMenu::item:selected { background-color: #45475a; }"
        )
        if not self._record.manually_included:
            act_include = menu.addAction("✓  Force Include")
            act_include.triggered.connect(lambda: self.include_toggled.emit(self._record))
        else:
            act_uninc = menu.addAction("Remove Force Include")
            act_uninc.triggered.connect(lambda: self.include_toggled.emit(self._record))

        if not self._record.manually_excluded:
            act_exclude = menu.addAction("✗  Force Exclude")
            act_exclude.triggered.connect(lambda: self.exclude_toggled.emit(self._record))
        else:
            act_unexc = menu.addAction("Remove Force Exclude")
            act_unexc.triggered.connect(lambda: self.exclude_toggled.emit(self._record))

        menu.exec(self.mapToGlobal(pos))

    @property
    def record(self) -> ImageRecord:
        return self._record


class ImageGrid(QScrollArea):
    """Scrollable grid of ImageCards with lazy thumbnail loading."""

    image_selected = Signal(object)
    include_toggled = Signal(object)
    exclude_toggled = Signal(object)

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._records: list[ImageRecord] = []
        self._cards: dict[str, ImageCard] = {}
        self._sort_order = SortOrder.SCORE
        self._last_columns = -1
        self._layouting = False
        self._thumbnail_pending: set[str] = set()
        self._thumbnail_loaded: set[str] = set()
        self._thumb_pool = ThumbnailPool(
            on_done=self._on_thumbnail_done,
            max_threads=4,
        )
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._grid.setSpacing(8)
        self._grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setWidget(self._container)

    def set_sort_order(self, order: SortOrder) -> None:
        self._sort_order = order
        self._relayout_cards()

    def load_records(self, records: list[ImageRecord]) -> None:
        """Replace records while reusing existing cards and thumbnails."""
        self._records = list(records)
        current_paths = {record.path for record in self._records}

        # Remove obsolete cards without disturbing cards that can be reused.
        for path, card in list(self._cards.items()):
            if path in current_paths:
                continue
            self._grid.removeWidget(card)
            card.deleteLater()
            self._cards.pop(path, None)
            self._thumbnail_pending.discard(path)
            self._thumbnail_loaded.discard(path)

        # Create only cards that do not already exist.
        for record in self._records:
            card = self._cards.get(record.path)
            if card is None:
                card = self._create_card(record)
                self._cards[record.path] = card
                self._request_thumbnail(record.path)
            else:
                card.update_from_record(record)

        self._relayout_cards()

    def update_record(self, record: ImageRecord) -> None:
        """Update a single existing card in place."""
        card = self._cards.get(record.path)
        if card:
            card.update_from_record(record)
            self._request_thumbnail(record.path)
            return

        self._records.append(record)
        card = self._create_card(record)
        self._cards[record.path] = card
        self._request_thumbnail(record.path)
        self._relayout_cards()

    def upsert_record(self, record: ImageRecord) -> None:
        for i, current in enumerate(self._records):
            if current.path == record.path:
                self._records[i] = record
                self.update_record(record)
                return
        self._records.append(record)
        card = self._create_card(record)
        self._cards[record.path] = card
        self._request_thumbnail(record.path)
        self._relayout_cards()

    def clear(self) -> None:
        self._records.clear()
        for card in self._cards.values():
            card.deleteLater()
        self._cards.clear()
        self._thumbnail_pending.clear()
        self._thumbnail_loaded.clear()
        self._clear_layout_items()
        self._last_columns = -1

    def get_selected_records(self) -> list[ImageRecord]:
        return [r for r in self._records if r.status == ImageStatus.SELECTED]

    def _sorted_records(self) -> list[ImageRecord]:
        key_fn = {
            SortOrder.SCORE:    lambda r: -(r.ranking.final_score if r.ranking else 0),
            SortOrder.RANK:     lambda r: r.ranking.rank if (r.ranking and r.ranking.rank > 0) else 9999,
            SortOrder.FILENAME: lambda r: r.filename.lower(),
            SortOrder.DATE:     lambda r: r.filename.lower(),
        }.get(self._sort_order, lambda r: r.filename.lower())
        return sorted(self._records, key=key_fn)

    def _clear_layout_items(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            # Do not delete the widget here; callers own its lifetime.

    def _create_card(self, record: ImageRecord) -> ImageCard:
        card = ImageCard(record, self._container)
        card.clicked.connect(self.image_selected.emit)
        card.include_toggled.connect(self.include_toggled.emit)
        card.exclude_toggled.connect(self.exclude_toggled.emit)
        return card

    def _relayout_cards(self) -> None:
        if self._layouting:
            return

        self._layouting = True
        try:
            self._clear_layout_items()
            sorted_records = self._sorted_records()
            cols = max(1, self._column_count())
            self._last_columns = cols

            for idx, record in enumerate(sorted_records):
                card = self._cards.get(record.path)
                if card is None:
                    continue
                row = idx // cols
                col = idx % cols
                self._grid.addWidget(card, row, col)
        finally:
            self._layouting = False

    def _column_count(self) -> int:
        available_width = max(1, self.viewport().width())
        return max(1, available_width // (_CARD_W + 8))

    def _request_thumbnail(self, path: str) -> None:
        if not path or path in self._thumbnail_pending or path in self._thumbnail_loaded:
            return
        self._thumbnail_pending.add(path)
        self._thumb_pool.request(path, _THUMB_SIZE)

    def _on_thumbnail_done(self, path: str, data: bytes) -> None:
        self._thumbnail_pending.discard(path)
        if data:
            self._thumbnail_loaded.add(path)
        card = self._cards.get(path)
        if card and data:
            card.set_thumbnail(data)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._layouting:
            return

        columns = self._column_count()
        if columns == self._last_columns:
            return
        self._relayout_cards()
