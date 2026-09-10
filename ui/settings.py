"""
Settings panel for AI Mosaic Builder.
Left-side configuration panel following AIStoryWriter's ui/settings.py pattern.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from engine.models import AppSettings
from engine.storage import list_gguf_models

logger = logging.getLogger("ui.settings")


class SettingsPanel(QWidget):
    """
    Left-side configuration panel.
    Emits settings_changed(AppSettings) whenever the user modifies any field.
    Emits analyze_requested() when the Analyze button is clicked.
    Emits stop_requested() when Stop is clicked.
    Emits generate_requested() when Generate Mosaic is clicked.
    """

    settings_changed   = Signal(object)   # AppSettings
    analyze_requested  = Signal()
    stop_requested     = Signal()
    generate_requested = Signal()

    def __init__(self, settings: AppSettings, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._building = True   # suppress signals during build
        self._build_ui()
        self._building = False
        self._populate(settings)

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # ── Source folder ──────────────────────────────────────────────────
        grp_src = QGroupBox("Source")
        f_src = QFormLayout(grp_src)
        f_src.setSpacing(6)

        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Select folder…")
        self._folder_edit.setReadOnly(True)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_folder)

        row_folder = QHBoxLayout()
        row_folder.addWidget(self._folder_edit)
        row_folder.addWidget(btn_browse)
        f_src.addRow("Folder:", row_folder)
        layout.addWidget(grp_src)

        # ── Model ──────────────────────────────────────────────────────────
        grp_model = QGroupBox("Vision Model")
        f_model = QFormLayout(grp_model)
        f_model.setSpacing(6)

        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText("Path to .gguf model…")
        btn_model = QPushButton("Browse…")
        btn_model.clicked.connect(self._browse_model)
        row_model = QHBoxLayout()
        row_model.addWidget(self._model_edit)
        row_model.addWidget(btn_model)
        f_model.addRow("Model:", row_model)

        self._mmproj_edit = QLineEdit()
        self._mmproj_edit.setPlaceholderText("mmproj .gguf (for vision)…")
        btn_mmproj = QPushButton("Browse…")
        btn_mmproj.clicked.connect(self._browse_mmproj)
        row_mm = QHBoxLayout()
        row_mm.addWidget(self._mmproj_edit)
        row_mm.addWidget(btn_mmproj)
        f_model.addRow("mmproj:", row_mm)

        layout.addWidget(grp_model)

        # ── Inference ──────────────────────────────────────────────────────
        grp_inf = QGroupBox("Inference")
        f_inf = QFormLayout(grp_inf)
        f_inf.setSpacing(6)

        self._gpu_spin = QSpinBox()
        self._gpu_spin.setRange(0, 999)
        self._gpu_spin.setSuffix(" layers")
        self._gpu_spin.setToolTip("Number of layers to offload to GPU. 0 = CPU only, -1 = all.")
        f_inf.addRow("GPU Layers:", self._gpu_spin)

        self._ctx_spin = QSpinBox()
        self._ctx_spin.setRange(512, 131072)
        self._ctx_spin.setSingleStep(512)
        self._ctx_spin.setSuffix(" tokens")
        f_inf.addRow("Context:", self._ctx_spin)

        self._threads_spin = QSpinBox()
        self._threads_spin.setRange(1, 64)
        f_inf.addRow("CPU Threads:", self._threads_spin)

        self._batch_threads_spin = QSpinBox()
        self._batch_threads_spin.setRange(0, 64)
        self._batch_threads_spin.setToolTip("0 = same as CPU Threads")
        f_inf.addRow("Batch Threads:", self._batch_threads_spin)

        layout.addWidget(grp_inf)

        # ── Mosaic parameters ──────────────────────────────────────────────
        grp_mosaic = QGroupBox("Mosaic")
        f_mosaic = QFormLayout(grp_mosaic)
        f_mosaic.setSpacing(6)

        self._target_combo = QComboBox()
        for val in [0, 4, 6, 9, 12, 16, 20]:
            label = "Automatic" if val == 0 else str(val)
            self._target_combo.addItem(label, val)
        f_mosaic.addRow("Target Images:", self._target_combo)

        self._padding_spin = QSpinBox()
        self._padding_spin.setRange(0, 500)
        self._padding_spin.setSuffix(" px")
        f_mosaic.addRow("Crop Padding:", self._padding_spin)

        self._canvas_w_spin = QSpinBox()
        self._canvas_w_spin.setRange(800, 16000)
        self._canvas_w_spin.setSingleStep(240)
        self._canvas_w_spin.setSuffix(" px")
        f_mosaic.addRow("Canvas Width:", self._canvas_w_spin)

        self._canvas_h_spin = QSpinBox()
        self._canvas_h_spin.setRange(600, 12000)
        self._canvas_h_spin.setSingleStep(240)
        self._canvas_h_spin.setSuffix(" px")
        f_mosaic.addRow("Canvas Height:", self._canvas_h_spin)

        layout.addWidget(grp_mosaic)

        # ── Analysis thresholds ────────────────────────────────────────────
        grp_thresh = QGroupBox("Thresholds")
        f_thresh = QFormLayout(grp_thresh)
        f_thresh.setSpacing(6)

        self._min_w_spin = QSpinBox()
        self._min_w_spin.setRange(50, 2000)
        self._min_w_spin.setSuffix(" px")
        f_thresh.addRow("Min Width:", self._min_w_spin)

        self._min_h_spin = QSpinBox()
        self._min_h_spin.setRange(50, 2000)
        self._min_h_spin.setSuffix(" px")
        f_thresh.addRow("Min Height:", self._min_h_spin)

        self._similarity_spin = QSpinBox()
        self._similarity_spin.setRange(0, 64)
        self._similarity_spin.setToolTip("Perceptual hash Hamming distance. Lower = stricter.")
        f_thresh.addRow("Similarity Threshold:", self._similarity_spin)

        layout.addWidget(grp_thresh)

        # ── Action buttons ─────────────────────────────────────────────────
        grp_actions = QGroupBox("Actions")
        f_actions = QVBoxLayout(grp_actions)
        f_actions.setSpacing(8)

        self._btn_analyze = QPushButton("▶  Analyze")
        self._btn_analyze.setObjectName("primary_btn")
        self._btn_analyze.setToolTip("Start LLM analysis of selected folder")
        self._btn_analyze.clicked.connect(self.analyze_requested.emit)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setObjectName("danger_btn")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self.stop_requested.emit)

        self._btn_generate = QPushButton("🖼  Generate Mosaic")
        self._btn_generate.setObjectName("success_btn")
        self._btn_generate.setEnabled(False)
        self._btn_generate.clicked.connect(self.generate_requested.emit)

        f_actions.addWidget(self._btn_analyze)
        f_actions.addWidget(self._btn_stop)
        f_actions.addWidget(self._btn_generate)

        layout.addWidget(grp_actions)
        layout.addStretch()

        scroll.setWidget(container)
        outer.addWidget(scroll)

        # Connect change signals
        self._folder_edit.textChanged.connect(self._on_changed)
        self._model_edit.textChanged.connect(self._on_changed)
        self._mmproj_edit.textChanged.connect(self._on_changed)
        self._gpu_spin.valueChanged.connect(self._on_changed)
        self._ctx_spin.valueChanged.connect(self._on_changed)
        self._threads_spin.valueChanged.connect(self._on_changed)
        self._batch_threads_spin.valueChanged.connect(self._on_changed)
        self._target_combo.currentIndexChanged.connect(self._on_changed)
        self._padding_spin.valueChanged.connect(self._on_changed)
        self._canvas_w_spin.valueChanged.connect(self._on_changed)
        self._canvas_h_spin.valueChanged.connect(self._on_changed)
        self._min_w_spin.valueChanged.connect(self._on_changed)
        self._min_h_spin.valueChanged.connect(self._on_changed)
        self._similarity_spin.valueChanged.connect(self._on_changed)

    # ── Populate from settings ─────────────────────────────────────────────

    def _populate(self, s: AppSettings) -> None:
        self._building = True
        self._folder_edit.setText(s.last_source_folder)
        self._model_edit.setText(s.model_path)
        self._mmproj_edit.setText(s.mmproj_path)
        self._gpu_spin.setValue(s.n_gpu_layers)
        self._ctx_spin.setValue(s.n_ctx)
        self._threads_spin.setValue(s.n_threads)
        self._batch_threads_spin.setValue(s.n_threads_batch)
        # Target combo
        idx = self._target_combo.findData(s.target_images)
        if idx < 0:
            idx = self._target_combo.findData(12)
        self._target_combo.setCurrentIndex(max(0, idx))
        self._padding_spin.setValue(s.padding_px)
        self._canvas_w_spin.setValue(s.canvas_width)
        self._canvas_h_spin.setValue(s.canvas_height)
        self._min_w_spin.setValue(s.min_image_width)
        self._min_h_spin.setValue(s.min_image_height)
        self._similarity_spin.setValue(s.phash_threshold)
        self._building = False

    # ── Read current values → AppSettings ─────────────────────────────────

    def current_settings(self) -> AppSettings:
        s = self._settings
        s.last_source_folder   = self._folder_edit.text().strip()
        s.model_path           = self._model_edit.text().strip()
        s.mmproj_path          = self._mmproj_edit.text().strip()
        s.n_gpu_layers         = self._gpu_spin.value()
        s.n_ctx                = self._ctx_spin.value()
        s.n_threads            = self._threads_spin.value()
        s.n_threads_batch      = self._batch_threads_spin.value()
        s.target_images        = self._target_combo.currentData()
        s.padding_px           = self._padding_spin.value()
        s.canvas_width         = self._canvas_w_spin.value()
        s.canvas_height        = self._canvas_h_spin.value()
        s.min_image_width      = self._min_w_spin.value()
        s.min_image_height     = self._min_h_spin.value()
        s.phash_threshold      = self._similarity_spin.value()
        return s

    # ── State control (called by main window) ──────────────────────────────

    def set_analyzing(self, active: bool) -> None:
        self._btn_analyze.setEnabled(not active)
        self._btn_stop.setEnabled(active)

    def set_generate_enabled(self, enabled: bool) -> None:
        self._btn_generate.setEnabled(enabled)

    # ── Slots ──────────────────────────────────────────────────────────────

    def _on_changed(self, *_) -> None:
        if not self._building:
            self.settings_changed.emit(self.current_settings())

    def _browse_folder(self) -> None:
        current = self._folder_edit.text().strip()
        start = current if current and os.path.isdir(current) else os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", start)
        if folder:
            self._folder_edit.setText(folder)

    def _browse_model(self) -> None:
        start = os.path.dirname(self._model_edit.text()) or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", start, "GGUF Models (*.gguf);;All Files (*)"
        )
        if path:
            self._model_edit.setText(path)

    def _browse_mmproj(self) -> None:
        start = os.path.dirname(self._mmproj_edit.text()) or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select mmproj File", start, "GGUF Files (*.gguf);;All Files (*)"
        )
        if path:
            self._mmproj_edit.setText(path)
