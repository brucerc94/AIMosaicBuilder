"""Settings panel for AI Mosaic Builder."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget,
)

from engine.models import AppSettings

logger = logging.getLogger("ui.settings")


class SettingsPanel(QWidget):
    settings_changed = Signal(object)
    analyze_requested = Signal()
    stop_requested = Signal()
    generate_requested = Signal()

    def __init__(self, settings: AppSettings, parent: QWidget = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._building = True
        self._build_ui()
        self._building = False
        self._populate(settings)

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

        grp_src = QGroupBox("Source")
        f_src = QFormLayout(grp_src)
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

        grp_model = QGroupBox("Vision Model")
        f_model = QFormLayout(grp_model)
        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText("Path to .gguf model…")
        btn_model = QPushButton("Browse…")
        btn_model.clicked.connect(self._browse_model)
        row_model = QHBoxLayout(); row_model.addWidget(self._model_edit); row_model.addWidget(btn_model)
        f_model.addRow("Model:", row_model)
        self._mmproj_edit = QLineEdit()
        self._mmproj_edit.setPlaceholderText("mmproj .gguf (for vision)…")
        btn_mmproj = QPushButton("Browse…")
        btn_mmproj.clicked.connect(self._browse_mmproj)
        row_mm = QHBoxLayout(); row_mm.addWidget(self._mmproj_edit); row_mm.addWidget(btn_mmproj)
        f_model.addRow("mmproj:", row_mm)
        layout.addWidget(grp_model)

        grp_inf = QGroupBox("Inference")
        f_inf = QFormLayout(grp_inf)
        self._gpu_spin = QSpinBox(); self._gpu_spin.setRange(-1, 999); self._gpu_spin.setSuffix(" layers")
        self._gpu_spin.setToolTip("-1 = all layers, 0 = CPU only")
        f_inf.addRow("GPU Layers:", self._gpu_spin)
        self._ctx_spin = QSpinBox(); self._ctx_spin.setRange(512, 131072); self._ctx_spin.setSingleStep(512)
        self._ctx_spin.setSuffix(" tokens"); f_inf.addRow("Context:", self._ctx_spin)
        self._threads_spin = QSpinBox(); self._threads_spin.setRange(1, 64); f_inf.addRow("CPU Threads:", self._threads_spin)
        self._batch_threads_spin = QSpinBox(); self._batch_threads_spin.setRange(0, 64); f_inf.addRow("Batch Threads:", self._batch_threads_spin)
        layout.addWidget(grp_inf)

        grp_detector = QGroupBox("Person Detector")
        f_det = QFormLayout(grp_detector)
        self._detector_model_edit = QLineEdit(); self._detector_model_edit.setPlaceholderText("YOLO model, e.g. yolo26n.pt")
        btn_detector = QPushButton("Browse…"); btn_detector.clicked.connect(self._browse_detector_model)
        row_det = QHBoxLayout(); row_det.addWidget(self._detector_model_edit); row_det.addWidget(btn_detector)
        f_det.addRow("YOLO Model:", row_det)
        self._detector_conf_spin = QDoubleSpinBox(); self._detector_conf_spin.setRange(0.05, 0.95); self._detector_conf_spin.setSingleStep(0.05); self._detector_conf_spin.setDecimals(2)
        f_det.addRow("Confidence:", self._detector_conf_spin)
        layout.addWidget(grp_detector)

        grp_mosaic = QGroupBox("Mosaic")
        f_mosaic = QFormLayout(grp_mosaic)
        self._target_combo = QComboBox()
        for val in [0, 4, 6, 9, 12, 16, 20]:
            self._target_combo.addItem("Automatic" if val == 0 else str(val), val)
        f_mosaic.addRow("Target Images:", self._target_combo)
        self._padding_spin = QSpinBox(); self._padding_spin.setRange(0, 500); self._padding_spin.setSuffix(" px"); f_mosaic.addRow("Crop Padding:", self._padding_spin)
        self._canvas_w_spin = QSpinBox(); self._canvas_w_spin.setRange(800, 16000); self._canvas_w_spin.setSingleStep(240); self._canvas_w_spin.setSuffix(" px"); f_mosaic.addRow("Canvas Width:", self._canvas_w_spin)
        self._canvas_h_spin = QSpinBox(); self._canvas_h_spin.setRange(600, 12000); self._canvas_h_spin.setSingleStep(240); self._canvas_h_spin.setSuffix(" px"); f_mosaic.addRow("Canvas Height:", self._canvas_h_spin)
        layout.addWidget(grp_mosaic)

        grp_thresh = QGroupBox("Thresholds")
        f_thresh = QFormLayout(grp_thresh)
        self._min_w_spin = QSpinBox(); self._min_w_spin.setRange(50, 2000); self._min_w_spin.setSuffix(" px"); f_thresh.addRow("Min Width:", self._min_w_spin)
        self._min_h_spin = QSpinBox(); self._min_h_spin.setRange(50, 2000); self._min_h_spin.setSuffix(" px"); f_thresh.addRow("Min Height:", self._min_h_spin)
        self._similarity_spin = QSpinBox(); self._similarity_spin.setRange(0, 64); f_thresh.addRow("Similarity Threshold:", self._similarity_spin)
        layout.addWidget(grp_thresh)

        grp_actions = QGroupBox("Actions")
        f_actions = QVBoxLayout(grp_actions)
        self._btn_analyze = QPushButton("▶  Analyze"); self._btn_analyze.setObjectName("primary_btn"); self._btn_analyze.clicked.connect(self.analyze_requested.emit)
        self._btn_stop = QPushButton("■  Stop"); self._btn_stop.setObjectName("danger_btn"); self._btn_stop.setEnabled(False); self._btn_stop.clicked.connect(self.stop_requested.emit)
        self._btn_generate = QPushButton("🖼  Generate Mosaic"); self._btn_generate.setObjectName("success_btn"); self._btn_generate.setEnabled(False); self._btn_generate.clicked.connect(self.generate_requested.emit)
        f_actions.addWidget(self._btn_analyze); f_actions.addWidget(self._btn_stop); f_actions.addWidget(self._btn_generate)
        layout.addWidget(grp_actions); layout.addStretch()
        scroll.setWidget(container); outer.addWidget(scroll)

        widgets = [self._folder_edit, self._model_edit, self._mmproj_edit, self._gpu_spin, self._ctx_spin, self._threads_spin, self._batch_threads_spin, self._detector_model_edit, self._detector_conf_spin, self._target_combo, self._padding_spin, self._canvas_w_spin, self._canvas_h_spin, self._min_w_spin, self._min_h_spin, self._similarity_spin]
        for widget in widgets:
            signal = widget.textChanged if isinstance(widget, QLineEdit) else widget.currentIndexChanged if isinstance(widget, QComboBox) else widget.valueChanged
            signal.connect(self._on_changed)

    def _populate(self, s: AppSettings) -> None:
        self._building = True
        self._folder_edit.setText(s.last_source_folder); self._model_edit.setText(s.model_path); self._mmproj_edit.setText(s.mmproj_path)
        self._gpu_spin.setValue(s.n_gpu_layers); self._ctx_spin.setValue(s.n_ctx); self._threads_spin.setValue(s.n_threads); self._batch_threads_spin.setValue(s.n_threads_batch)
        self._detector_model_edit.setText(s.person_detector_model); self._detector_conf_spin.setValue(s.person_detector_confidence)
        idx = self._target_combo.findData(s.target_images); self._target_combo.setCurrentIndex(max(0, idx if idx >= 0 else self._target_combo.findData(12)))
        self._padding_spin.setValue(s.padding_px); self._canvas_w_spin.setValue(s.canvas_width); self._canvas_h_spin.setValue(s.canvas_height); self._min_w_spin.setValue(s.min_image_width); self._min_h_spin.setValue(s.min_image_height); self._similarity_spin.setValue(s.phash_threshold)
        self._building = False

    def current_settings(self) -> AppSettings:
        s = self._settings
        s.last_source_folder = self._folder_edit.text().strip(); s.model_path = self._model_edit.text().strip(); s.mmproj_path = self._mmproj_edit.text().strip()
        s.n_gpu_layers = self._gpu_spin.value(); s.n_ctx = self._ctx_spin.value(); s.n_threads = self._threads_spin.value(); s.n_threads_batch = self._batch_threads_spin.value()
        s.person_detector_model = self._detector_model_edit.text().strip(); s.person_detector_confidence = self._detector_conf_spin.value()
        s.target_images = self._target_combo.currentData(); s.padding_px = self._padding_spin.value(); s.canvas_width = self._canvas_w_spin.value(); s.canvas_height = self._canvas_h_spin.value(); s.min_image_width = self._min_w_spin.value(); s.min_image_height = self._min_h_spin.value(); s.phash_threshold = self._similarity_spin.value()
        return s

    def set_analyzing(self, active: bool) -> None:
        self._btn_analyze.setEnabled(not active); self._btn_stop.setEnabled(active)

    def set_generate_enabled(self, enabled: bool) -> None:
        self._btn_generate.setEnabled(enabled)

    def _on_changed(self, *_) -> None:
        if not self._building: self.settings_changed.emit(self.current_settings())

    def _browse_folder(self) -> None:
        current = self._folder_edit.text().strip(); start = current if current and os.path.isdir(current) else os.path.expanduser("~")
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder", start)
        if folder: self._folder_edit.setText(folder)

    def _browse_model(self) -> None:
        start = os.path.dirname(self._model_edit.text()) or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF Model", start, "GGUF Models (*.gguf);;All Files (*)")
        if path: self._model_edit.setText(path)

    def _browse_mmproj(self) -> None:
        start = os.path.dirname(self._mmproj_edit.text()) or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Select mmproj File", start, "GGUF Files (*.gguf);;All Files (*)")
        if path: self._mmproj_edit.setText(path)

    def _browse_detector_model(self) -> None:
        start = os.path.dirname(self._detector_model_edit.text()) or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", start, "Model Files (*.pt *.onnx);;All Files (*)")
        if path: self._detector_model_edit.setText(path)
