"""Settings panel for AI Mosaic Builder."""
from __future__ import annotations

import logging
import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
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

        grp_inf = QGroupBox("Inference")
        f_inf = QFormLayout(grp_inf)
        self._gpu_spin = QSpinBox()
        self._gpu_spin.setRange(-1, 999)
        self._gpu_spin.setSuffix(" layers")
        self._gpu_spin.setToolTip("-1 = all layers, 0 = CPU only")
        f_inf.addRow("GPU Layers:", self._gpu_spin)
        self._ctx_spin = QSpinBox()
        self._ctx_spin.setRange(2048, 131072)
        self._ctx_spin.setSingleStep(512)
        self._ctx_spin.setSuffix(" tokens")
        self._ctx_spin.setToolTip("Minimum 2048 tokens for Gemma-4 vision analysis.")
        f_inf.addRow("Context:", self._ctx_spin)
        self._threads_spin = QSpinBox()
        self._threads_spin.setRange(1, 64)
        f_inf.addRow("CPU Threads:", self._threads_spin)
        self._batch_threads_spin = QSpinBox()
        self._batch_threads_spin.setRange(0, 64)
        self._batch_threads_spin.setToolTip("0 = use CPU Threads value. Controls parallel token evaluation on CPU.")
        f_inf.addRow("Batch Threads:", self._batch_threads_spin)
        self._n_batch_spin = QSpinBox()
        self._n_batch_spin.setRange(64, 4096)
        self._n_batch_spin.setSingleStep(64)
        self._n_batch_spin.setToolTip("Tokens evaluated per batch. 512 recommended. Higher = faster prompt eval, more VRAM.")
        f_inf.addRow("n_batch:", self._n_batch_spin)
        self._n_ubatch_spin = QSpinBox()
        self._n_ubatch_spin.setRange(64, 4096)
        self._n_ubatch_spin.setSingleStep(64)
        self._n_ubatch_spin.setToolTip("Micro-batch size for CUDA. Must be ≤ n_batch. 512 recommended for GTX 1660 Ti.")
        f_inf.addRow("n_ubatch:", self._n_ubatch_spin)
        layout.addWidget(grp_inf)

        grp_detector = QGroupBox("Person Detector")
        f_det = QFormLayout(grp_detector)
        self._detector_model_edit = QLineEdit()
        self._detector_model_edit.setPlaceholderText("YOLO model, e.g. yolo26n.pt")
        btn_detector = QPushButton("Browse…")
        btn_detector.clicked.connect(self._browse_detector_model)
        row_det = QHBoxLayout()
        row_det.addWidget(self._detector_model_edit)
        row_det.addWidget(btn_detector)
        f_det.addRow("YOLO Model:", row_det)
        self._detector_conf_spin = QDoubleSpinBox()
        self._detector_conf_spin.setRange(0.05, 0.95)
        self._detector_conf_spin.setSingleStep(0.05)
        self._detector_conf_spin.setDecimals(2)
        f_det.addRow("Confidence:", self._detector_conf_spin)
        layout.addWidget(grp_detector)

        grp_mosaic = QGroupBox("Mosaic")
        f_mosaic = QFormLayout(grp_mosaic)
        self._target_combo = QComboBox()
        for val in [0, 4, 6, 9, 12, 16, 20]:
            self._target_combo.addItem("Automatic" if val == 0 else str(val), val)
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

        grp_req = QGroupBox("Mosaic Requirements")
        req_layout = QVBoxLayout(grp_req)
        req_layout.addWidget(QLabel("Minimum required images (0 = no requirement)"))
        req_form = QFormLayout()
        self._req_face_only = QSpinBox(); self._req_face_only.setRange(0, 99)
        self._req_full_body = QSpinBox(); self._req_full_body.setRange(0, 99)
        self._req_front = QSpinBox(); self._req_front.setRange(0, 99)
        self._req_side = QSpinBox(); self._req_side.setRange(0, 99)
        self._req_back = QSpinBox(); self._req_back.setRange(0, 99)
        self._req_male = QSpinBox(); self._req_male.setRange(0, 99)
        self._req_female = QSpinBox(); self._req_female.setRange(0, 99)
        self._req_face_visible = QSpinBox(); self._req_face_visible.setRange(0, 99)
        self._req_body_visible = QSpinBox(); self._req_body_visible.setRange(0, 99)
        req_form.addRow("Face only:", self._req_face_only)
        req_form.addRow("Full body:", self._req_full_body)
        req_form.addRow("Front:", self._req_front)
        req_form.addRow("Side:", self._req_side)
        req_form.addRow("Back:", self._req_back)
        req_form.addRow("Male:", self._req_male)
        req_form.addRow("Female:", self._req_female)
        req_form.addRow("Face visible:", self._req_face_visible)
        req_form.addRow("Body visible:", self._req_body_visible)
        req_layout.addLayout(req_form)

        content_row = QHBoxLayout()
        content_row.addWidget(QLabel("Content:"))
        self._req_nsfw = QComboBox()
        self._req_nsfw.addItem("Don't care", "any")
        self._req_nsfw.addItem("Safe only", "safe_only")
        self._req_nsfw.addItem("NSFW only", "nsfw_only")
        content_row.addWidget(self._req_nsfw)
        req_layout.addLayout(content_row)

        self._req_exclude_blurry = QCheckBox("Exclude blurry images")
        self._req_exclude_occluded = QCheckBox("Exclude heavily occluded images")
        req_layout.addWidget(self._req_exclude_blurry)
        req_layout.addWidget(self._req_exclude_occluded)

        quality_form = QFormLayout()
        self._req_min_quality = QDoubleSpinBox(); self._req_min_quality.setRange(0.0, 1.0); self._req_min_quality.setSingleStep(0.05); self._req_min_quality.setDecimals(2); self._req_min_quality.setSuffix(" / 1")
        self._req_min_visibility = QDoubleSpinBox(); self._req_min_visibility.setRange(0.0, 1.0); self._req_min_visibility.setSingleStep(0.05); self._req_min_visibility.setDecimals(2); self._req_min_visibility.setSuffix(" / 1")
        quality_form.addRow("Min image quality:", self._req_min_quality)
        quality_form.addRow("Min person visibility:", self._req_min_visibility)
        req_layout.addLayout(quality_form)

        req_layout.addWidget(QLabel("Soft preferences (used to break ties and improve the recipe)"))
        self._pref_face_only = QCheckBox("Prefer face-only")
        self._pref_full_body = QCheckBox("Prefer full-body")
        self._pref_front = QCheckBox("Prefer front")
        self._pref_side = QCheckBox("Prefer side")
        self._pref_back = QCheckBox("Prefer back")
        self._pref_solo = QCheckBox("Prefer solo person")
        self._pref_face_visible = QCheckBox("Prefer face visible")
        self._pref_body_visible = QCheckBox("Prefer body visible")
        for widget in (
            self._pref_face_only, self._pref_full_body, self._pref_front, self._pref_side,
            self._pref_back, self._pref_solo, self._pref_face_visible, self._pref_body_visible,
        ):
            req_layout.addWidget(widget)
        layout.addWidget(grp_req)

        grp_thresh = QGroupBox("Thresholds")
        f_thresh = QFormLayout(grp_thresh)
        self._min_w_spin = QSpinBox(); self._min_w_spin.setRange(50, 2000); self._min_w_spin.setSuffix(" px"); f_thresh.addRow("Min Width:", self._min_w_spin)
        self._min_h_spin = QSpinBox(); self._min_h_spin.setRange(50, 2000); self._min_h_spin.setSuffix(" px"); f_thresh.addRow("Min Height:", self._min_h_spin)
        self._similarity_spin = QSpinBox(); self._similarity_spin.setRange(0, 64); f_thresh.addRow("Similarity Threshold:", self._similarity_spin)
        layout.addWidget(grp_thresh)

        grp_actions = QGroupBox("Actions")
        f_actions = QVBoxLayout(grp_actions)
        self._btn_analyze = QPushButton("▶  Analyze")
        self._btn_analyze.setObjectName("primary_btn")
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

        widgets = [
            self._folder_edit, self._model_edit, self._mmproj_edit, self._gpu_spin, self._ctx_spin,
            self._threads_spin, self._batch_threads_spin, self._detector_model_edit,
            self._detector_conf_spin, self._target_combo, self._padding_spin, self._canvas_w_spin,
            self._canvas_h_spin, self._min_w_spin, self._min_h_spin, self._similarity_spin,
            self._req_face_only, self._req_full_body, self._req_front, self._req_side, self._req_back,
            self._req_male, self._req_female, self._req_face_visible, self._req_body_visible,
            self._req_nsfw, self._req_exclude_blurry, self._req_exclude_occluded,
            self._req_min_quality, self._req_min_visibility, self._pref_face_only,
            self._pref_full_body, self._pref_front, self._pref_side, self._pref_back,
            self._pref_solo, self._pref_face_visible, self._pref_body_visible,
        ]
        for widget in widgets:
            if isinstance(widget, QLineEdit):
                signal = widget.textChanged
            elif isinstance(widget, QComboBox):
                signal = widget.currentIndexChanged
            elif isinstance(widget, QCheckBox):
                signal = widget.stateChanged
            else:
                signal = widget.valueChanged
            signal.connect(self._on_changed)

    def _populate(self, s: AppSettings) -> None:
        self._building = True
        self._folder_edit.setText(s.last_source_folder)
        self._model_edit.setText(s.model_path)
        self._mmproj_edit.setText(s.mmproj_path)
        self._gpu_spin.setValue(s.n_gpu_layers)
        self._ctx_spin.setValue(max(2048, int(s.n_ctx)))
        self._threads_spin.setValue(s.n_threads)
        self._batch_threads_spin.setValue(s.n_threads_batch)
        self._n_batch_spin.setValue(s.n_batch)
        self._n_ubatch_spin.setValue(s.n_ubatch)
        self._detector_model_edit.setText(s.person_detector_model)
        self._detector_conf_spin.setValue(s.person_detector_confidence)
        idx = self._target_combo.findData(s.target_images)
        self._target_combo.setCurrentIndex(max(0, idx if idx >= 0 else self._target_combo.findData(12)))
        self._padding_spin.setValue(s.padding_px)
        self._canvas_w_spin.setValue(s.canvas_width)
        self._canvas_h_spin.setValue(s.canvas_height)
        self._min_w_spin.setValue(s.min_image_width)
        self._min_h_spin.setValue(s.min_image_height)
        self._similarity_spin.setValue(s.phash_threshold)

        r = s.mosaic_requirements
        self._req_face_only.setValue(r.min_face_only)
        self._req_full_body.setValue(r.min_full_body)
        self._req_front.setValue(r.min_front)
        self._req_side.setValue(r.min_side)
        self._req_back.setValue(r.min_back)
        self._req_male.setValue(r.min_male)
        self._req_female.setValue(r.min_female)
        self._req_face_visible.setValue(r.min_face_visible)
        self._req_body_visible.setValue(r.min_body_visible)
        idx = self._req_nsfw.findData(r.nsfw_policy)
        self._req_nsfw.setCurrentIndex(max(0, idx))
        self._req_exclude_blurry.setChecked(r.exclude_blurry)
        self._req_exclude_occluded.setChecked(r.exclude_occluded)
        self._req_min_quality.setValue(r.min_quality)
        self._req_min_visibility.setValue(r.min_person_visibility)
        self._pref_face_only.setChecked(r.prefer_face_only)
        self._pref_full_body.setChecked(r.prefer_full_body)
        self._pref_front.setChecked(r.prefer_front)
        self._pref_side.setChecked(r.prefer_side)
        self._pref_back.setChecked(r.prefer_back)
        self._pref_solo.setChecked(r.prefer_solo)
        self._pref_face_visible.setChecked(r.prefer_face_visible)
        self._pref_body_visible.setChecked(r.prefer_body_visible)
        self._building = False

    def current_settings(self) -> AppSettings:
        s = self._settings
        s.last_source_folder = self._folder_edit.text().strip()
        s.model_path = self._model_edit.text().strip()
        s.mmproj_path = self._mmproj_edit.text().strip()
        s.n_gpu_layers = self._gpu_spin.value()
        s.n_ctx = max(2048, self._ctx_spin.value())
        s.n_threads = self._threads_spin.value()
        s.n_threads_batch = self._batch_threads_spin.value()
        s.n_batch = self._n_batch_spin.value()
        s.n_ubatch = min(self._n_ubatch_spin.value(), self._n_batch_spin.value())
        s.person_detector_model = self._detector_model_edit.text().strip()
        s.person_detector_confidence = self._detector_conf_spin.value()
        s.target_images = self._target_combo.currentData()
        s.padding_px = self._padding_spin.value()
        s.canvas_width = self._canvas_w_spin.value()
        s.canvas_height = self._canvas_h_spin.value()
        s.min_image_width = self._min_w_spin.value()
        s.min_image_height = self._min_h_spin.value()
        s.phash_threshold = self._similarity_spin.value()

        r = s.mosaic_requirements
        r.min_face_only = self._req_face_only.value()
        r.min_full_body = self._req_full_body.value()
        r.min_front = self._req_front.value()
        r.min_side = self._req_side.value()
        r.min_back = self._req_back.value()
        r.min_male = self._req_male.value()
        r.min_female = self._req_female.value()
        r.min_face_visible = self._req_face_visible.value()
        r.min_body_visible = self._req_body_visible.value()
        r.nsfw_policy = str(self._req_nsfw.currentData() or "any")
        r.exclude_blurry = self._req_exclude_blurry.isChecked()
        r.exclude_occluded = self._req_exclude_occluded.isChecked()
        r.min_quality = self._req_min_quality.value()
        r.min_person_visibility = self._req_min_visibility.value()
        r.prefer_face_only = self._pref_face_only.isChecked()
        r.prefer_full_body = self._pref_full_body.isChecked()
        r.prefer_front = self._pref_front.isChecked()
        r.prefer_side = self._pref_side.isChecked()
        r.prefer_back = self._pref_back.isChecked()
        r.prefer_solo = self._pref_solo.isChecked()
        r.prefer_face_visible = self._pref_face_visible.isChecked()
        r.prefer_body_visible = self._pref_body_visible.isChecked()
        return s

    def set_analyzing(self, active: bool) -> None:
        self._btn_analyze.setEnabled(not active)
        self._btn_stop.setEnabled(active)

    def set_generate_enabled(self, enabled: bool) -> None:
        self._btn_generate.setEnabled(enabled)

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
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF Model", start, "GGUF Models (*.gguf);;All Files (*)")
        if path:
            self._model_edit.setText(path)

    def _browse_mmproj(self) -> None:
        start = os.path.dirname(self._mmproj_edit.text()) or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Select mmproj File", start, "GGUF Files (*.gguf);;All Files (*)")
        if path:
            self._mmproj_edit.setText(path)

    def _browse_detector_model(self) -> None:
        start = os.path.dirname(self._detector_model_edit.text()) or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", start, "Model Files (*.pt *.onnx);;All Files (*)")
        if path:
            self._detector_model_edit.setText(path)
