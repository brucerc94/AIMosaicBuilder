"""Organized settings and workflow panel for AI Mosaic Builder."""
from __future__ import annotations

import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from engine.models import AppSettings


class SettingsPanel(QWidget):
    """Compact workflow-first configuration panel.

    Actions stay visible at the top; detailed settings are grouped into tabs.
    All controls update the shared AppSettings object and remain persistent.
    """

    settings_changed = Signal(object)
    open_folder_requested = Signal()
    exclude_folders_requested = Signal()
    analyze_requested = Signal()
    stop_requested = Signal()
    generate_requested = Signal()
    preview_requested = Signal()
    export_mosaic_requested = Signal()

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._building = True
        self._build_ui()
        self._populate(settings)
        self._building = False

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        workflow = QGroupBox("Workflow")
        workflow_layout = QGridLayout(workflow)
        workflow_layout.setContentsMargins(8, 8, 8, 8)
        workflow_layout.setHorizontalSpacing(6)
        workflow_layout.setVerticalSpacing(6)

        self._btn_open_folder = QPushButton("📂  Open Folder")
        self._btn_exclude_folders = QPushButton("🚫  Exclude Folders")
        self._btn_analyze = QPushButton("▶  Analyze")
        self._btn_stop = QPushButton("■  Stop")
        self._btn_generate = QPushButton("🖼  Generate Mosaic")
        self._btn_preview = QPushButton("👁  Preview")
        self._btn_export_mosaic = QPushButton("💾  Export Mosaic")

        self._btn_analyze.setObjectName("primary_btn")
        self._btn_stop.setObjectName("danger_btn")
        self._btn_generate.setObjectName("success_btn")
        self._btn_export_mosaic.setObjectName("success_btn")

        self._btn_exclude_folders.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_generate.setEnabled(False)
        self._btn_preview.setEnabled(False)
        self._btn_export_mosaic.setEnabled(False)

        self._btn_open_folder.clicked.connect(self.open_folder_requested.emit)
        self._btn_exclude_folders.clicked.connect(self.exclude_folders_requested.emit)
        self._btn_analyze.clicked.connect(self.analyze_requested.emit)
        self._btn_stop.clicked.connect(self.stop_requested.emit)
        self._btn_generate.clicked.connect(self.generate_requested.emit)
        self._btn_preview.clicked.connect(self.preview_requested.emit)
        self._btn_export_mosaic.clicked.connect(self.export_mosaic_requested.emit)

        workflow_layout.addWidget(self._btn_open_folder, 0, 0)
        workflow_layout.addWidget(self._btn_exclude_folders, 0, 1)
        workflow_layout.addWidget(self._btn_analyze, 1, 0)
        workflow_layout.addWidget(self._btn_stop, 1, 1)
        workflow_layout.addWidget(self._btn_generate, 2, 0, 1, 2)
        workflow_layout.addWidget(self._btn_preview, 3, 0)
        workflow_layout.addWidget(self._btn_export_mosaic, 3, 1)
        outer.addWidget(workflow)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_project_tab(), "Project")
        self._tabs.addTab(self._build_analysis_tab(), "Analysis")
        self._tabs.addTab(self._build_mosaic_tab(), "Mosaic")
        self._tabs.addTab(self._build_filters_tab(), "Filters")
        outer.addWidget(self._tabs, 1)

    def _scroll_tab(self, content: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _tab_container(self) -> tuple[QWidget, QVBoxLayout]:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(2, 6, 2, 6)
        layout.setSpacing(8)
        return widget, layout

    def _build_project_tab(self) -> QWidget:
        content, layout = self._tab_container()
        source = QGroupBox("Source")
        source_form = QFormLayout(source)
        self._folder_edit = QLineEdit()
        self._folder_edit.setReadOnly(True)
        self._folder_edit.setPlaceholderText("Open an image folder…")
        source_form.addRow("Folder:", self._folder_edit)
        layout.addWidget(source)

        model = QGroupBox("Vision Model")
        model_form = QFormLayout(model)
        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText("Path to .gguf model…")
        model_form.addRow("Model:", self._browse_row(self._model_edit, self._browse_model))
        self._mmproj_edit = QLineEdit()
        self._mmproj_edit.setPlaceholderText("mmproj .gguf…")
        model_form.addRow("mmproj:", self._browse_row(self._mmproj_edit, self._browse_mmproj))
        layout.addWidget(model)
        layout.addStretch(1)
        return self._scroll_tab(content)

    def _build_analysis_tab(self) -> QWidget:
        content, layout = self._tab_container()
        inference = QGroupBox("Vision Inference")
        form = QFormLayout(inference)
        self._gpu_spin = QSpinBox(); self._gpu_spin.setRange(-1, 999); self._gpu_spin.setSuffix(" layers"); self._gpu_spin.setToolTip("-1 = all layers, 0 = CPU only")
        self._ctx_spin = QSpinBox(); self._ctx_spin.setRange(2048, 131072); self._ctx_spin.setSingleStep(512); self._ctx_spin.setSuffix(" tokens")
        self._max_tokens_spin = QSpinBox(); self._max_tokens_spin.setRange(64, 4096); self._max_tokens_spin.setSingleStep(32); self._max_tokens_spin.setSuffix(" tokens")
        self._threads_spin = QSpinBox(); self._threads_spin.setRange(1, 64); self._threads_spin.setSuffix(" threads")
        self._batch_threads_spin = QSpinBox(); self._batch_threads_spin.setRange(0, 64); self._batch_threads_spin.setSuffix(" threads")
        self._n_batch_spin = QSpinBox(); self._n_batch_spin.setRange(64, 4096); self._n_batch_spin.setSingleStep(64)
        self._n_ubatch_spin = QSpinBox(); self._n_ubatch_spin.setRange(64, 4096); self._n_ubatch_spin.setSingleStep(64)
        self._ctx_spin.setToolTip("Context window. 4096 is the recommended default for Gemma-4 vision.")
        self._max_tokens_spin.setToolTip("Maximum generated tokens for the analysis JSON. 576 is the recommended default.")
        self._n_ubatch_spin.setToolTip("CUDA micro-batch. Must be ≤ n_batch.")
        form.addRow("GPU Layers:", self._gpu_spin)
        form.addRow("Context:", self._ctx_spin)
        form.addRow("Max Tokens:", self._max_tokens_spin)
        form.addRow("CPU Threads:", self._threads_spin)
        form.addRow("Batch Threads:", self._batch_threads_spin)
        form.addRow("n_batch:", self._n_batch_spin)
        form.addRow("n_ubatch:", self._n_ubatch_spin)
        layout.addWidget(inference)

        detector = QGroupBox("Person Detector")
        dform = QFormLayout(detector)
        self._detector_model_edit = QLineEdit(); self._detector_model_edit.setPlaceholderText("YOLO model .pt / .onnx")
        dform.addRow("YOLO Model:", self._browse_row(self._detector_model_edit, self._browse_detector_model))
        self._detector_conf_spin = QDoubleSpinBox(); self._detector_conf_spin.setRange(0.05, 0.95); self._detector_conf_spin.setSingleStep(0.05); self._detector_conf_spin.setDecimals(2)
        dform.addRow("Confidence:", self._detector_conf_spin)
        layout.addWidget(detector)
        layout.addStretch(1)
        return self._scroll_tab(content)

    def _build_mosaic_tab(self) -> QWidget:
        content, layout = self._tab_container()
        mosaic = QGroupBox("Layout")
        form = QFormLayout(mosaic)
        self._target_spin = QSpinBox(); self._target_spin.setRange(0, 200); self._target_spin.setSuffix(" images"); self._target_spin.setToolTip("0 = Automatic. Any positive number is allowed.")
        self._padding_spin = QSpinBox(); self._padding_spin.setRange(0, 500); self._padding_spin.setSuffix(" px")
        self._canvas_w_spin = QSpinBox(); self._canvas_w_spin.setRange(800, 16000); self._canvas_w_spin.setSingleStep(240); self._canvas_w_spin.setSuffix(" px")
        self._canvas_h_spin = QSpinBox(); self._canvas_h_spin.setRange(600, 12000); self._canvas_h_spin.setSingleStep(240); self._canvas_h_spin.setSuffix(" px")
        self._min_subject_percent_spin = QDoubleSpinBox(); self._min_subject_percent_spin.setRange(1.0, 50.0); self._min_subject_percent_spin.setSingleStep(1.0); self._min_subject_percent_spin.setDecimals(1); self._min_subject_percent_spin.setSuffix(" % of H")
        self._target_subject_percent_spin = QDoubleSpinBox(); self._target_subject_percent_spin.setRange(1.0, 75.0); self._target_subject_percent_spin.setSingleStep(1.0); self._target_subject_percent_spin.setDecimals(1); self._target_subject_percent_spin.setSuffix(" % of H")
        form.addRow("Target Images:", self._target_spin)
        form.addRow("Crop Padding:", self._padding_spin)
        form.addRow("Canvas Width:", self._canvas_w_spin)
        form.addRow("Canvas Height:", self._canvas_h_spin)
        form.addRow("Min Subject:", self._min_subject_percent_spin)
        form.addRow("Target Subject:", self._target_subject_percent_spin)
        layout.addWidget(mosaic)
        explanation = QLabel("Subject sizes are relative to canvas height. The minimum is a hard lower bound; the target is preferred when choosing the best layout.")
        explanation.setWordWrap(True)
        explanation.setObjectName("section_label")
        layout.addWidget(explanation)
        layout.addStretch(1)
        return self._scroll_tab(content)

    def _build_filters_tab(self) -> QWidget:
        content, layout = self._tab_container()

        request = QGroupBox("AI Custom Request")
        request_layout = QVBoxLayout(request)
        request_hint = QLabel(
            "Optional. This is evaluated by Gemma during Analyze together with the normal photo analysis. "
            "Examples: ‘solo caminando’, ‘todos saltando’, or ‘fotos en la playa’."
        )
        request_hint.setWordWrap(True)
        request_hint.setObjectName("section_label")
        request_layout.addWidget(request_hint)
        self._custom_prompt_edit = QTextEdit()
        self._custom_prompt_edit.setPlaceholderText("Describe what you want the selected photos to show…")
        self._custom_prompt_edit.setMinimumHeight(76)
        self._custom_prompt_edit.setMaximumHeight(120)
        request_layout.addWidget(self._custom_prompt_edit)
        weight_form = QFormLayout()
        self._custom_prompt_weight_spin = QDoubleSpinBox()
        self._custom_prompt_weight_spin.setRange(0.0, 100.0)
        self._custom_prompt_weight_spin.setSingleStep(5.0)
        self._custom_prompt_weight_spin.setDecimals(0)
        self._custom_prompt_weight_spin.setSuffix(" %")
        self._custom_prompt_weight_spin.setToolTip("How strongly the custom request affects the final ranking. Default: 30%.")
        weight_form.addRow("Request Weight:", self._custom_prompt_weight_spin)
        request_layout.addLayout(weight_form)
        layout.addWidget(request)

        req = QGroupBox("Required Composition")
        form = QFormLayout(req)
        self._req_face_only = self._count_spin(); self._req_full_body = self._count_spin()
        self._req_front = self._count_spin(); self._req_side = self._count_spin(); self._req_back = self._count_spin()
        self._req_male = self._count_spin(); self._req_female = self._count_spin()
        self._req_face_visible = self._count_spin(); self._req_body_visible = self._count_spin()
        form.addRow("Face only:", self._req_face_only)
        form.addRow("Full body:", self._req_full_body)
        form.addRow("Front:", self._req_front)
        form.addRow("Side:", self._req_side)
        form.addRow("Back:", self._req_back)
        form.addRow("Male:", self._req_male)
        form.addRow("Female:", self._req_female)
        form.addRow("Face visible:", self._req_face_visible)
        form.addRow("Body visible:", self._req_body_visible)
        layout.addWidget(req)

        quality = QGroupBox("Quality / Content")
        qbox = QVBoxLayout(quality)
        content_row = QHBoxLayout()
        content_row.addWidget(QLabel("Content:"))
        self._req_nsfw = QComboBox(); self._req_nsfw.addItem("Don't care", "any"); self._req_nsfw.addItem("Safe only", "safe_only"); self._req_nsfw.addItem("NSFW only", "nsfw_only")
        content_row.addWidget(self._req_nsfw, 1)
        qbox.addLayout(content_row)
        self._req_exclude_blurry = QCheckBox("Exclude blurry")
        self._req_exclude_occluded = QCheckBox("Exclude heavily occluded")
        qbox.addWidget(self._req_exclude_blurry)
        qbox.addWidget(self._req_exclude_occluded)
        qform = QFormLayout()
        self._req_min_quality = QDoubleSpinBox(); self._req_min_quality.setRange(0, 1); self._req_min_quality.setSingleStep(0.05); self._req_min_quality.setDecimals(2)
        self._req_min_visibility = QDoubleSpinBox(); self._req_min_visibility.setRange(0, 1); self._req_min_visibility.setSingleStep(0.05); self._req_min_visibility.setDecimals(2)
        qform.addRow("Min image quality:", self._req_min_quality)
        qform.addRow("Min visibility:", self._req_min_visibility)
        qbox.addLayout(qform)
        layout.addWidget(quality)

        prefs = QGroupBox("Preferences")
        pbox = QVBoxLayout(prefs)
        self._pref_face_only = QCheckBox("Prefer face-only")
        self._pref_full_body = QCheckBox("Prefer full-body")
        self._pref_front = QCheckBox("Prefer front")
        self._pref_side = QCheckBox("Prefer side")
        self._pref_back = QCheckBox("Prefer back")
        self._pref_solo = QCheckBox("Prefer solo person")
        self._pref_face_visible = QCheckBox("Prefer face visible")
        self._pref_body_visible = QCheckBox("Prefer body visible")
        for item in (self._pref_face_only, self._pref_full_body, self._pref_front, self._pref_side, self._pref_back, self._pref_solo, self._pref_face_visible, self._pref_body_visible):
            pbox.addWidget(item)
        layout.addWidget(prefs)

        thresholds = QGroupBox("Image / Similarity")
        tform = QFormLayout(thresholds)
        self._min_w_spin = QSpinBox(); self._min_w_spin.setRange(50, 2000); self._min_w_spin.setSuffix(" px")
        self._min_h_spin = QSpinBox(); self._min_h_spin.setRange(50, 2000); self._min_h_spin.setSuffix(" px")
        self._similarity_spin = QSpinBox(); self._similarity_spin.setRange(0, 64)
        tform.addRow("Min Width:", self._min_w_spin)
        tform.addRow("Min Height:", self._min_h_spin)
        tform.addRow("Similarity:", self._similarity_spin)
        layout.addWidget(thresholds)
        layout.addStretch(1)
        return self._scroll_tab(content)

    @staticmethod
    def _count_spin() -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, 99)
        return spin

    @staticmethod
    def _browse_row(edit: QLineEdit, callback) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(edit, 1)
        button = QPushButton("…")
        button.setFixedWidth(30)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return row

    def _connect_settings_widget(self, widget: QWidget) -> None:
        if isinstance(widget, QLineEdit):
            widget.textChanged.connect(self._on_changed)
        elif isinstance(widget, QTextEdit):
            widget.textChanged.connect(self._on_changed)
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(self._on_changed)
        elif isinstance(widget, QCheckBox):
            widget.stateChanged.connect(self._on_changed)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(self._on_changed)

    def _all_settings_widgets(self) -> list[QWidget]:
        return [
            self._model_edit, self._mmproj_edit,
            self._gpu_spin, self._ctx_spin, self._max_tokens_spin, self._threads_spin,
            self._batch_threads_spin, self._n_batch_spin, self._n_ubatch_spin,
            self._detector_model_edit, self._detector_conf_spin,
            self._target_spin, self._padding_spin, self._canvas_w_spin, self._canvas_h_spin,
            self._min_subject_percent_spin, self._target_subject_percent_spin,
            self._custom_prompt_edit, self._custom_prompt_weight_spin,
            self._req_face_only, self._req_full_body, self._req_front, self._req_side,
            self._req_back, self._req_male, self._req_female, self._req_face_visible,
            self._req_body_visible, self._req_nsfw, self._req_exclude_blurry,
            self._req_exclude_occluded, self._req_min_quality, self._req_min_visibility,
            self._pref_face_only, self._pref_full_body, self._pref_front, self._pref_side,
            self._pref_back, self._pref_solo, self._pref_face_visible, self._pref_body_visible,
            self._min_w_spin, self._min_h_spin, self._similarity_spin,
        ]

    def _populate(self, s: AppSettings) -> None:
        self._folder_edit.setText(s.last_source_folder)
        self._btn_exclude_folders.setEnabled(bool(s.last_source_folder and os.path.isdir(s.last_source_folder)))
        self._model_edit.setText(s.model_path)
        self._mmproj_edit.setText(s.mmproj_path)
        self._gpu_spin.setValue(s.n_gpu_layers)
        self._ctx_spin.setValue(max(2048, int(s.n_ctx)))
        self._max_tokens_spin.setValue(max(64, min(4096, int(s.max_tokens))))
        self._threads_spin.setValue(s.n_threads)
        self._batch_threads_spin.setValue(s.n_threads_batch)
        self._n_batch_spin.setValue(s.n_batch)
        self._n_ubatch_spin.setValue(min(s.n_ubatch, s.n_batch))
        self._detector_model_edit.setText(s.person_detector_model)
        self._detector_conf_spin.setValue(s.person_detector_confidence)
        self._target_spin.setValue(max(0, int(s.target_images)))
        self._padding_spin.setValue(s.padding_px)
        self._canvas_w_spin.setValue(s.canvas_width)
        self._canvas_h_spin.setValue(s.canvas_height)
        self._min_subject_percent_spin.setValue(getattr(s, "min_subject_percent", 10.0))
        self._target_subject_percent_spin.setValue(max(self._min_subject_percent_spin.value(), getattr(s, "target_subject_percent", 15.0)))
        self._min_w_spin.setValue(s.min_image_width)
        self._min_h_spin.setValue(s.min_image_height)
        self._similarity_spin.setValue(s.phash_threshold)
        self._custom_prompt_edit.setPlainText(str(getattr(s, "custom_prompt", "")))
        self._custom_prompt_weight_spin.setValue(max(0.0, min(100.0, float(getattr(s, "custom_prompt_weight", 30.0)))))

        r = s.mosaic_requirements
        for widget, value in (
            (self._req_face_only, r.min_face_only), (self._req_full_body, r.min_full_body),
            (self._req_front, r.min_front), (self._req_side, r.min_side), (self._req_back, r.min_back),
            (self._req_male, r.min_male), (self._req_female, r.min_female),
            (self._req_face_visible, r.min_face_visible), (self._req_body_visible, r.min_body_visible),
        ):
            widget.setValue(value)
        idx = self._req_nsfw.findData(r.nsfw_policy)
        self._req_nsfw.setCurrentIndex(max(0, idx))
        self._req_exclude_blurry.setChecked(r.exclude_blurry)
        self._req_exclude_occluded.setChecked(r.exclude_occluded)
        self._req_min_quality.setValue(r.min_quality)
        self._req_min_visibility.setValue(r.min_person_visibility)
        for widget, value in (
            (self._pref_face_only, r.prefer_face_only), (self._pref_full_body, r.prefer_full_body),
            (self._pref_front, r.prefer_front), (self._pref_side, r.prefer_side),
            (self._pref_back, r.prefer_back), (self._pref_solo, r.prefer_solo),
            (self._pref_face_visible, r.prefer_face_visible), (self._pref_body_visible, r.prefer_body_visible),
        ):
            widget.setChecked(value)

        for widget in self._all_settings_widgets():
            self._connect_settings_widget(widget)

    def current_settings(self) -> AppSettings:
        s = self._settings
        s.last_source_folder = self._folder_edit.text().strip()
        s.model_path = self._model_edit.text().strip()
        s.mmproj_path = self._mmproj_edit.text().strip()
        s.n_gpu_layers = self._gpu_spin.value()
        s.n_ctx = max(2048, self._ctx_spin.value())
        s.max_tokens = self._max_tokens_spin.value()
        s.n_threads = self._threads_spin.value()
        s.n_threads_batch = self._batch_threads_spin.value()
        s.n_batch = self._n_batch_spin.value()
        s.n_ubatch = min(self._n_ubatch_spin.value(), self._n_batch_spin.value())
        s.person_detector_model = self._detector_model_edit.text().strip()
        s.person_detector_confidence = self._detector_conf_spin.value()
        s.target_images = self._target_spin.value()
        s.padding_px = self._padding_spin.value()
        s.canvas_width = self._canvas_w_spin.value()
        s.canvas_height = self._canvas_h_spin.value()
        s.min_subject_percent = self._min_subject_percent_spin.value()
        s.target_subject_percent = max(s.min_subject_percent, self._target_subject_percent_spin.value())
        s.min_image_width = self._min_w_spin.value()
        s.min_image_height = self._min_h_spin.value()
        s.phash_threshold = self._similarity_spin.value()
        s.custom_prompt = self._custom_prompt_edit.toPlainText().strip()[:2000]
        s.custom_prompt_weight = max(0.0, min(100.0, self._custom_prompt_weight_spin.value()))

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

    def _on_changed(self, *_args) -> None:
        if not self._building:
            self.settings_changed.emit(self.current_settings())

    def set_source_folder(self, folder: str) -> None:
        self._building = True
        self._folder_edit.setText(folder)
        self._building = False
        self._btn_exclude_folders.setEnabled(bool(folder and os.path.isdir(folder)))

    def set_analyzing(self, active: bool) -> None:
        self._btn_open_folder.setEnabled(not active)
        self._btn_exclude_folders.setEnabled(not active and bool(self._folder_edit.text().strip()) and os.path.isdir(self._folder_edit.text().strip()))
        self._btn_analyze.setEnabled(not active)
        self._btn_stop.setEnabled(active)
        if active:
            self._btn_generate.setEnabled(False)
            self._btn_preview.setEnabled(False)
            self._btn_export_mosaic.setEnabled(False)

    def set_exclude_folders_enabled(self, enabled: bool) -> None:
        self._btn_exclude_folders.setEnabled(bool(enabled) and bool(self._folder_edit.text().strip()) and os.path.isdir(self._folder_edit.text().strip()))

    def set_generate_enabled(self, enabled: bool) -> None:
        self._btn_generate.setEnabled(bool(enabled))

    def set_preview_enabled(self, enabled: bool) -> None:
        self._btn_preview.setEnabled(bool(enabled))

    def set_export_mosaic_enabled(self, enabled: bool) -> None:
        self._btn_export_mosaic.setEnabled(bool(enabled))

    def _browse_model(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        start = os.path.dirname(self._model_edit.text()) or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF Model", start, "GGUF Models (*.gguf);;All Files (*)")
        if path:
            self._model_edit.setText(path)

    def _browse_mmproj(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        start = os.path.dirname(self._mmproj_edit.text()) or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Select mmproj File", start, "GGUF Files (*.gguf);;All Files (*)")
        if path:
            self._mmproj_edit.setText(path)

    def _browse_detector_model(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        start = os.path.dirname(self._detector_model_edit.text()) or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Select Person Detector", start, "Model Files (*.pt *.onnx);;All Files (*)")
        if path:
            self._detector_model_edit.setText(path)
