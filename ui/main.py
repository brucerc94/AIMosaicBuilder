"""Main PySide6 window for AI Mosaic Builder."""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QLabel, QMainWindow, QMessageBox, QSplitter, QVBoxLayout, QWidget

from engine.cache import get_cache, source_cache_dir
from engine.image_analyzer import build_record, discover_images
from engine.models import ImageRecord, ImageStatus
from engine.project_export import export_project
from engine.session import save_session
from engine.storage import load_settings, save_settings
from engine.vision_llm import get_vision_engine
from ui.image_detail import ImageDetailPanel
from ui.image_grid import ImageGrid
from ui.settings import SettingsPanel
from ui.styles import DARK_STYLESHEET
from ui.workers import AnalysisWorker, ModelLoaderWorker, PostProcessWorker

logger = logging.getLogger("ui.main")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Mosaic Builder")
        self.resize(1500, 900)

        self._settings = load_settings()
        self._engine = get_vision_engine()
        self._cache = self._get_cache_for_source(self._settings.last_source_folder)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)

        self._records: list[ImageRecord] = []
        self._paths: list[str] = []
        self._loader = None
        self._analysis_worker = None
        self._post_worker = None

        self._build_ui()
        self._connect()
        self._update_summary()

    @staticmethod
    def _get_cache_for_source(source_folder: str) :
        """Get the analysis cache owned by the selected source folder."""
        if source_folder and Path(source_folder).is_dir():
            return get_cache(str(source_cache_dir(source_folder)))
        return get_cache("")

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        title = QLabel("AI Mosaic Builder")
        title.setObjectName("title_label")
        layout.addWidget(title)

        splitter = QSplitter()
        self.settings_panel = SettingsPanel(self._settings)
        self.grid = ImageGrid()
        self.details = ImageDetailPanel()
        splitter.addWidget(self.settings_panel)
        splitter.addWidget(self.grid)
        splitter.addWidget(self.details)
        splitter.setSizes([300, 820, 380])
        layout.addWidget(splitter, 1)

        self._summary = QLabel("Ready")
        self._summary.setObjectName("section_label")
        layout.addWidget(self._summary)
        self.setCentralWidget(root)
        self.setStyleSheet(DARK_STYLESHEET)

    def _connect(self) -> None:
        self.settings_panel.settings_changed.connect(self._settings_changed)
        self.settings_panel.analyze_requested.connect(self._start_analysis)
        self.settings_panel.stop_requested.connect(self._stop)
        self.settings_panel.generate_requested.connect(self._generate_project)
        self.grid.image_selected.connect(self.details.show_record)
        self.grid.include_toggled.connect(self._toggle_include)
        self.grid.exclude_toggled.connect(self._toggle_exclude)
        self.details.include_toggled.connect(self._toggle_include)
        self.details.exclude_toggled.connect(self._toggle_exclude)

    def _settings_changed(self, settings) -> None:
        self._settings = settings
        if settings.last_source_folder:
            self._cache = self._get_cache_for_source(settings.last_source_folder)
        save_settings(settings)

    def _start_analysis(self) -> None:
        if self._loader or self._analysis_worker or self._post_worker:
            return
        folder = Path(self._settings.last_source_folder)
        model = Path(self._settings.model_path)
        mmproj = Path(self._settings.mmproj_path)
        if not folder.is_dir():
            QMessageBox.warning(self, "Source folder", "Select a valid image folder first.")
            return
        if not model.is_file():
            QMessageBox.warning(self, "Vision model", "Select a valid GGUF vision model.")
            return
        if not mmproj.is_file():
            QMessageBox.warning(self, "Vision projector", "A valid mmproj file is required.")
            return

        # Always switch to the cache owned by the current source folder before
        # discovering/analyzing images. This prevents one source folder from
        # reusing another folder's analysis cache.
        self._cache = self._get_cache_for_source(str(folder))
        self._settings.cache_directory = str(source_cache_dir(folder))
        save_settings(self._settings)

        self._paths = discover_images(str(folder))
        if not self._paths:
            QMessageBox.information(self, "Images", "No supported images were found.")
            return

        self._records = [build_record(path) for path in self._paths]
        self.grid.load_records(self._records)
        self.details.clear()
        self.settings_panel.set_analyzing(True)
        self.settings_panel.set_generate_enabled(False)
        self._summary.setText(f"Loading vision model… {len(self._paths)} images")

        worker = ModelLoaderWorker(self._engine, str(model), str(mmproj), self._settings)
        worker.signals.progress.connect(self._summary.setText)
        worker.signals.finished.connect(self._on_model_loaded)
        worker.signals.error.connect(self._on_worker_error)
        self._loader = worker
        self._pool.start(worker)

    def _on_model_loaded(self) -> None:
        self._loader = None
        worker = AnalysisWorker(self._engine, self._cache, self._settings, self._paths)
        worker.signals.progress.connect(self._on_analysis_progress)
        worker.signals.finished.connect(self._on_analysis_finished)
        worker.signals.error.connect(self._on_worker_error)
        self._analysis_worker = worker
        self._pool.start(worker)

    def _on_analysis_progress(self, done: int, total: int, record: ImageRecord) -> None:
        self._records = [record if item.path == record.path else item for item in self._records]
        self.grid.upsert_record(record)
        self._summary.setText(f"Analyzing {done}/{total} — {record.filename}")
        if self._settings.last_source_folder:
            save_session(self._settings.last_source_folder, self._records)

    def _on_analysis_finished(self, records: list[ImageRecord]) -> None:
        self._analysis_worker = None
        self._records = records
        self.grid.load_records(records)
        self._summary.setText("Detecting people and ranking…")
        worker = PostProcessWorker(records, self._settings)
        worker.signals.progress.connect(
            lambda done, total, name: self._summary.setText(f"Detecting people {done}/{total} — {name}")
        )
        worker.signals.finished.connect(self._on_post_finished)
        worker.signals.error.connect(self._on_worker_error)
        self._post_worker = worker
        self._pool.start(worker)

    def _on_post_finished(self, records: list[ImageRecord]) -> None:
        self._post_worker = None
        self._records = records
        self.grid.load_records(records)
        self.settings_panel.set_analyzing(False)
        self.settings_panel.set_generate_enabled(bool(self._selected_records()))
        if self._settings.last_source_folder:
            save_session(self._settings.last_source_folder, records)
        self._update_summary()

    def _selected_records(self) -> list[ImageRecord]:
        return [r for r in self._records if r.status == ImageStatus.SELECTED]

    def _update_summary(self) -> None:
        people = sum(1 for r in self._records if r.analysis and r.analysis.has_person)
        selected = len(self._selected_records())
        errors = sum(1 for r in self._records if r.status == ImageStatus.ERROR)
        self._summary.setText(f"{len(self._records)} images | people: {people} | selected: {selected} | errors: {errors}")

    def _stop(self) -> None:
        if self._analysis_worker:
            self._analysis_worker.cancel()
            self._summary.setText("Stopping after the current image…")
        if self._post_worker:
            self._post_worker.cancel()

    def _toggle_include(self, record: ImageRecord) -> None:
        record.manually_included = not record.manually_included
        if record.manually_included:
            record.manually_excluded = False
        self._rerun_postprocess()
        self.details.show_record(record)

    def _toggle_exclude(self, record: ImageRecord) -> None:
        record.manually_excluded = not record.manually_excluded
        if record.manually_excluded:
            record.manually_included = False
        self._rerun_postprocess()
        self.details.show_record(record)

    def _rerun_postprocess(self) -> None:
        if not self._records or self._analysis_worker or self._post_worker:
            return
        worker = PostProcessWorker(self._records, self._settings)
        worker.signals.progress.connect(
            lambda done, total, name: self._summary.setText(f"Updating selection {done}/{total} — {name}")
        )
        worker.signals.finished.connect(self._on_post_finished)
        worker.signals.error.connect(self._on_worker_error)
        self._post_worker = worker
        self.settings_panel.set_generate_enabled(False)
        self._pool.start(worker)

    def _generate_project(self) -> None:
        selected = self._selected_records()
        if not selected:
            return

        source_folder = Path(self._settings.last_source_folder)
        try:
            path = export_project(
                str(source_folder),
                selected,
                (self._settings.canvas_width, self._settings.canvas_height),
                self._settings.padding_px,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Generate failed", str(exc))
            return
        QMessageBox.information(self, "Project generated", f"Saved to:\n{path}")

    def _on_worker_error(self, message: str) -> None:
        self._loader = None
        self._analysis_worker = None
        self._post_worker = None
        self.settings_panel.set_analyzing(False)
        self.settings_panel.set_generate_enabled(False)
        self._summary.setText("Error")
        QMessageBox.critical(self, "AI Mosaic Builder", message)

    def closeEvent(self, event) -> None:
        save_settings(self._settings)
        if self._settings.last_source_folder and self._records:
            save_session(self._settings.last_source_folder, self._records)
        self._engine.unload()
        event.accept()
