"""Main PySide6 window for AI Mosaic Builder."""
from __future__ import annotations

import copy
import logging
from pathlib import Path

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QFileDialog, QLabel, QMainWindow, QMessageBox, QSplitter, QVBoxLayout, QWidget

from engine.cache import cache_model_key, get_cache, source_cache_dir
from engine.image_analyzer import build_record, discover_images
from engine.models import ImageRecord, ImageStatus
from engine.ranking import rank_records
from engine.runtime_config import configure_vision_engine
from engine.session import save_session
from engine.storage import load_settings, save_settings
from engine.vision_llm import get_vision_engine
from ui.image_detail import ImageDetailPanel
from ui.image_grid import ImageGrid
from ui.mosaic_export_worker import MosaicExportWorker
from ui.preview import PreviewWindow
from ui.settings_extended import SettingsPanel
from ui.styles import DARK_STYLESHEET
from ui.workers import AnalysisWorker, GenerateMosaicWorker, ModelLoaderWorker, PostProcessWorker

logger = logging.getLogger("ui.main")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Mosaic Builder")
        self.resize(1500, 900)

        self._settings = load_settings()
        self._engine = get_vision_engine()
        configure_vision_engine(self._engine, self._settings)
        self._cache = self._get_cache_for_source(self._settings.last_source_folder)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)

        self._records: list[ImageRecord] = []
        self._paths: list[str] = []
        self._loader = None
        self._analysis_worker = None
        self._post_worker = None
        self._generate_worker = None
        self._export_worker = None
        self._preview_window: PreviewWindow | None = None

        self._build_ui()
        self._connect()
        self._update_summary()

    @staticmethod
    def _get_cache_for_source(source_folder: str):
        """Get the analysis cache owned by the selected source/project folder."""
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
        self.settings_panel.open_folder_requested.connect(self._open_folder)
        self.settings_panel.analyze_requested.connect(self._start_analysis)
        self.settings_panel.stop_requested.connect(self._stop)
        self.settings_panel.generate_requested.connect(self._generate_project)
        self.settings_panel.preview_requested.connect(self._show_preview)
        self.settings_panel.export_mosaic_requested.connect(self._export_mosaic)
        self.grid.image_selected.connect(self.details.show_record)
        self.grid.include_toggled.connect(self._toggle_include)
        self.grid.exclude_toggled.connect(self._toggle_exclude)
        self.details.include_toggled.connect(self._toggle_include)
        self.details.exclude_toggled.connect(self._toggle_exclude)

    def _settings_changed(self, settings) -> None:
        self._settings = settings
        configure_vision_engine(self._engine, self._settings)
        if settings.last_source_folder:
            self._cache = self._get_cache_for_source(settings.last_source_folder)
        save_settings(settings)

    def _open_folder(self) -> None:
        """Load a source folder and cached analysis without loading/invoking the model."""
        if self._loader or self._analysis_worker or self._post_worker or self._generate_worker or self._export_worker:
            return

        folder = QFileDialog.getExistingDirectory(
            self,
            "Open Image Folder",
            self._settings.last_source_folder or str(Path.home()),
        )
        if not folder:
            return
        self._load_source_folder(folder)

    def _load_source_folder(self, folder: str) -> None:
        source = Path(folder).expanduser().resolve()
        if not source.is_dir():
            QMessageBox.warning(self, "Source folder", "The selected folder is not valid.")
            return

        self._settings.last_source_folder = str(source)
        self._settings.cache_directory = str(source_cache_dir(source))
        save_settings(self._settings)
        self.settings_panel.set_source_folder(str(source))

        self._cache = self._get_cache_for_source(str(source))
        self._paths = discover_images(str(source))
        if not self._paths:
            self._records = []
            self.grid.clear()
            self.details.clear()
            self.settings_panel.set_generate_enabled(False)
            self.settings_panel.set_preview_enabled(False)
            self.settings_panel.set_export_mosaic_enabled(False)
            self._summary.setText(f"No supported images found in {source}")
            return

        model_name = Path(self._settings.model_path).name if self._settings.model_path else ""
        custom_prompt = str(getattr(self._settings, "custom_prompt", "") or "")
        cache_model = cache_model_key(model_name, custom_prompt) if model_name else ""
        records: list[ImageRecord] = []
        cache_hits = 0
        for path in self._paths:
            p = Path(path)
            try:
                file_size = p.stat().st_size
            except OSError:
                file_size = 0
            record = ImageRecord(
                path=str(p),
                filename=p.name,
                file_size=file_size,
                status=ImageStatus.PENDING,
            )
            cached = self._cache.get_by_path(str(p), model=cache_model)
            if cached is not None:
                record.analysis = cached
                record.status = ImageStatus.CACHED
                cache_hits += 1
            records.append(record)

        self._records = records
        self.grid.load_records(records)
        self.details.clear()
        self.settings_panel.set_analyzing(False)
        self.settings_panel.set_generate_enabled(False)
        self.settings_panel.set_preview_enabled(False)
        self.settings_panel.set_export_mosaic_enabled(False)
        prompt_state = "custom request" if custom_prompt.strip() else "standard analysis"
        self._summary.setText(
            f"{len(records)} images loaded | cache: {cache_hits}/{len(records)} | {prompt_state} | Ready to Analyze"
        )
        logger.info(
            "[source] Open Folder | folder=%s | images=%d | cache_hits=%d | model_loaded=%s | custom_request=%s",
            source, len(records), cache_hits, self._engine.model_loaded,
            "yes" if custom_prompt.strip() else "no",
        )

    def _start_analysis(self) -> None:
        if self._loader or self._analysis_worker or self._post_worker or self._generate_worker or self._export_worker:
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

        self._cache = self._get_cache_for_source(str(folder))
        self._settings.cache_directory = str(source_cache_dir(folder))
        save_settings(self._settings)

        if not self._paths:
            self._paths = discover_images(str(folder))
        if not self._paths:
            QMessageBox.information(self, "Images", "No supported images were found.")
            return

        self._records = [build_record(path) for path in self._paths]
        self.grid.load_records(self._records)
        self.details.clear()
        self.settings_panel.set_analyzing(True)
        self.settings_panel.set_generate_enabled(False)
        self.settings_panel.set_preview_enabled(False)
        self.settings_panel.set_export_mosaic_enabled(False)
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
        self.settings_panel.set_generate_enabled(bool(self._layout_candidates()))
        self.settings_panel.set_preview_enabled(False)
        self.settings_panel.set_export_mosaic_enabled(False)
        if self._settings.last_source_folder:
            save_session(self._settings.last_source_folder, records)
        self._update_summary()

    def _selected_records(self) -> list[ImageRecord]:
        return [r for r in self._records if r.status == ImageStatus.SELECTED]

    def _layout_candidates(self) -> list[ImageRecord]:
        return [
            r for r in self._records
            if r.analysis
            and r.analysis.has_person
            and r.detections
            and r.ranking
            and r.ranking.final_score > 0
            and not r.manually_excluded
        ]

    def _update_summary(self) -> None:
        people = sum(1 for r in self._records if r.analysis and r.analysis.has_person)
        selected = len(self._selected_records())
        candidates = len(self._layout_candidates())
        errors = sum(1 for r in self._records if r.status == ImageStatus.ERROR)
        self._summary.setText(
            f"{len(self._records)} images | people: {people} | candidates: {candidates} | selected: {selected} | errors: {errors}"
        )

    def _stop(self) -> None:
        if self._analysis_worker:
            self._analysis_worker.cancel()
            self._summary.setText("Stopping after the current image…")
        if self._post_worker:
            self._post_worker.cancel()
        if self._generate_worker:
            self._generate_worker.cancel()
            self._summary.setText("Stopping mosaic generation…")
        if self._export_worker:
            self._export_worker.cancel()
            self._summary.setText("Stopping mosaic export…")

    def _toggle_include(self, record: ImageRecord) -> None:
        if self._generate_worker or self._export_worker:
            return
        record.manually_included = not record.manually_included
        if record.manually_included:
            record.manually_excluded = False
        self.settings_panel.set_preview_enabled(False)
        self.settings_panel.set_export_mosaic_enabled(False)
        self._rerank_without_detection()
        self.details.show_record(record)

    def _toggle_exclude(self, record: ImageRecord) -> None:
        if self._generate_worker or self._export_worker:
            return
        record.manually_excluded = not record.manually_excluded
        if record.manually_excluded:
            record.manually_included = False
        self.settings_panel.set_preview_enabled(False)
        self.settings_panel.set_export_mosaic_enabled(False)
        self._rerank_without_detection()
        self.details.show_record(record)

    def _rerank_without_detection(self) -> None:
        if not self._records or self._generate_worker or self._export_worker:
            return
        custom_prompt = str(getattr(self._settings, "custom_prompt", "") or "").strip()
        custom_weight = max(
            0.0,
            min(1.0, float(getattr(self._settings, "custom_prompt_weight", 30.0)) / 100.0),
        ) if custom_prompt else 0.0
        self._records = rank_records(
            self._records,
            self._settings.ranking_weights,
            self._settings.mosaic_requirements,
            user_request_weight=custom_weight,
        )
        self.grid.load_records(self._records)
        self.settings_panel.set_generate_enabled(bool(self._layout_candidates()))
        if self._settings.last_source_folder:
            save_session(self._settings.last_source_folder, self._records)
        self._update_summary()

    def _generate_project(self) -> None:
        if self._generate_worker or self._loader or self._analysis_worker or self._post_worker or self._export_worker:
            return
        if not self._layout_candidates():
            QMessageBox.warning(self, "Generate Mosaic", "No valid mosaic candidates are available. Run Analyze first.")
            return

        generation_settings = copy.deepcopy(self._settings)
        worker = GenerateMosaicWorker(list(self._records), generation_settings)
        worker.signals.progress.connect(self._summary.setText)
        worker.signals.finished.connect(self._on_generate_finished)
        worker.signals.cancelled.connect(self._on_generate_cancelled)
        worker.signals.error.connect(self._on_generate_error)
        self._generate_worker = worker
        self.settings_panel.set_generate_enabled(False)
        self.settings_panel.set_preview_enabled(False)
        self.settings_panel.set_export_mosaic_enabled(False)
        self._summary.setText("Preparing mosaic recipe…")
        self._pool.start(worker)

    def _on_generate_finished(
        self,
        records: list[ImageRecord],
        path: str,
        mode: str,
        image_count: int,
        canvas_fill: float,
        average_zoom: float,
    ) -> None:
        self._generate_worker = None
        self._records = records
        self.grid.load_records(records)
        self.settings_panel.set_generate_enabled(bool(self._layout_candidates()))
        ready = bool(self._selected_records())
        self.settings_panel.set_preview_enabled(ready)
        self.settings_panel.set_export_mosaic_enabled(ready)
        self._update_summary()
        QMessageBox.information(
            self,
            "Project generated",
            f"Saved to:\n{path}\n\nMode: {mode}\nImages: {image_count}\nCanvas fill: {canvas_fill:.1%}\nAverage zoom: {average_zoom:.3f}",
        )

    def _show_preview(self) -> None:
        selected = self._selected_records()
        if not selected:
            QMessageBox.warning(self, "Preview", "Generate a mosaic first to create a preview.")
            self.settings_panel.set_preview_enabled(False)
            return

        if self._preview_window is not None:
            try:
                self._preview_window.close()
            except Exception:
                pass
            self._preview_window = None

        self._preview_window = PreviewWindow(
            selected,
            canvas_size=(self._settings.canvas_width, self._settings.canvas_height),
            padding_px=self._settings.padding_px,
            parent=self,
        )
        self._preview_window.finished.connect(lambda _result: setattr(self, "_preview_window", None))
        self._preview_window.show()
        self._preview_window.raise_()
        self._preview_window.activateWindow()

    def _export_mosaic(self) -> None:
        """Render the generated mosaic to a final image file."""
        if self._export_worker or self._generate_worker or self._loader or self._analysis_worker or self._post_worker:
            return
        selected = self._selected_records()
        if not selected:
            QMessageBox.warning(self, "Export Mosaic", "Generate a mosaic first.")
            self.settings_panel.set_export_mosaic_enabled(False)
            return

        source = Path(self._settings.last_source_folder) if self._settings.last_source_folder else Path.home()
        default_path = source / "mosaic.png"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Mosaic Image",
            str(default_path),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;WebP Image (*.webp)",
        )
        if not output_path:
            return

        worker = MosaicExportWorker(
            selected,
            canvas_size=(self._settings.canvas_width, self._settings.canvas_height),
            padding_px=self._settings.padding_px,
            output_path=output_path,
        )
        worker.signals.progress.connect(self._summary.setText)
        worker.signals.finished.connect(self._on_export_mosaic_finished)
        worker.signals.cancelled.connect(self._on_export_mosaic_cancelled)
        worker.signals.error.connect(self._on_export_mosaic_error)
        self._export_worker = worker
        self.settings_panel.set_generate_enabled(False)
        self.settings_panel.set_preview_enabled(False)
        self.settings_panel.set_export_mosaic_enabled(False)
        self._summary.setText("Rendering mosaic image…")
        self._pool.start(worker)

    def _on_export_mosaic_finished(self, path: str) -> None:
        self._export_worker = None
        ready = bool(self._selected_records())
        self.settings_panel.set_generate_enabled(bool(self._layout_candidates()))
        self.settings_panel.set_preview_enabled(ready)
        self.settings_panel.set_export_mosaic_enabled(ready)
        self._update_summary()
        QMessageBox.information(self, "Mosaic exported", f"Image saved to:\n{path}")

    def _on_export_mosaic_cancelled(self) -> None:
        self._export_worker = None
        ready = bool(self._selected_records())
        self.settings_panel.set_generate_enabled(bool(self._layout_candidates()))
        self.settings_panel.set_preview_enabled(ready)
        self.settings_panel.set_export_mosaic_enabled(ready)
        self._summary.setText("Mosaic export cancelled")

    def _on_export_mosaic_error(self, message: str) -> None:
        self._export_worker = None
        ready = bool(self._selected_records())
        self.settings_panel.set_generate_enabled(bool(self._layout_candidates()))
        self.settings_panel.set_preview_enabled(ready)
        self.settings_panel.set_export_mosaic_enabled(ready)
        self._update_summary()
        QMessageBox.critical(self, "Export Mosaic failed", message)

    def _on_generate_cancelled(self) -> None:
        self._generate_worker = None
        self.settings_panel.set_generate_enabled(bool(self._layout_candidates()))
        self.settings_panel.set_preview_enabled(False)
        self.settings_panel.set_export_mosaic_enabled(False)
        self._update_summary()
        self._summary.setText("Mosaic generation cancelled")

    def _on_generate_error(self, message: str) -> None:
        self._generate_worker = None
        self.settings_panel.set_generate_enabled(bool(self._layout_candidates()))
        self.settings_panel.set_preview_enabled(False)
        self.settings_panel.set_export_mosaic_enabled(False)
        self._update_summary()
        QMessageBox.critical(self, "Generate failed", message)

    def _on_worker_error(self, message: str) -> None:
        self._loader = None
        self._analysis_worker = None
        self._post_worker = None
        self._generate_worker = None
        self._export_worker = None
        self.settings_panel.set_analyzing(False)
        self.settings_panel.set_generate_enabled(False)
        self.settings_panel.set_preview_enabled(False)
        self.settings_panel.set_export_mosaic_enabled(False)
        self._summary.setText("Error")
        QMessageBox.critical(self, "AI Mosaic Builder", message)

    def closeEvent(self, event) -> None:
        if self._preview_window is not None:
            try:
                self._preview_window.close()
            except Exception:
                pass
            self._preview_window = None
        save_settings(self._settings)
        if self._settings.last_source_folder and self._records:
            save_session(self._settings.last_source_folder, self._records)
        self._engine.unload()
        event.accept()
