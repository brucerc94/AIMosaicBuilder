"""Background workers for AI Mosaic Builder."""
from __future__ import annotations

import logging
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from engine.cache import AnalysisCache
from engine.image_analyzer import AnalysisPipeline
from engine.layout_optimizer import apply_layout_selection, optimize_auto_layout
from engine.models import AppSettings, ImageRecord
from engine.ranking import rank_records
from engine.vision_llm import VisionLLMEngine
from vision.person_detector import detect_persons

logger = logging.getLogger("workers")


class ModelLoaderSignals(QObject):
    progress = Signal(str)
    finished = Signal()
    error = Signal(str)


class ModelLoaderWorker(QRunnable):
    def __init__(self, engine: VisionLLMEngine, model_path: str, mmproj_path: str, settings: AppSettings):
        super().__init__()
        self._engine = engine
        self._model_path = model_path
        self._mmproj_path = mmproj_path
        self._settings = settings
        self.signals = ModelLoaderSignals()

    def run(self) -> None:
        try:
            self._engine.load_model(
                model_path=self._model_path,
                mmproj_path=self._mmproj_path,
                n_ctx=self._settings.n_ctx,
                n_gpu_layers=self._settings.n_gpu_layers,
                n_threads=self._settings.n_threads,
                n_threads_batch=self._settings.n_threads_batch,
                progress_callback=self.signals.progress.emit,
            )
            self.signals.finished.emit()
        except Exception as exc:
            logger.error("Model load failed: %s", exc, exc_info=True)
            self.signals.error.emit(str(exc))


class AnalysisSignals(QObject):
    progress = Signal(int, int, object)
    finished = Signal(list)
    error = Signal(str)


class AnalysisWorker(QRunnable):
    def __init__(self, engine: VisionLLMEngine, cache: AnalysisCache, settings: AppSettings, paths: list[str]):
        super().__init__()
        self._engine = engine
        self._cache = cache
        self._settings = settings
        self._paths = paths
        self._cancelled = False
        self.signals = AnalysisSignals()

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            pipeline = AnalysisPipeline(
                engine=self._engine,
                cache=self._cache,
                settings=self._settings,
                progress_callback=lambda done, total, record: self.signals.progress.emit(done, total, record),
                cancel_check=lambda: self._cancelled,
            )
            self.signals.finished.emit(pipeline.run(self._paths))
        except Exception as exc:
            logger.error("Analysis pipeline failed: %s", exc, exc_info=True)
            self.signals.error.emit(str(exc))


class PostProcessSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(list)
    error = Signal(str)


class PostProcessWorker(QRunnable):
    """Run person detection, ranking and automatic layout selection off the UI thread."""

    def __init__(self, records: list[ImageRecord], settings: AppSettings):
        super().__init__()
        self._records = records
        self._settings = settings
        self._cancelled = False
        self.signals = PostProcessSignals()

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            total = len(self._records)
            detector_model = getattr(self._settings, "person_detector_model", "yolo26n.pt")
            detector_conf = float(getattr(self._settings, "person_detector_confidence", 0.25))
            detector = None
            detector_error = None

            try:
                from vision.person_detector import get_default_detector
                detector = get_default_detector(model_path=detector_model, confidence=detector_conf)
                logger.info("[post] person detector=%s", detector.name)
            except Exception as exc:
                detector_error = str(exc)
                logger.error("[post] person detector initialization failed: %s", exc, exc_info=True)

            for index, record in enumerate(self._records, start=1):
                if self._cancelled:
                    break
                if record.analysis and record.analysis.has_person and not record.manually_excluded:
                    if detector is None:
                        record.error_message = detector_error or "No person detector available."
                    else:
                        record.detections = detect_persons(
                            record.path,
                            record.analysis,
                            backend=detector,
                            model_path=detector_model,
                            confidence=detector_conf,
                        )
                        if not record.detections:
                            record.error_message = "No real person bounding box detected."
                self.signals.progress.emit(index, total, record.filename)

            requirements = self._settings.mosaic_requirements
            ranked = rank_records(self._records, self._settings.ranking_weights, requirements)

            max_images = 100 if self._settings.target_images == 0 else max(0, self._settings.target_images)
            layout = optimize_auto_layout(
                ranked,
                canvas_size=(self._settings.canvas_width, self._settings.canvas_height),
                padding_px=self._settings.padding_px,
                requirements=requirements,
                phash_threshold=self._settings.phash_threshold,
                max_images=max_images,
                initial_zoom=0.5,
                min_zoom=0.1,
                zoom_decay=0.9,
                min_subject_px=120,
            )
            apply_layout_selection(layout, ranked)

            if layout.unmet_requirements:
                logger.warning(
                    "[post] Unmet mosaic requirements: %s",
                    ", ".join(layout.unmet_requirements),
                )
            mode = "AUTO" if self._settings.target_images == 0 else f"FIXED={self._settings.target_images}"
            logger.info(
                "[post] layout mode=%s selected=%d avg_zoom=%.3f canvas_fill=%.1f%%",
                mode,
                len(layout.placements),
                layout.average_zoom,
                layout.canvas_fill_ratio * 100.0,
            )
            self.signals.finished.emit(ranked)
        except Exception as exc:
            logger.error("Post-processing failed: %s", exc, exc_info=True)
            self.signals.error.emit(str(exc))


class ThumbnailSignals(QObject):
    done = Signal(str, bytes)


class ThumbnailWorker(QRunnable):
    def __init__(self, image_path: str, size: tuple[int, int] = (180, 180)):
        super().__init__()
        self._path = image_path
        self._size = size
        self.signals = ThumbnailSignals()

    def run(self) -> None:
        try:
            from vision.cropper import make_thumbnail_bytes
            self.signals.done.emit(self._path, make_thumbnail_bytes(self._path, self._size))
        except Exception:
            self.signals.done.emit(self._path, b"")


class ThumbnailPool:
    def __init__(self, on_done: Callable[[str, bytes], None], max_threads: int = 4):
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(max_threads)
        self._on_done = on_done

    def request(self, path: str, size: tuple[int, int] = (180, 180)) -> None:
        worker = ThumbnailWorker(path, size)
        worker.signals.done.connect(self._on_done)
        self._pool.start(worker)

    def wait_all(self) -> None:
        self._pool.waitForDone()
