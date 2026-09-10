"""
Background workers for AI Mosaic Builder.
All heavy operations run in QThread — never on the UI thread.

Workers:
  - ModelLoaderWorker : loads the GGUF model
  - AnalysisWorker    : runs the full analysis pipeline
  - ThumbnailWorker   : generates thumbnails for image grid
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QThread, QThreadPool, Signal

from engine.cache import AnalysisCache
from engine.image_analyzer import AnalysisPipeline, discover_images
from engine.models import AppSettings, ImageRecord, ImageStatus
from engine.vision_llm import VisionLLMEngine

logger = logging.getLogger("workers")


# ─── Model Loader ─────────────────────────────────────────────────────────────

class ModelLoaderSignals(QObject):
    progress   = Signal(str)          # status message
    finished   = Signal()
    error      = Signal(str)


class ModelLoaderWorker(QRunnable):
    """
    Loads the GGUF model in the background.
    Emits progress/finished/error signals.
    """

    def __init__(
        self,
        engine: VisionLLMEngine,
        model_path: str,
        mmproj_path: str,
        settings: AppSettings,
    ) -> None:
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
                progress_callback=lambda msg: self.signals.progress.emit(msg),
            )
            self.signals.finished.emit()
        except Exception as e:
            logger.error(f"[workers] Model load failed: {e}")
            self.signals.error.emit(str(e))


# ─── Analysis Worker ──────────────────────────────────────────────────────────

class AnalysisSignals(QObject):
    progress   = Signal(int, int, object)   # done, total, ImageRecord
    finished   = Signal(list)               # list[ImageRecord]
    error      = Signal(str)
    status_msg = Signal(str)


class AnalysisWorker(QRunnable):
    """
    Runs the full analysis pipeline for a list of image paths.

    Emits:
      progress(done, total, record) — after each image
      finished(records)             — when complete
      error(message)                — on fatal error
    """

    def __init__(
        self,
        engine: VisionLLMEngine,
        cache: AnalysisCache,
        settings: AppSettings,
        paths: list[str],
    ) -> None:
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
                progress_callback=self._on_progress,
                cancel_check=lambda: self._cancelled,
            )
            records = pipeline.run(self._paths)
            self.signals.finished.emit(records)
        except Exception as e:
            logger.error(f"[workers] Analysis pipeline error: {e}", exc_info=True)
            self.signals.error.emit(str(e))

    def _on_progress(self, done: int, total: int, record: ImageRecord) -> None:
        self.signals.progress.emit(done, total, record)


# ─── Thumbnail Worker ─────────────────────────────────────────────────────────

class ThumbnailSignals(QObject):
    done = Signal(str, bytes)   # (image_path, png_bytes)


class ThumbnailWorker(QRunnable):
    """
    Generates a single thumbnail in the background.
    """

    def __init__(
        self,
        image_path: str,
        size: tuple[int, int] = (180, 180),
    ) -> None:
        super().__init__()
        self._path = image_path
        self._size = size
        self.signals = ThumbnailSignals()

    def run(self) -> None:
        from vision.cropper import make_thumbnail_bytes
        data = make_thumbnail_bytes(self._path, self._size)
        self.signals.done.emit(self._path, data)


# ─── Thumbnail Pool ───────────────────────────────────────────────────────────

class ThumbnailPool:
    """
    Manages a pool of thumbnail workers.
    Emits thumbnails as they complete.
    """

    def __init__(
        self,
        on_done: Callable[[str, bytes], None],
        max_threads: int = 4,
    ) -> None:
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(max_threads)
        self._on_done = on_done

    def request(self, path: str, size: tuple[int, int] = (180, 180)) -> None:
        worker = ThumbnailWorker(path, size)
        worker.signals.done.connect(self._on_done)
        self._pool.start(worker)

    def wait_all(self) -> None:
        self._pool.waitForDone()
