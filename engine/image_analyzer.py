"""
Image analysis pipeline for AI Mosaic Builder.

Responsibilities:
  1. Discover image files in a folder
  2. Pre-filter (dimensions, file validity, duplicates)
  3. Check cache — skip already-analysed images
  4. Queue remaining images for LLM analysis (one at a time)
  5. Compute perceptual hash for similarity detection
  6. Store results via cache

The actual LLM call is delegated to engine.vision_llm.VisionLLMEngine.
This module is called from background workers — never from the UI thread.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

from engine.cache import AnalysisCache, compute_file_hash
from engine.models import (
    AppSettings,
    ImageAnalysis,
    ImageRecord,
    ImageStatus,
    RejectReason,
)
from engine.vision_llm import VisionLLMEngine

logger = logging.getLogger("image_analyzer")

# Supported image extensions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}

# Minimum file size (1 KB) — skip obviously corrupt tiny files
_MIN_FILE_SIZE = 1024


# ─── Discovery ────────────────────────────────────────────────────────────────

def discover_images(folder: str) -> list[str]:
    """
    Recursively find all image files under folder.
    Returns absolute paths sorted alphabetically.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        logger.error(f"[analyzer] Not a directory: {folder}")
        return []

    found = []
    for root, _dirs, files in os.walk(folder_path):
        for name in files:
            if Path(name).suffix.lower() in _IMAGE_EXTENSIONS:
                found.append(str(Path(root) / name))

    found.sort()
    logger.info(f"[analyzer] Discovered {len(found)} image file(s) in {folder}")
    return found


# ─── Pre-filter ───────────────────────────────────────────────────────────────

def prefilter_image(path: str, settings: AppSettings) -> tuple[bool, str]:
    """
    Quick checks before spending inference on an image.
    Returns (passed: bool, reason: str).
    """
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return False, f"Cannot stat file: {e}"

    if size < _MIN_FILE_SIZE:
        return False, RejectReason.CORRUPT.value

    # Try opening with Pillow to check dimensions and integrity
    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as img:
            w, h = img.size
            img.verify()  # checks file integrity (does NOT decode fully)
    except Exception as e:
        return False, f"{RejectReason.CORRUPT.value}: {e}"

    # Re-open to get dimensions after verify() (verify leaves file in bad state)
    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as img:
            w, h = img.size
    except Exception as e:
        return False, f"{RejectReason.CORRUPT.value}: {e}"

    if w < settings.min_image_width or h < settings.min_image_height:
        return False, f"{RejectReason.TOO_SMALL.value}: {w}x{h}"

    return True, ""


def get_image_dimensions(path: str) -> tuple[int, int]:
    """Return (width, height) or (0, 0) on failure."""
    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as img:
            return img.size
    except Exception:
        return 0, 0


# ─── Perceptual hashing ───────────────────────────────────────────────────────

def compute_phash(path: str) -> str:
    """
    Compute a perceptual hash string for image similarity detection.
    Falls back to empty string if imagehash is not installed.
    """
    try:
        import imagehash  # type: ignore
        from PIL import Image as PILImage
        with PILImage.open(path) as img:
            return str(imagehash.phash(img))
    except ImportError:
        logger.debug("[analyzer] imagehash not available — phash skipped")
        return ""
    except Exception as e:
        logger.debug(f"[analyzer] phash failed for {path}: {e}")
        return ""


def are_similar(phash_a: str, phash_b: str, threshold: int = 10) -> bool:
    """
    Return True if two phash strings are within Hamming distance threshold.
    """
    if not phash_a or not phash_b:
        return False
    try:
        import imagehash  # type: ignore
        h1 = imagehash.hex_to_hash(phash_a)
        h2 = imagehash.hex_to_hash(phash_b)
        return (h1 - h2) <= threshold
    except Exception:
        return False


# ─── Build ImageRecord from path ──────────────────────────────────────────────

def build_record(path: str) -> ImageRecord:
    """Build a minimal ImageRecord from file path (before analysis)."""
    p = Path(path)
    file_size = 0
    try:
        file_size = os.path.getsize(path)
    except OSError:
        pass

    w, h = get_image_dimensions(path)
    return ImageRecord(
        path=path,
        filename=p.name,
        width=w,
        height=h,
        file_size=file_size,
        status=ImageStatus.PENDING,
    )


# ─── Main analysis pipeline ───────────────────────────────────────────────────

class AnalysisPipeline:
    """
    Orchestrates the full per-image pipeline:
      pre-filter → hash → cache check → LLM analysis → store

    Designed to be run from a QThread worker.
    progress_callback receives (done: int, total: int, record: ImageRecord).
    cancel_check returns True when the user pressed Stop.
    """

    def __init__(
        self,
        engine: VisionLLMEngine,
        cache: AnalysisCache,
        settings: AppSettings,
        progress_callback: Optional[Callable[[int, int, ImageRecord], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._engine = engine
        self._cache = cache
        self._settings = settings
        self._progress_callback = progress_callback
        self._cancel_check = cancel_check

        # Stats for logging
        self._n_cache_hits = 0
        self._n_cache_misses = 0
        self._n_prefilter_rejected = 0
        self._n_errors = 0
        self._n_people = 0

    def run(self, paths: list[str]) -> list[ImageRecord]:
        """
        Process a list of image paths and return ImageRecord list.
        Records for previously cached images are restored from cache.
        """
        total = len(paths)
        results: list[ImageRecord] = []
        model_name = self._engine.model_name

        seen_hashes: set[str] = set()   # for exact-duplicate detection

        for idx, path in enumerate(paths):
            if self._cancel_check and self._cancel_check():
                logger.info(f"[analyzer] Cancelled at {idx}/{total}")
                break

            record = build_record(path)

            # ── Step 1: pre-filter ─────────────────────────────────────────────
            passed, reason = prefilter_image(path, self._settings)
            if not passed:
                record.status = ImageStatus.REJECTED
                record.error_message = reason
                self._n_prefilter_rejected += 1
                logger.debug(f"[analyzer] Pre-filter rejected: {path} ({reason})")
                results.append(record)
                self._emit_progress(idx + 1, total, record)
                continue

            record.status = ImageStatus.PREFILTERED

            # ── Step 2: compute file hash ──────────────────────────────────────
            try:
                file_hash = compute_file_hash(path)
                record.file_hash = file_hash
            except Exception as e:
                record.status = ImageStatus.ERROR
                record.error_message = f"Hash error: {e}"
                self._n_errors += 1
                results.append(record)
                self._emit_progress(idx + 1, total, record)
                continue

            # ── Step 3: exact duplicate check ──────────────────────────────────
            if file_hash in seen_hashes:
                record.status = ImageStatus.REJECTED
                record.error_message = RejectReason.SIMILAR.value
                results.append(record)
                self._emit_progress(idx + 1, total, record)
                continue
            seen_hashes.add(file_hash)

            # ── Step 4: cache check ────────────────────────────────────────────
            cached = self._cache.get(file_hash, model=model_name)
            if cached is not None:
                record.analysis = cached
                record.status = ImageStatus.CACHED
                self._n_cache_hits += 1
                if cached.has_person:
                    self._n_people += 1
                # Still compute phash (needed for diversity, not cached by analysis)
                record.phash = compute_phash(path)
                results.append(record)
                self._emit_progress(idx + 1, total, record)
                continue

            # ── Step 5: LLM analysis ───────────────────────────────────────────
            self._n_cache_misses += 1
            record.status = ImageStatus.ANALYZING
            self._emit_progress(idx + 1, total, record)

            t_start = time.perf_counter()
            try:
                analysis = self._engine.analyze_image(path)
            except Exception as e:
                logger.error(f"[analyzer] LLM error for {path}: {e}")
                analysis = ImageAnalysis.error_result(str(e), model_name)
                self._n_errors += 1

            elapsed = time.perf_counter() - t_start
            logger.debug(f"[analyzer] Analysed in {elapsed:.1f}s: {Path(path).name}")

            record.analysis = analysis
            record.status = ImageStatus.ANALYZED if not analysis.reject else ImageStatus.REJECTED

            if analysis.has_person:
                self._n_people += 1

            # ── Step 6: compute phash ──────────────────────────────────────────
            record.phash = compute_phash(path)

            # ── Step 7: store in cache ─────────────────────────────────────────
            self._cache.put(file_hash, path, analysis, model=model_name)
            self._cache.flush()

            results.append(record)
            self._emit_progress(idx + 1, total, record)

        self._log_summary(len(paths))
        return results

    def _emit_progress(
        self, done: int, total: int, record: ImageRecord
    ) -> None:
        if self._progress_callback:
            try:
                self._progress_callback(done, total, record)
            except Exception:
                pass

    def _log_summary(self, total: int) -> None:
        logger.info(
            f"[analyzer] Summary: total={total} "
            f"cache_hits={self._n_cache_hits} "
            f"cache_misses={self._n_cache_misses} "
            f"prefilter_rejected={self._n_prefilter_rejected} "
            f"errors={self._n_errors} "
            f"people_detected={self._n_people}"
        )
