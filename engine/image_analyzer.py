"""
Image analysis pipeline for AI Mosaic Builder.

Responsibilities:
  1. Discover image files in a folder
  2. Pre-filter (dimensions, file validity, duplicates)
  3. Check cache — skip already-analysed images
  4. Queue remaining images for LLM analysis (one at a time)
  5. Compute perceptual hash for similarity detection
  6. Store successful results via cache

The actual LLM call is delegated to engine.vision_llm.VisionLLMEngine.
This module is called from background workers — never from the UI thread.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

from engine.cache import AnalysisCache, compute_file_hash
from engine.models import (
    AppSettings,
    ImageAnalysis,
    ImageRecord,
    ImageStatus,
    RejectReason,
)
from engine.session import load_session
from engine.vision_llm import VisionLLMEngine

logger = logging.getLogger("image_analyzer")

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
_MIN_FILE_SIZE = 1024


def _normalize_excluded_folder(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def discover_images(folder: str, excluded_folders: list[str] | None = None) -> list[str]:
    """Recursively find image files under folder, skipping excluded subfolders."""
    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.is_dir():
        logger.error(f"[analyzer] Not a directory: {folder}")
        return []

    excluded = {
        _normalize_excluded_folder(value)
        for value in (excluded_folders or [])
        if _normalize_excluded_folder(value) not in {"", "."}
    }

    found: list[str] = []
    for root, dirs, files in os.walk(folder_path):
        root_path = Path(root)
        try:
            relative_root = root_path.relative_to(folder_path).as_posix()
        except ValueError:
            relative_root = ""

        # Prune excluded directories before os.walk descends into them.
        kept_dirs: list[str] = []
        for name in dirs:
            child_relative = f"{relative_root}/{name}".strip("/") if relative_root else name
            child_normalized = _normalize_excluded_folder(child_relative)
            if any(
                child_normalized == excluded_path
                or child_normalized.startswith(excluded_path + "/")
                for excluded_path in excluded
            ):
                logger.debug("[analyzer] Skipping excluded folder: %s", child_relative)
                continue
            kept_dirs.append(name)
        dirs[:] = kept_dirs

        for name in files:
            if Path(name).suffix.lower() in _IMAGE_EXTENSIONS:
                found.append(str(root_path / name))

    found.sort()
    logger.info(
        f"[analyzer] Discovered {len(found)} image file(s) in {folder} "
        f"(excluded_folders={len(excluded)})"
    )
    return found


def prefilter_image(path: str, settings: AppSettings) -> tuple[bool, str]:
    """Quick checks before spending inference on an image."""
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return False, f"Cannot stat file: {e}"

    if size < _MIN_FILE_SIZE:
        return False, RejectReason.CORRUPT.value

    try:
        from PIL import Image as PILImage
        with PILImage.open(path) as img:
            w, h = img.size
            img.verify()
    except Exception as e:
        return False, f"{RejectReason.CORRUPT.value}: {e}"

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


def compute_phash(path: str) -> str:
    """Compute a perceptual hash string for image similarity detection."""
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
    """Return True if two phash strings are within Hamming distance threshold."""
    if not phash_a or not phash_b:
        return False
    try:
        import imagehash  # type: ignore
        h1 = imagehash.hex_to_hash(phash_a)
        h2 = imagehash.hex_to_hash(phash_b)
        return (h1 - h2) <= threshold
    except Exception:
        return False


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


def cache_model_key(model_name: str, custom_prompt: str = "") -> str:
    """Return a cache model identifier that changes when the custom request changes."""
    request = (custom_prompt or "").strip()[:2000]
    if not request:
        return model_name
    digest = hashlib.sha256(request.encode("utf-8")).hexdigest()[:16]
    return f"{model_name}|request:{digest}"


class AnalysisPipeline:
    """
    Orchestrates the full per-image pipeline:
      pre-filter → hash → cache check → LLM analysis → store
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

        self._n_cache_hits = 0
        self._n_cache_misses = 0
        self._n_prefilter_rejected = 0
        self._n_errors = 0
        self._n_people = 0
        self._inference_seconds = 0.0

    def run(self, paths: list[str]) -> list[ImageRecord]:
        """Process image paths, reusing cache and compatible saved records."""
        run_started = time.perf_counter()
        total = len(paths)
        results: list[ImageRecord] = []
        model_name = self._engine.model_name
        custom_prompt = str(getattr(self._settings, "custom_prompt", "") or "").strip()[:2000]
        cache_model = cache_model_key(model_name, custom_prompt)

        session_by_path: dict[str, ImageRecord] = {}
        source_folder = str(getattr(self._settings, "last_source_folder", "") or "")
        if source_folder:
            try:
                saved_records = load_session(source_folder) or []
                session_by_path = {
                    str(Path(record.path).expanduser().resolve()): record
                    for record in saved_records
                    if record.path
                }
                if session_by_path:
                    logger.info("[analyzer] Loaded %d saved records from source session", len(session_by_path))
            except Exception as exc:
                logger.warning("[analyzer] Could not load saved session: %s", exc)

        logger.info(
            "[analyzer] START | images=%d | model=%s | engine_loaded=%s | custom_request=%s",
            total,
            model_name or "unknown",
            self._engine.model_loaded,
            "yes" if custom_prompt else "no",
        )

        seen_hashes: set[str] = set()

        for idx, path in enumerate(paths):
            if self._cancel_check and self._cancel_check():
                logger.info(f"[analyzer] Cancelled at {idx}/{total}")
                break

            image_started = time.perf_counter()
            record = build_record(path)

            prefilter_started = time.perf_counter()
            passed, reason = prefilter_image(path, self._settings)
            prefilter_elapsed = time.perf_counter() - prefilter_started
            if not passed:
                record.status = ImageStatus.REJECTED
                record.error_message = reason
                self._n_prefilter_rejected += 1
                logger.info(
                    "[analyzer] REJECT PREFILTER | %d/%d | image=%s | prefilter=%.3fs | reason=%s",
                    idx + 1, total, Path(path).name, prefilter_elapsed, reason,
                )
                results.append(record)
                self._emit_progress(idx + 1, total, record)
                continue

            record.status = ImageStatus.PREFILTERED

            hash_started = time.perf_counter()
            try:
                file_hash = compute_file_hash(path)
                record.file_hash = file_hash
            except Exception as e:
                record.status = ImageStatus.ERROR
                record.error_message = f"Hash error: {e}"
                self._n_errors += 1
                logger.error("[analyzer] HASH ERROR | image=%s | error=%s", Path(path).name, e)
                results.append(record)
                self._emit_progress(idx + 1, total, record)
                continue
            hash_elapsed = time.perf_counter() - hash_started

            if file_hash in seen_hashes:
                record.status = ImageStatus.REJECTED
                record.error_message = RejectReason.SIMILAR.value
                logger.info(
                    "[analyzer] REJECT DUPLICATE | %d/%d | image=%s | hash=%.3fs",
                    idx + 1, total, Path(path).name, hash_elapsed,
                )
                results.append(record)
                self._emit_progress(idx + 1, total, record)
                continue
            seen_hashes.add(file_hash)

            # Restore the full ImageRecord only when the exact file contents
            # match the saved record. This preserves detections/ranking/manual
            # choices without ever applying stale state to a replaced image.
            saved_record = session_by_path.get(str(Path(path).expanduser().resolve()))
            if saved_record is not None and saved_record.file_hash == file_hash:
                record = saved_record
                record.path = path
                record.filename = Path(path).name
                record.file_size = os.path.getsize(path)
                record.width, record.height = get_image_dimensions(path)
                record.file_hash = file_hash

            cache_started = time.perf_counter()
            cached = self._cache.get(file_hash, model=cache_model)
            cache_elapsed = time.perf_counter() - cache_started
            if cached is not None:
                record.analysis = cached
                record.status = ImageStatus.CACHED
                record.error_message = ""
                self._n_cache_hits += 1
                if cached.has_person:
                    self._n_people += 1
                phash_started = time.perf_counter()
                record.phash = compute_phash(path)
                phash_elapsed = time.perf_counter() - phash_started
                total_elapsed = time.perf_counter() - image_started
                logger.info(
                    "[analyzer] CACHE HIT | %d/%d | image=%s | cache=%.3fs | phash=%.3fs | total=%.3fs | restored_record=%s",
                    idx + 1, total, Path(path).name, cache_elapsed, phash_elapsed, total_elapsed,
                    "yes" if saved_record is not None and saved_record.file_hash == file_hash else "no",
                )
                results.append(record)
                self._emit_progress(idx + 1, total, record)
                continue

            self._n_cache_misses += 1
            logger.info(
                "[analyzer] CACHE MISS | %d/%d | image=%s | hash=%.3fs | cache=%.3fs",
                idx + 1, total, Path(path).name, hash_elapsed, cache_elapsed,
            )

            record.status = ImageStatus.ANALYZING
            self._emit_progress(idx + 1, total, record)

            t_start = time.perf_counter()
            analysis_failed = False
            try:
                analysis = self._engine.analyze_image(
                    path,
                    max_tokens=int(getattr(self._settings, "max_tokens", 576)),
                    custom_prompt=custom_prompt,
                )
            except Exception as e:
                logger.error(f"[analyzer] LLM error for {path}: {e}")
                analysis = ImageAnalysis.error_result(str(e), model_name)
                analysis_failed = True
                self._n_errors += 1

            elapsed = time.perf_counter() - t_start
            self._inference_seconds += elapsed
            logger.debug(f"[analyzer] Analysed in {elapsed:.1f}s: {Path(path).name}")

            record.analysis = analysis
            if analysis_failed or analysis.reject_reason == RejectReason.LLM_ERROR.value:
                record.status = ImageStatus.ERROR
                record.error_message = analysis.notes or "Vision analysis failed."
            else:
                record.status = ImageStatus.ANALYZED if not analysis.reject else ImageStatus.REJECTED
                if analysis.has_person:
                    self._n_people += 1

            phash_started = time.perf_counter()
            record.phash = compute_phash(path)
            phash_elapsed = time.perf_counter() - phash_started

            cache_write_started = time.perf_counter()
            self._cache.put(file_hash, path, analysis, model=cache_model)
            self._cache.flush()
            cache_write_elapsed = time.perf_counter() - cache_write_started

            total_elapsed = time.perf_counter() - image_started
            logger.info(
                "[analyzer] ANALYZED | %d/%d | image=%s | inference=%.2fs | phash=%.3fs | cache_write=%.3fs | total=%.2fs | status=%s",
                idx + 1, total, Path(path).name, elapsed, phash_elapsed, cache_write_elapsed, total_elapsed,
                record.status.value.upper(),
            )

            results.append(record)
            self._emit_progress(idx + 1, total, record)

        processed_paths = {record.path for record in results}
        if len(results) < total:
            remaining = [path for path in paths if path not in processed_paths]
            results.extend(build_record(path) for path in remaining)
            logger.info(
                "[analyzer] Preserved %d pending image(s) after cancellation",
                len(remaining),
            )

        run_elapsed = time.perf_counter() - run_started
        avg_inference = self._inference_seconds / self._n_cache_misses if self._n_cache_misses else 0.0
        logger.info(
            "[analyzer] END | total=%d | elapsed=%.2fs | cache_hits=%d | cache_misses=%d | inference_total=%.2fs | inference_avg=%.2fs | prefilter_rejected=%d | errors=%d | people_detected=%d | custom_request=%s",
            total, run_elapsed, self._n_cache_hits, self._n_cache_misses, self._inference_seconds,
            avg_inference, self._n_prefilter_rejected, self._n_errors, self._n_people,
            "yes" if custom_prompt else "no",
        )
        return results

    def _emit_progress(self, done: int, total: int, record: ImageRecord) -> None:
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
