"""
Analysis cache for AI Mosaic Builder.

Each source/project folder can have its own cache. Cache entries are still
keyed by SHA-256, so adding new photos only analyzes files that are not already
present in that source folder's cache.

Cache layout for a source folder:
    <source>/.aimosaic/analysis_cache.json

The cache stores analysis results, not image data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from engine.models import ImageAnalysis, RejectReason

logger = logging.getLogger("cache")

# ImageAnalysis.analysis_version is currently 3. Keep the cache schema aligned
# with it. Version 2 cache entries are accepted when they already contain a
# version-3 ImageAnalysis; they are upgraded in memory and rewritten on flush.
_CACHE_VERSION = 3
_LEGACY_CACHE_VERSIONS = {2}
_DEFAULT_CACHE_FILENAME = "analysis_cache.json"
_DEFAULT_CACHE_DIRNAME = ".aimosaic"


def _is_reusable_analysis(analysis: ImageAnalysis) -> bool:
    """Return True only for real model analysis results that are safe to cache.

    LLM/runtime failures use the `llm_error` reject reason. Those records must
    never become cache hits because a transient software/model failure would
    otherwise permanently poison the analysis for that image.
    """
    return analysis.reject_reason != RejectReason.LLM_ERROR.value


def _is_compatible_entry_version(value: object) -> bool:
    return value == _CACHE_VERSION or value in _LEGACY_CACHE_VERSIONS


def _cache_model_name(model_identifier: str) -> str:
    """Extract the vision model identity from current or legacy cache keys."""
    return str(model_identifier or "").split("|request:", 1)[0]


class RequestScoreCache:
    """Persist custom-prompt evaluations separately from reusable image analysis."""

    _FILENAME = "request_scores.json"

    def __init__(self, cache_dir: str | Path) -> None:
        self._cache_path = Path(cache_dir) / self._FILENAME
        self._data: dict[str, dict] = {}
        self._dirty = False
        self._load()

    @staticmethod
    def _key(file_hash: str, model: str, prompt: str) -> tuple[str, str]:
        normalized_prompt = str(prompt or "").strip()[:2000]
        prompt_hash = hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest()
        identity = f"{file_hash}|{model}|{prompt_hash}|request-score-v1"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest(), prompt_hash

    def get(self, file_hash: str, model: str, prompt: str) -> Optional[float]:
        if not str(prompt or "").strip():
            return None
        key, prompt_hash = self._key(file_hash, model, prompt)
        entry = self._data.get(key)
        if not isinstance(entry, dict):
            return None
        if entry.get("prompt_hash") != prompt_hash or entry.get("model") != model:
            return None
        try:
            score = float(entry.get("score"))
        except (TypeError, ValueError):
            return None
        return score if math.isfinite(score) and 0.0 <= score <= 1.0 else None

    def put(self, file_hash: str, model: str, prompt: str, score: float) -> None:
        if not str(prompt or "").strip():
            return
        value = float(score)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("request score must be a finite number between 0 and 1")
        key, prompt_hash = self._key(file_hash, model, prompt)
        self._data[key] = {
            "file_hash": file_hash,
            "model": model,
            "prompt_hash": prompt_hash,
            "score": value,
            "timestamp": datetime.now().isoformat(),
        }
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._cache_path)
            self._dirty = False
            logger.info("[cache] Flushed %d prompt scores to %s", len(self._data), self._cache_path)
        except Exception as exc:
            logger.error("[cache] Failed to flush prompt scores: %s", exc)

    def clear(self) -> None:
        self._data.clear()
        self._dirty = True
        self.flush()

    def _load(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            parsed = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("prompt score cache root must be an object")
            self._data = parsed
        except Exception as exc:
            logger.warning("[cache] Could not load prompt score cache %s: %s", self._cache_path, exc)
            self._data = {}



class AnalysisCache:
    """Read/write JSON cache scoped to a specific source/project folder."""

    def __init__(self, cache_dir: str = "") -> None:
        if cache_dir:
            self._cache_path = Path(cache_dir) / _DEFAULT_CACHE_FILENAME
        else:
            self._cache_path = (
                Path(__file__).parent.parent
                / "data"
                / _DEFAULT_CACHE_FILENAME
            )

        self._data: dict[str, dict] = {}
        self._dirty = False
        self._load()
        self._request_score_cache = RequestScoreCache(self.cache_dir)

    @property
    def cache_path(self) -> Path:
        return self._cache_path.resolve()

    @property
    def cache_dir(self) -> Path:
        return self.cache_path.parent

    def get(self, file_hash: str, model: str = "") -> Optional[ImageAnalysis]:
        entry = self._data.get(file_hash)
        if entry is None:
            logger.debug(f"[cache] MISS {file_hash[:12]}…")
            return None

        if not _is_compatible_entry_version(entry.get("analysis_version")):
            logger.debug(
                f"[cache] MISS (analysis version changed) {file_hash[:12]}… "
                f"cached={entry.get('analysis_version', '?')} requested={_CACHE_VERSION}"
            )
            return None

        cached_model = str(entry.get("model", "") or "")
        cached_model_name = _cache_model_name(cached_model)
        requested_model_name = _cache_model_name(model)
        if model and cached_model_name and cached_model_name != requested_model_name:
            logger.debug(
                "[cache] MISS (vision model changed) %s… cached=%s requested=%s",
                file_hash[:12], cached_model_name, requested_model_name,
            )
            return None
        if cached_model and cached_model != model:
            logger.debug(
                "[cache] HIT (legacy prompt-scoped analysis reused) %s… cached_key=%s requested_model=%s",
                file_hash[:12], cached_model, model,
            )

        try:
            analysis = ImageAnalysis.from_dict(entry["analysis_result"])
            analysis.user_request_score = 1.0
            analysis.visual_tags.pop("user_request_score", None)
            if analysis.analysis_version != _CACHE_VERSION:
                logger.debug(
                    f"[cache] MISS (analysis schema changed) {file_hash[:12]}… "
                    f"cached={analysis.analysis_version} requested={_CACHE_VERSION}"
                )
                return None
            if not _is_reusable_analysis(analysis):
                logger.info(
                    "[cache] INVALIDATE failed analysis %s… reason=%s",
                    file_hash[:12],
                    analysis.reject_reason,
                )
                self._data.pop(file_hash, None)
                self._dirty = True
                return None
            if entry.get("analysis_version") != _CACHE_VERSION:
                entry["analysis_version"] = _CACHE_VERSION
                self._dirty = True
            logger.debug(
                f"[cache] HIT  {file_hash[:12]}… "
                f"path={entry.get('path', '?')}"
            )
            return analysis
        except Exception as exc:
            logger.warning(f"[cache] Corrupt entry {file_hash[:12]}: {exc}")
            return None

    def get_entry_by_path(self, path: str, model: str = "") -> Optional[tuple[str, ImageAnalysis]]:
        """Return the cached file hash and analysis for a path without hashing the file."""
        requested = Path(path).expanduser().resolve()
        for file_hash, entry in list(self._data.items()):
            if not _is_compatible_entry_version(entry.get("analysis_version")):
                continue
            cached_model = str(entry.get("model", "") or "")
            cached_model_name = _cache_model_name(cached_model)
            requested_model_name = _cache_model_name(model)
            if model and cached_model_name and cached_model_name != requested_model_name:
                continue
            cached_path = entry.get("path", "")
            if not cached_path:
                continue
            try:
                if Path(cached_path).expanduser().resolve() != requested:
                    continue
                analysis = ImageAnalysis.from_dict(entry["analysis_result"])
                analysis.user_request_score = 1.0
                analysis.visual_tags.pop("user_request_score", None)
                if analysis.analysis_version != _CACHE_VERSION:
                    continue
                if not _is_reusable_analysis(analysis):
                    logger.info(
                        "[cache] INVALIDATE failed path cache path=%s reason=%s",
                        path,
                        analysis.reject_reason,
                    )
                    self._data.pop(file_hash, None)
                    self._dirty = True
                    return None
                if entry.get("analysis_version") != _CACHE_VERSION:
                    entry["analysis_version"] = _CACHE_VERSION
                    self._dirty = True
                logger.debug("[cache] PATH HIT path=%s hash=%s…", path, file_hash[:12])
                return file_hash, analysis
            except Exception:
                continue
        logger.debug("[cache] PATH MISS path=%s", path)
        return None

    def get_by_path(self, path: str, model: str = "") -> Optional[ImageAnalysis]:
        """Backward-compatible analysis-only lookup by source path."""
        cached_entry = self.get_entry_by_path(path, model=model)
        return cached_entry[1] if cached_entry is not None else None

    def put(self, file_hash: str, path: str, analysis: ImageAnalysis, model: str = "") -> None:
        """Store only reusable analysis results; never cache LLM/runtime failures."""
        if not _is_reusable_analysis(analysis):
            logger.debug(
                "[cache] SKIP failed analysis %s… reason=%s",
                file_hash[:12],
                analysis.reject_reason,
            )
            return
        self._data[file_hash] = {
            "file_hash": file_hash,
            "path": path,
            "analysis_result": analysis.to_dict(),
            "model": model,
            "analysis_version": _CACHE_VERSION,
            "timestamp": datetime.now().isoformat(),
        }
        self._dirty = True
        logger.debug(f"[cache] STORE {file_hash[:12]}… path={path}")

    def get_request_score(self, file_hash: str, model: str, prompt: str) -> Optional[float]:
        return self._request_score_cache.get(file_hash, model, prompt)

    def put_request_score(self, file_hash: str, model: str, prompt: str, score: float) -> None:
        self._request_score_cache.put(file_hash, model, prompt, score)

    def has(self, file_hash: str) -> bool:
        entry = self._data.get(file_hash)
        return bool(entry and _is_compatible_entry_version(entry.get("analysis_version")))

    def size(self) -> int:
        return len(self._data)

    def flush(self) -> None:
        if not self._dirty:
            self._request_score_cache.flush()
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(self._cache_path)
            self._dirty = False
            logger.info(
                f"[cache] Flushed {len(self._data)} entries to {self._cache_path}"
            )
        except Exception as exc:
            logger.error(f"[cache] Failed to flush: {exc}")
        self._request_score_cache.flush()

    def clear(self) -> None:
        self._data.clear()
        self._dirty = True
        self._request_score_cache.clear()
        self.flush()

    def _load(self) -> None:
        if not self._cache_path.exists():
            logger.info(
                f"[cache] No cache file found at {self._cache_path} — starting fresh"
            )
            return
        try:
            text = self._cache_path.read_text(encoding="utf-8")
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("cache root must be a JSON object")
            self._data = parsed
            logger.info(
                f"[cache] Loaded {len(self._data)} cached entries from {self._cache_path}"
            )
        except Exception as exc:
            logger.warning(
                f"[cache] Could not load cache {self._cache_path}: {exc} — starting fresh"
            )
            self._data = {}


def source_cache_dir(source_folder: str | Path) -> Path:
    """Return the cache directory belonging to a source/project folder."""
    source = Path(source_folder).expanduser().resolve()
    return source / _DEFAULT_CACHE_DIRNAME


def compute_file_hash(path: str, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


_default_cache: Optional[AnalysisCache] = None


def get_cache(cache_dir: str = "") -> AnalysisCache:
    """Return a cache singleton scoped to the requested cache directory."""
    global _default_cache
    requested = Path(cache_dir).expanduser().resolve() if cache_dir else None

    if _default_cache is None:
        _default_cache = AnalysisCache(str(requested) if requested else "")
        return _default_cache

    current = _default_cache.cache_dir.resolve()
    desired = requested if requested is not None else current
    if current != desired:
        _default_cache.flush()
        _default_cache = AnalysisCache(str(desired))
    return _default_cache


def reset_cache_singleton() -> None:
    global _default_cache
    if _default_cache is not None:
        _default_cache.flush()
    _default_cache = None
