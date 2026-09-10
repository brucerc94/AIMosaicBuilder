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
from datetime import datetime
from pathlib import Path
from typing import Optional

from engine.models import ImageAnalysis

logger = logging.getLogger("cache")

_CACHE_VERSION = 2
_DEFAULT_CACHE_FILENAME = "analysis_cache.json"
_DEFAULT_CACHE_DIRNAME = ".aimosaic"


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

        if entry.get("analysis_version") != _CACHE_VERSION:
            logger.debug(
                f"[cache] MISS (analysis version changed) {file_hash[:12]}… "
                f"cached={entry.get('analysis_version', '?')} requested={_CACHE_VERSION}"
            )
            return None

        if model and entry.get("model", "") != model:
            logger.debug(
                f"[cache] MISS (model changed) {file_hash[:12]}… "
                f"cached={entry.get('model', '?')} requested={model}"
            )
            return None

        try:
            analysis = ImageAnalysis.from_dict(entry["analysis_result"])
            if analysis.analysis_version != _CACHE_VERSION:
                logger.debug(
                    f"[cache] MISS (analysis schema changed) {file_hash[:12]}… "
                    f"cached={analysis.analysis_version} requested={_CACHE_VERSION}"
                )
                return None
            logger.debug(
                f"[cache] HIT  {file_hash[:12]}… "
                f"path={entry.get('path', '?')}"
            )
            return analysis
        except Exception as exc:
            logger.warning(f"[cache] Corrupt entry {file_hash[:12]}: {exc}")
            return None

    def put(self, file_hash: str, path: str, analysis: ImageAnalysis, model: str = "") -> None:
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

    def has(self, file_hash: str) -> bool:
        entry = self._data.get(file_hash)
        return bool(entry and entry.get("analysis_version") == _CACHE_VERSION)

    def size(self) -> int:
        return len(self._data)

    def flush(self) -> None:
        if not self._dirty:
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

    def clear(self) -> None:
        self._data.clear()
        self._dirty = True
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
