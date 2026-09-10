"""
Analysis cache for AI Mosaic Builder.

Stores analysis results keyed by file SHA-256 hash.
This means:
  - Moving or renaming a file does NOT invalidate its cache entry.
  - Changing target_images or ranking weights does NOT re-trigger analysis.
  - Corrupt or truncated files that change their content DO get re-analysed.

The cache is a single JSON file (one dict per entry).
Format:
  {
    "<sha256>": {
      "file_hash":        "<sha256>",
      "path":             "<last known absolute path>",
      "analysis_result":  { ... ImageAnalysis.to_dict() ... },
      "model":            "<model filename>",
      "analysis_version": 1,
      "timestamp":        "<iso datetime>"
    },
    ...
  }

Cache hits/misses are logged at INFO level.
The path stored is for human reference only — the hash is the real key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

from engine.models import ImageAnalysis

logger = logging.getLogger("cache")

_CACHE_VERSION = 1
_DEFAULT_CACHE_FILENAME = "analysis_cache.json"


class AnalysisCache:
    """Thread-safe read/write cache backed by a JSON file."""

    def __init__(self, cache_dir: str = "") -> None:
        if cache_dir:
            self._cache_path = Path(cache_dir) / _DEFAULT_CACHE_FILENAME
        else:
            self._cache_path = Path(__file__).parent.parent / "data" / _DEFAULT_CACHE_FILENAME

        self._data: dict[str, dict] = {}
        self._dirty = False
        self._load()

    # ── Public interface ───────────────────────────────────────────────────────

    def get(self, file_hash: str, model: str = "") -> Optional[ImageAnalysis]:
        """
        Return cached analysis if available and model matches.
        model="" skips the model check (useful for testing).
        """
        entry = self._data.get(file_hash)
        if entry is None:
            logger.debug(f"[cache] MISS {file_hash[:12]}…")
            return None
        if model and entry.get("model", "") != model:
            logger.debug(
                f"[cache] MISS (model changed) {file_hash[:12]}… "
                f"cached={entry.get('model', '?')} requested={model}"
            )
            return None
        try:
            analysis = ImageAnalysis.from_dict(entry["analysis_result"])
            logger.debug(f"[cache] HIT  {file_hash[:12]}… path={entry.get('path', '?')}")
            return analysis
        except Exception as e:
            logger.warning(f"[cache] Corrupt entry {file_hash[:12]}: {e}")
            return None

    def put(
        self,
        file_hash: str,
        path: str,
        analysis: ImageAnalysis,
        model: str = "",
    ) -> None:
        """Store or overwrite an analysis result."""
        from datetime import datetime
        self._data[file_hash] = {
            "file_hash":        file_hash,
            "path":             path,
            "analysis_result":  analysis.to_dict(),
            "model":            model,
            "analysis_version": _CACHE_VERSION,
            "timestamp":        datetime.now().isoformat(),
        }
        self._dirty = True
        logger.debug(f"[cache] STORE {file_hash[:12]}… path={path}")

    def has(self, file_hash: str) -> bool:
        return file_hash in self._data

    def size(self) -> int:
        return len(self._data)

    def flush(self) -> None:
        """Write cache to disk if dirty."""
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
            logger.debug(f"[cache] Flushed {len(self._data)} entries to {self._cache_path}")
        except Exception as e:
            logger.error(f"[cache] Failed to flush: {e}")

    def clear(self) -> None:
        self._data.clear()
        self._dirty = True
        self.flush()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._cache_path.exists():
            logger.info(f"[cache] No cache file found at {self._cache_path} — starting fresh")
            return
        try:
            text = self._cache_path.read_text(encoding="utf-8")
            self._data = json.loads(text)
            logger.info(f"[cache] Loaded {len(self._data)} cached entries from {self._cache_path}")
        except Exception as e:
            logger.warning(f"[cache] Could not load cache: {e} — starting fresh")
            self._data = {}


# ─── File hashing ─────────────────────────────────────────────────────────────

def compute_file_hash(path: str, chunk_size: int = 65536) -> str:
    """
    Compute SHA-256 of a file's content.
    Returns hex string, e.g. "a3f2...".
    Raises OSError if file cannot be read.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# ─── Module-level singleton ───────────────────────────────────────────────────

_default_cache: Optional[AnalysisCache] = None


def get_cache(cache_dir: str = "") -> AnalysisCache:
    """Return (or create) the module-level cache singleton."""
    global _default_cache
    if _default_cache is None:
        _default_cache = AnalysisCache(cache_dir)
    return _default_cache


def reset_cache_singleton() -> None:
    """For tests: discard the singleton so a new one is created on next call."""
    global _default_cache
    _default_cache = None
