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


def _cache_request_key(model_identifier: str) -> str:
    """Return the custom-request digest used for cache compatibility.

    Model identity and preprocessing details are intentionally ignored here.
    A vision resize changes performance/input representation, not the meaning
    of an already completed semantic analysis, so cached results remain
    reusable across AI detection image sizes.
    """
    marker = "|request:"
    text = str(model_identifier or "")
    if marker not in text:
        return ""
    request_part = text.split(marker, 1)[1]
    return request_part.split("|", 1)[0]

