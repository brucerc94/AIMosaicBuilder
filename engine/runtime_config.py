"""Runtime configuration adapters for AI Mosaic Builder."""
from __future__ import annotations

from functools import wraps
from typing import Any

_DEFAULT_MAX_TOKENS = 576
_DEFAULT_IMAGE_SIZE = 1024

def _coerce_max_tokens(value: Any) -> int:
    try:
        return max(64, min(4096, int(value)))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_TOKENS


def _coerce_image_size(value: Any) -> int:
    try:
        return max(384, min(2048, int(value)))
    except (TypeError, ValueError):
        return _DEFAULT_IMAGE_SIZE


def configure_vision_engine(engine: Any, settings: Any) -> None:
    """Bind persisted UI settings to the existing runtime components."""
    engine._aimosaic_max_tokens = _coerce_max_tokens(
        getattr(settings, "max_tokens", _DEFAULT_MAX_TOKENS)
    )
    engine._aimosaic_image_max_dimension = _coerce_image_size(
        getattr(settings, "ai_detection_image_size", _DEFAULT_IMAGE_SIZE)
    )
    if getattr(engine, "_aimosaic_settings_wrapper_installed", False):
        return

    original_analyze_image = engine.analyze_image

    @wraps(original_analyze_image)
    def analyze_image(
        image_path: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        custom_prompt: str = "",
        image_max_dimension: int | None = None,
    ):
        effective = (
            engine._aimosaic_max_tokens
            if max_tokens is None
            else _coerce_max_tokens(max_tokens)
        )
        effective_image_size = (
            engine._aimosaic_image_max_dimension
            if image_max_dimension is None
            else _coerce_image_size(image_max_dimension)
        )
        return original_analyze_image(
            image_path,
            max_tokens=effective,
            temperature=temperature,
            custom_prompt=custom_prompt,
            image_max_dimension=effective_image_size,
        )

    engine.analyze_image = analyze_image
    engine._aimosaic_settings_wrapper_installed = True
