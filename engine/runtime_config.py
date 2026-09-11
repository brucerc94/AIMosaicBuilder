"""Runtime configuration adapters for AI Mosaic Builder."""
from __future__ import annotations

from functools import wraps
from typing import Any

_DEFAULT_MAX_TOKENS = 256


def _coerce_max_tokens(value: Any) -> int:
    try:
        return max(64, min(4096, int(value)))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_TOKENS


def configure_vision_engine(engine: Any, settings: Any) -> None:
    """Bind UI/settings max_tokens to the existing vision engine instance.

    The engine's public analyze_image() API keeps its optional argument for
    programmatic callers, while the application pipeline gets its default from
    persisted UI settings.
    """
    engine._aimosaic_max_tokens = _coerce_max_tokens(
        getattr(settings, "max_tokens", _DEFAULT_MAX_TOKENS)
    )
    if getattr(engine, "_aimosaic_settings_wrapper_installed", False):
        return

    original_analyze_image = engine.analyze_image

    @wraps(original_analyze_image)
    def analyze_image(image_path: str, max_tokens: int | None = None, temperature: float = 0.0):
        effective = (
            engine._aimosaic_max_tokens
            if max_tokens is None
            else _coerce_max_tokens(max_tokens)
        )
        return original_analyze_image(
            image_path,
            max_tokens=effective,
            temperature=temperature,
        )

    engine.analyze_image = analyze_image
    engine._aimosaic_settings_wrapper_installed = True
