"""Runtime configuration adapters for AI Mosaic Builder."""
from __future__ import annotations

from functools import wraps
from typing import Any

_DEFAULT_MAX_TOKENS = 576

def configure_vision_engine(engine: Any, settings: Any) -> None:
    """Bind persisted UI settings to the existing runtime components."""
    engine._aimosaic_max_tokens = _coerce_max_tokens(
        getattr(settings, "max_tokens", _DEFAULT_MAX_TOKENS)
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
    ):
        effective = (
            engine._aimosaic_max_tokens
            if max_tokens is None
            else _coerce_max_tokens(max_tokens)
        )
        return original_analyze_image(
            image_path,
            max_tokens=effective,
            temperature=temperature,
            custom_prompt=custom_prompt,
        )

    engine.analyze_image = analyze_image
    engine._aimosaic_settings_wrapper_installed = True
