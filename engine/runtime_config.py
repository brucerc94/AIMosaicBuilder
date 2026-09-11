"""Runtime configuration adapters for AI Mosaic Builder."""
from __future__ import annotations

import math
from functools import wraps
from typing import Any

_DEFAULT_MAX_TOKENS = 576

# Layout optimizer hook state. The existing optimizer remains the source of
# truth for packing; this adapter only replaces the fixed 260px readability
# preference with the user-configured target subject size.
_layout_patch_installed = False
_original_selected_layout = None
_original_simulate_exact_viewer = None
_active_target_subject_px: int | None = None


def _coerce_max_tokens(value: Any) -> int:
    try:
        return max(64, min(4096, int(value)))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_TOKENS


def _coerce_subject_px(value: Any, fallback: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return fallback


def _install_layout_target_adapter() -> None:
    """Make Target Subject Size affect the existing optimizer's layout score.

    The base optimizer already receives min/target subject pixels from the UI,
    but its final readability score used a fixed 260px reference. We replace
    only that reference at runtime so AUTO/FIXED can keep the same packing
    behavior while selecting the layout closest to the configured target.
    """
    global _layout_patch_installed, _original_selected_layout, _original_simulate_exact_viewer
    if _layout_patch_installed:
        return

    from engine import layout_optimizer

    _original_selected_layout = layout_optimizer._optimize_selected_layout
    _original_simulate_exact_viewer = layout_optimizer._simulate_exact_viewer

    @wraps(_original_simulate_exact_viewer)
    def simulate_exact_viewer(
        order,
        zooms,
        canvas_size,
        padding_px,
        min_zoom,
        zoom_decay,
        min_subject_px,
    ):
        evaluation = _original_simulate_exact_viewer(
            order,
            zooms,
            canvas_size,
            padding_px,
            min_zoom,
            zoom_decay,
            min_subject_px,
        )
        if evaluation is None:
            return None

        target = _active_target_subject_px
        if target is None:
            return evaluation

        average_subject = float(evaluation.average_subject_px)
        deviation = abs(average_subject - target) / max(1.0, float(target))
        target_fit = math.exp(-2.0 * deviation)

        # Preserve the optimizer's fill/extent priorities while making the
        # user target a meaningful part of choosing between candidate layouts.
        fill = float(evaluation.canvas_fill_ratio)
        extent = max(0.0, min(1.0, evaluation.layout_score * 0.0 + (
            max((p.x + p.width) for p in evaluation.placements) / max(1, int(canvas_size[0]))
        ) * (
            max((p.y + p.height) for p in evaluation.placements) / max(1, int(canvas_size[1]))
        )))
        evaluation.layout_score = 0.55 * fill + 0.20 * extent + 0.25 * target_fit
        return evaluation

    @wraps(_original_selected_layout)
    def selected_layout(
        selected,
        canvas_size,
        padding_px,
        min_subject_px,
        target_subject_px,
        max_zoom,
    ):
        global _active_target_subject_px
        previous_target = _active_target_subject_px
        _active_target_subject_px = _coerce_subject_px(target_subject_px, 260)
        try:
            return _original_selected_layout(
                selected,
                canvas_size,
                padding_px,
                min_subject_px,
                target_subject_px,
                max_zoom,
            )
        finally:
            _active_target_subject_px = previous_target

    layout_optimizer._simulate_exact_viewer = simulate_exact_viewer
    layout_optimizer._optimize_selected_layout = selected_layout
    _layout_patch_installed = True


def configure_vision_engine(engine: Any, settings: Any) -> None:
    """Bind persisted UI settings to the existing runtime components."""
    engine._aimosaic_max_tokens = _coerce_max_tokens(
        getattr(settings, "max_tokens", _DEFAULT_MAX_TOKENS)
    )
    _install_layout_target_adapter()

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
