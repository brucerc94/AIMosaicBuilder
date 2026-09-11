"""Target-subject-aware scoring hook for the existing layout optimizer."""
from __future__ import annotations

import inspect
from contextvars import ContextVar
from typing import Optional

from engine import layout_optimizer as _optimizer

_ACTIVE_TARGET_SUBJECT_PX: ContextVar[Optional[int]] = ContextVar(
    "aimosaic_target_subject_px",
    default=None,
)
_INSTALLED = False

_ORIGINAL_SIMULATE = _optimizer._simulate_exact_viewer
_ORIGINAL_AUTO = _optimizer.optimize_auto_layout
_ORIGINAL_FIXED = _optimizer.optimize_fixed_layout
_SIMULATE_SIGNATURE = inspect.signature(_ORIGINAL_SIMULATE)


def _target_fit(average_subject_px: float, target_subject_px: int) -> float:
    """Return a smooth 0..1 score for closeness to the requested subject size."""
    target = max(1.0, float(target_subject_px))
    relative_error = abs(float(average_subject_px) - target) / target
    return 1.0 / (1.0 + relative_error)


def _simulate_with_target(*args, **kwargs):
    evaluation = _ORIGINAL_SIMULATE(*args, **kwargs)
    target = _ACTIVE_TARGET_SUBJECT_PX.get()
    if evaluation is None or target is None or target <= 0:
        return evaluation

    bound = _SIMULATE_SIGNATURE.bind_partial(*args, **kwargs)
    canvas_size = tuple(bound.arguments["canvas_size"])
    canvas_w, canvas_h = map(float, canvas_size)
    max_x = max(p.x + p.width for p in evaluation.placements)
    max_y = max(p.y + p.height for p in evaluation.placements)
    extent = max(
        0.0,
        min(
            1.0,
            (max_x / max(1.0, canvas_w)) * (max_y / max(1.0, canvas_h)),
        ),
    )
    readability = max(0.0, min(1.25, evaluation.average_subject_px / 260.0))
    target_fit = _target_fit(evaluation.average_subject_px, target)

    # Preserve the existing layout terms while giving Target Subject a
    # meaningful soft preference. It can influence row count/packing without
    # becoming a hard constraint.
    evaluation.layout_score = (
        0.60 * evaluation.canvas_fill_ratio
        + 0.15 * extent
        + 0.10 * readability
        + 0.15 * target_fit
    )
    return evaluation


def _wrap_optimizer(original):
    signature = inspect.signature(original)

    def wrapped(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        target = int(bound.arguments.get("target_subject_px", 260) or 260)
        token = _ACTIVE_TARGET_SUBJECT_PX.set(max(1, target))
        try:
            return original(*args, **kwargs)
        finally:
            _ACTIVE_TARGET_SUBJECT_PX.reset(token)

    wrapped.__name__ = original.__name__
    wrapped.__doc__ = original.__doc__
    return wrapped


def install_target_subject_bias() -> None:
    """Install the target-subject-aware score hook once."""
    global _INSTALLED
    if _INSTALLED:
        return
    _optimizer._simulate_exact_viewer = _simulate_with_target
    _optimizer.optimize_auto_layout = _wrap_optimizer(_ORIGINAL_AUTO)
    _optimizer.optimize_fixed_layout = _wrap_optimizer(_ORIGINAL_FIXED)
    _INSTALLED = True
