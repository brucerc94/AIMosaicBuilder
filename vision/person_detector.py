"""
Person detection for AI Mosaic Builder.

Provides a backend-agnostic interface for detecting persons in images
and returning bounding boxes. The implementation can be swapped without
touching the rest of the system.

Current implementations:
  - LLMPersonDetector: uses the vision LLM's analysis to estimate bounding boxes
    from relative position/size hints. Less accurate but no extra dependencies.
  - HeuristicPersonDetector: pure Pillow/OpenCV fallback using color/edge analysis.

Future: YOLO, MediaPipe, etc. can be added as new classes implementing
        BasePersonDetector.

The `detect()` function picks the best available backend automatically.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from engine.models import BoundingBox, ImageAnalysis, PersonDetection

logger = logging.getLogger("person_detector")


# ─── Abstract base ────────────────────────────────────────────────────────────

class BasePersonDetector(ABC):
    """Swappable person detection backend."""

    @abstractmethod
    def detect(
        self,
        image_path: str,
        analysis: Optional[ImageAnalysis] = None,
    ) -> list[PersonDetection]:
        """
        Return a list of PersonDetection objects for persons found in the image.
        `analysis` may be provided to give the detector context from the LLM.
        Empty list if no persons found.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name for logging."""
        ...


# ─── LLM-based heuristic detector ────────────────────────────────────────────

class LLMHeuristicDetector(BasePersonDetector):
    """
    Derives an estimated bounding box from the LLM analysis:
    - Uses image dimensions and generic person placement heuristics
    - Returns a centre-weighted box if person_count >= 1
    - No additional ML model needed

    This is a "good enough" fallback. Accuracy is limited because
    the LLM gives qualitative descriptions, not pixel coordinates.
    A future backend (YOLO, MediaPipe) can replace this transparently.
    """

    @property
    def name(self) -> str:
        return "llm_heuristic"

    def detect(
        self,
        image_path: str,
        analysis: Optional[ImageAnalysis] = None,
    ) -> list[PersonDetection]:
        if analysis is None or not analysis.has_person:
            return []

        try:
            from PIL import Image as PILImage
            with PILImage.open(image_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            logger.warning(f"[person_detector] Cannot open {image_path}: {e}")
            img_w, img_h = 1000, 750  # fallback dimensions

        count = max(1, analysis.person_count)
        detections: list[PersonDetection] = []

        for i in range(min(count, 5)):  # cap at 5 persons
            bbox = _estimate_person_bbox(
                img_w, img_h,
                person_index=i,
                person_count=count,
                visibility=analysis.person_visibility,
                face_visible=analysis.face_visible,
            )
            position = _classify_position(bbox, img_w, img_h)
            relative_size = (bbox.width * bbox.height) / max(img_w * img_h, 1)

            is_main = (i == 0)  # first is always main by convention

            det = PersonDetection(
                bbox=bbox,
                confidence=analysis.person_visibility * 0.8,  # heuristic
                relative_size=relative_size,
                is_main=is_main,
                face_visible=analysis.face_visible if is_main else False,
                body_visible=analysis.body_visible if is_main else False,
                position=position,
                source=self.name,
            )
            detections.append(det)

        return detections


def _estimate_person_bbox(
    img_w: int,
    img_h: int,
    person_index: int = 0,
    person_count: int = 1,
    visibility: float = 0.8,
    face_visible: bool = True,
) -> BoundingBox:
    """
    Estimate a plausible bounding box for a person in the image.

    Strategy:
    - Main subject (index 0) → centred, occupying ~60% width, ~85% height
    - Additional subjects → spread horizontally
    - Visibility hints: if low, shrink box and push off-centre
    """
    # Base box dimensions relative to image
    base_w_frac = 0.55 / person_count
    base_w_frac = max(0.2, min(base_w_frac, 0.8))

    if face_visible:
        # Full person likely in frame
        base_h_frac = 0.85
        y_offset_frac = 0.05
    else:
        # Only partial body visible
        base_h_frac = 0.60
        y_offset_frac = 0.20

    # Horizontal placement
    if person_count == 1:
        x_frac = 0.5 - base_w_frac / 2  # centred
    else:
        slot_w = 1.0 / person_count
        x_frac = slot_w * person_index + (slot_w - base_w_frac) / 2
        x_frac = max(0.0, min(x_frac, 1.0 - base_w_frac))

    # Apply visibility: lower visibility → smaller, more off-centre box
    shrink = max(0.0, 1.0 - visibility) * 0.3
    base_w_frac = max(0.15, base_w_frac - shrink)
    base_h_frac = max(0.3, base_h_frac - shrink)

    x = int(x_frac * img_w)
    y = int(y_offset_frac * img_h)
    w = int(base_w_frac * img_w)
    h = int(base_h_frac * img_h)

    # Clamp to image bounds
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    return BoundingBox(x=x, y=y, width=w, height=h)


def _classify_position(bbox: BoundingBox, img_w: int, img_h: int) -> str:
    cx, cy = bbox.center
    cx_frac = cx / img_w
    cy_frac = cy / img_h

    if cx_frac < 0.33:
        h_pos = "left"
    elif cx_frac > 0.66:
        h_pos = "right"
    else:
        h_pos = "center"

    if cy_frac < 0.33:
        v_pos = "top"
    elif cy_frac > 0.66:
        v_pos = "bottom"
    else:
        v_pos = ""

    if v_pos:
        return f"{v_pos}_{h_pos}" if h_pos != "center" else v_pos
    return h_pos


# ─── Main detect function ─────────────────────────────────────────────────────

def detect_persons(
    image_path: str,
    analysis: Optional[ImageAnalysis] = None,
    backend: Optional[BasePersonDetector] = None,
) -> list[PersonDetection]:
    """
    Detect persons in an image. Returns a list of PersonDetection objects.
    Auto-selects the best available backend if none is specified.

    Priority:
      1. `backend` param (explicit choice)
      2. LLMHeuristicDetector (always available)
    """
    if backend is None:
        backend = LLMHeuristicDetector()

    try:
        detections = backend.detect(image_path, analysis=analysis)
        logger.debug(
            f"[person_detector] {backend.name}: "
            f"{len(detections)} person(s) in {Path(image_path).name}"
        )
        return detections
    except Exception as e:
        logger.error(f"[person_detector] Detection failed ({backend.name}): {e}")
        return []


def select_main_person(detections: list[PersonDetection]) -> Optional[PersonDetection]:
    """
    From a list of detections, return the one most likely to be
    the main subject of the photograph.

    Scoring:
      - is_main flag from detector
      - relative size (larger = more important)
      - face visible bonus
      - body visible bonus
      - centre position bonus
    """
    if not detections:
        return None
    if len(detections) == 1:
        return detections[0]

    def _score(d: PersonDetection) -> float:
        s = d.confidence * 0.3
        s += d.relative_size * 0.4
        if d.face_visible:
            s += 0.15
        if d.body_visible:
            s += 0.10
        if d.is_main:
            s += 0.20
        if "center" in d.position:
            s += 0.05
        return s

    return max(detections, key=_score)
