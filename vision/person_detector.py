"""Real person detection backends. No invented bounding boxes."""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Optional

from engine.models import BoundingBox, ImageAnalysis, PersonDetection

logger = logging.getLogger("person_detector")


class BasePersonDetector(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def detect(self, image_path: str, analysis: Optional[ImageAnalysis] = None) -> list[PersonDetection]:
        raise NotImplementedError


class OpenCVHOGPersonDetector(BasePersonDetector):
    @property
    def name(self) -> str:
        return "opencv_hog"

    def detect(self, image_path: str, analysis: Optional[ImageAnalysis] = None) -> list[PersonDetection]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install opencv-python for real person detection.") from exc

        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Cannot decode image: {image_path}")
        image_h, image_w = image.shape[:2]

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rects, weights = hog.detectMultiScale(
            image,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )

        detections: list[PersonDetection] = []
        for index, rect in enumerate(rects):
            x, y, width, height = map(int, rect)
            raw_weight = float(weights[index]) if index < len(weights) else 0.0
            confidence = 1.0 / (1.0 + math.exp(-max(-8.0, min(8.0, raw_weight))))
            relative_size = (width * height) / max(1, image_w * image_h)
            detections.append(
                PersonDetection(
                    bbox=BoundingBox(x, y, width, height),
                    confidence=max(0.0, min(1.0, confidence)),
                    relative_size=max(0.0, min(1.0, relative_size)),
                    position=_position(x + width / 2, y + height / 2, image_w, image_h),
                    source=self.name,
                )
            )

        detections.sort(key=lambda item: (item.relative_size, item.confidence), reverse=True)
        if detections:
            detections[0].is_main = True
        return detections


class UnavailablePersonDetector(BasePersonDetector):
    @property
    def name(self) -> str:
        return "unavailable"

    def detect(self, image_path: str, analysis: Optional[ImageAnalysis] = None) -> list[PersonDetection]:
        raise RuntimeError("No real person detector backend is installed.")


def _position(cx: float, cy: float, width: int, height: int) -> str:
    horizontal = "left" if cx < width * 0.33 else "right" if cx > width * 0.66 else "center"
    vertical = "top" if cy < height * 0.33 else "bottom" if cy > height * 0.66 else ""
    return f"{vertical}_{horizontal}" if vertical and horizontal != "center" else (vertical or horizontal)


def get_default_detector() -> BasePersonDetector:
    try:
        import cv2  # noqa: F401
        return OpenCVHOGPersonDetector()
    except ImportError:
        return UnavailablePersonDetector()


def detect_persons(image_path: str, analysis: Optional[ImageAnalysis] = None, backend: Optional[BasePersonDetector] = None) -> list[PersonDetection]:
    detector = backend or get_default_detector()
    try:
        detections = detector.detect(image_path, analysis)
        logger.debug("%s detected %d person(s) in %s", detector.name, len(detections), image_path)
        return detections
    except Exception as exc:
        logger.error("Detection failed (%s): %s", detector.name, exc)
        return []


def select_main_person(detections: list[PersonDetection]) -> Optional[PersonDetection]:
    if not detections:
        return None
    return max(detections, key=lambda d: (d.confidence, d.relative_size, d.face_visible, d.body_visible, d.is_main))
