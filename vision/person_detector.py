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

    @staticmethod
    def _create_hog(cv2):
        """Create HOG across OpenCV Python builds that expose either API."""
        factory = getattr(cv2, "HOGDescriptor_create", None)
        if callable(factory):
            try:
                return factory()
            except Exception as exc:
                raise RuntimeError(f"HOGDescriptor_create() failed: {exc}") from exc

        hog_cls = getattr(cv2, "HOGDescriptor", None)
        if callable(hog_cls):
            try:
                return hog_cls()
            except Exception as exc:
                raise RuntimeError(f"HOGDescriptor() failed: {exc}") from exc

        version = getattr(cv2, "__version__", "unknown")
        module_path = getattr(cv2, "__file__", "unknown")
        raise RuntimeError(
            "This OpenCV build does not expose HOGDescriptor or HOGDescriptor_create. "
            f"cv2 version={version}; module={module_path}. "
            "Check that a standard opencv-python package is installed and that no local "
            "cv2.py/cv2 package is shadowing it."
        )

    @staticmethod
    def _default_people_detector(cv2):
        detector_factory = getattr(cv2, "HOGDescriptor_getDefaultPeopleDetector", None)
        if not callable(detector_factory):
            version = getattr(cv2, "__version__", "unknown")
            module_path = getattr(cv2, "__file__", "unknown")
            raise RuntimeError(
                "This OpenCV build does not expose the default HOG people detector. "
                f"cv2 version={version}; module={module_path}."
            )
        try:
            return detector_factory()
        except Exception as exc:
            raise RuntimeError(f"Default HOG people detector failed: {exc}") from exc

    def detect(self, image_path: str, analysis: Optional[ImageAnalysis] = None) -> list[PersonDetection]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install opencv-python for real person detection.") from exc

        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Cannot decode image: {image_path}")
        image_h, image_w = image.shape[:2]

        hog = self._create_hog(cv2)
        hog.setSVMDetector(self._default_people_detector(cv2))
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
        import cv2
        version = getattr(cv2, "__version__", "unknown")
        module_path = getattr(cv2, "__file__", "unknown")
        logger.info("OpenCV detected: version=%s module=%s", version, module_path)

        detector = OpenCVHOGPersonDetector()
        # Validate the required symbols now, so UI can report a useful error before a batch starts.
        detector._create_hog(cv2)
        detector._default_people_detector(cv2)
        logger.info("OpenCV HOG person detector ready")
        return detector
    except (ImportError, RuntimeError) as exc:
        logger.error("OpenCV person detector unavailable: %s", exc)
        return UnavailablePersonDetector()


def detect_persons(
    image_path: str,
    analysis: Optional[ImageAnalysis] = None,
    backend: Optional[BasePersonDetector] = None,
) -> list[PersonDetection]:
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
    return max(
        detections,
        key=lambda d: (d.confidence, d.relative_size, d.face_visible, d.body_visible, d.is_main),
    )
