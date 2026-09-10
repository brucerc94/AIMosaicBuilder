"""Real person detection backends.

YOLO is preferred for reliable person bounding boxes. OpenCV HOG remains
available as a lightweight fallback. No fabricated bounding boxes are ever
returned.
"""
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


class YOLOPersonDetector(BasePersonDetector):
    """Ultralytics YOLO detector restricted to the COCO person class."""

    def __init__(self, model_path: str = "yolo26n.pt", confidence: float = 0.25) -> None:
        self.model_path = model_path
        self.confidence = max(0.0, min(1.0, confidence))
        self._model = None

    @property
    def name(self) -> str:
        return "yolo"

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Install the 'ultralytics' package "
                "or select another detector backend."
            ) from exc
        logger.info("Loading YOLO person detector: %s", self.model_path)
        self._model = YOLO(self.model_path)
        return self._model

    def detect(self, image_path: str, analysis: Optional[ImageAnalysis] = None) -> list[PersonDetection]:
        model = self._load()
        try:
            results = model.predict(source=image_path, conf=self.confidence, verbose=False)
        except Exception as exc:
            raise RuntimeError(f"YOLO inference failed: {exc}") from exc

        if not results:
            return []
        result = results[0]
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = boxes.xyxy.cpu().tolist() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.tolist()
        confs = boxes.conf.cpu().tolist() if hasattr(boxes.conf, "cpu") else boxes.conf.tolist()
        classes = boxes.cls.cpu().tolist() if hasattr(boxes.cls, "cpu") else boxes.cls.tolist()

        image_width = image_height = 0
        try:
            from PIL import Image
            with Image.open(image_path) as image:
                image_width, image_height = image.size
        except Exception:
            pass

        detections: list[PersonDetection] = []
        for idx, coords in enumerate(xyxy):
            class_id = int(classes[idx]) if idx < len(classes) else -1
            class_name = str(names.get(class_id, class_id)).lower()
            if class_id != 0 and class_name != "person":
                continue
            x1, y1, x2, y2 = [int(round(value)) for value in coords]
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            if width <= 0 or height <= 0:
                continue
            confidence = float(confs[idx]) if idx < len(confs) else 0.0
            relative_size = (width * height) / float(max(1, image_width * image_height)) if image_width and image_height else 0.0
            detections.append(
                PersonDetection(
                    bbox=BoundingBox(x1, y1, width, height),
                    confidence=max(0.0, min(1.0, confidence)),
                    relative_size=max(0.0, min(1.0, relative_size)),
                    position=_position(x1 + width / 2, y1 + height / 2, image_width, image_height),
                    source=self.name,
                )
            )

        detections.sort(key=lambda item: (item.confidence, item.relative_size), reverse=True)
        if detections:
            detections[0].is_main = True
        return detections


class OpenCVHOGPersonDetector(BasePersonDetector):
    @property
    def name(self) -> str:
        return "opencv_hog"

    @staticmethod
    def _create_hog(cv2):
        factory = getattr(cv2, "HOGDescriptor_create", None)
        if callable(factory):
            return factory()
        hog_cls = getattr(cv2, "HOGDescriptor", None)
        if callable(hog_cls):
            return hog_cls()
        raise RuntimeError("OpenCV HOGDescriptor API is unavailable.")

    @staticmethod
    def _default_people_detector(cv2):
        detector_factory = getattr(cv2, "HOGDescriptor_getDefaultPeopleDetector", None)
        if not callable(detector_factory):
            raise RuntimeError("OpenCV default HOG people detector is unavailable.")
        return detector_factory()

    def detect(self, image_path: str, analysis: Optional[ImageAnalysis] = None) -> list[PersonDetection]:
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Cannot decode image: {image_path}")
        image_h, image_w = image.shape[:2]
        hog = self._create_hog(cv2)
        hog.setSVMDetector(self._default_people_detector(cv2))
        rects, weights = hog.detectMultiScale(image, winStride=(8, 8), padding=(8, 8), scale=1.05)
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
        detections.sort(key=lambda item: (item.confidence, item.relative_size), reverse=True)
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
    if width <= 0 or height <= 0:
        return ""
    horizontal = "left" if cx < width * 0.33 else "right" if cx > width * 0.66 else "center"
    vertical = "top" if cy < height * 0.33 else "bottom" if cy > height * 0.66 else ""
    return f"{vertical}_{horizontal}" if vertical and horizontal != "center" else (vertical or horizontal)


def get_default_detector(model_path: str = "yolo26n.pt", confidence: float = 0.25) -> BasePersonDetector:
    """Prefer YOLO for accurate person boxes; use HOG only as a fallback."""
    try:
        detector = YOLOPersonDetector(model_path=model_path, confidence=confidence)
        detector._load()
        logger.info("YOLO person detector ready")
        return detector
    except Exception as exc:
        logger.warning("YOLO detector unavailable: %s", exc)

    try:
        import cv2
        detector = OpenCVHOGPersonDetector()
        detector._create_hog(cv2)
        detector._default_people_detector(cv2)
        logger.info("OpenCV HOG person detector ready")
        return detector
    except (ImportError, RuntimeError) as exc:
        logger.error("OpenCV fallback unavailable: %s", exc)
        return UnavailablePersonDetector()


def detect_persons(
    image_path: str,
    analysis: Optional[ImageAnalysis] = None,
    backend: Optional[BasePersonDetector] = None,
    model_path: str = "yolo26n.pt",
    confidence: float = 0.25,
) -> list[PersonDetection]:
    detector = backend or get_default_detector(model_path=model_path, confidence=confidence)
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
