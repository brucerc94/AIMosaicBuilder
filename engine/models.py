"""
Core data models for AI Mosaic Builder.
All entities are plain dataclasses — no ORM, no database.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ─── Enums ────────────────────────────────────────────────────────────────────

class ImageStatus(str, Enum):
    """Processing status of a single image."""
    PENDING     = "pending"      # Not yet processed
    PREFILTERED = "prefiltered"  # Passed basic checks, awaiting LLM
    ANALYZING   = "analyzing"    # Currently being analyzed by LLM
    ANALYZED    = "analyzed"     # Analysis complete
    SELECTED    = "selected"     # Chosen for the mosaic
    REJECTED    = "rejected"     # Excluded (low quality, no person, etc.)
    ERROR       = "error"        # Could not process
    CACHED      = "cached"       # Result loaded from cache


class RejectReason(str, Enum):
    NONE            = ""
    NO_PERSON       = "no_person"
    BLURRY          = "blurry"
    TOO_SMALL       = "too_small"
    CORRUPT         = "corrupt"
    LOW_QUALITY     = "low_quality"
    SIMILAR         = "similar"
    OCCLUDED        = "occluded"
    PERSON_TOO_SMALL = "person_too_small"
    MANUAL          = "manual"
    LLM_ERROR       = "llm_error"


class SortOrder(str, Enum):
    SCORE     = "score"
    RANK      = "rank"
    FILENAME  = "filename"
    DATE      = "date"


# ─── ImageAnalysis (from LLM) ─────────────────────────────────────────────────

@dataclass
class ImageAnalysis:
    """
    Structured result from the vision LLM analysis of a single image.
    All float fields are 0.0–1.0 unless noted.
    """
    has_person:              bool    = False
    person_count:            int     = 0
    main_subject_is_person:  bool    = False
    person_visibility:       float   = 0.0   # 0=hidden, 1=fully visible
    face_visible:            bool    = False
    body_visible:            bool    = False
    occluded:                bool    = False
    blur:                    float   = 0.0   # 0=sharp, 1=very blurry
    composition:             float   = 0.0   # photographic quality of composition
    image_quality:           float   = 0.0   # overall technical quality
    subject_quality:         float   = 0.0   # quality of the main subject
    mosaic_value:            float   = 0.0   # how well it would look in a mosaic
    reject:                  bool    = False
    reject_reason:           str     = ""
    notes:                   str     = ""
    raw_response:            str     = ""    # full LLM text for debugging
    analysis_version:        int     = 1
    model_used:              str     = ""
    timestamp:               str     = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "has_person":              self.has_person,
            "person_count":            self.person_count,
            "main_subject_is_person":  self.main_subject_is_person,
            "person_visibility":       self.person_visibility,
            "face_visible":            self.face_visible,
            "body_visible":            self.body_visible,
            "occluded":                self.occluded,
            "blur":                    self.blur,
            "composition":             self.composition,
            "image_quality":           self.image_quality,
            "subject_quality":         self.subject_quality,
            "mosaic_value":            self.mosaic_value,
            "reject":                  self.reject,
            "reject_reason":           self.reject_reason,
            "notes":                   self.notes,
            "raw_response":            self.raw_response,
            "analysis_version":        self.analysis_version,
            "model_used":              self.model_used,
            "timestamp":               self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImageAnalysis":
        return cls(
            has_person              = bool(d.get("has_person", False)),
            person_count            = int(d.get("person_count", 0)),
            main_subject_is_person  = bool(d.get("main_subject_is_person", False)),
            person_visibility       = float(d.get("person_visibility", 0.0)),
            face_visible            = bool(d.get("face_visible", False)),
            body_visible            = bool(d.get("body_visible", False)),
            occluded                = bool(d.get("occluded", False)),
            blur                    = float(d.get("blur", 0.0)),
            composition             = float(d.get("composition", 0.0)),
            image_quality           = float(d.get("image_quality", 0.0)),
            subject_quality         = float(d.get("subject_quality", 0.0)),
            mosaic_value            = float(d.get("mosaic_value", 0.0)),
            reject                  = bool(d.get("reject", False)),
            reject_reason           = str(d.get("reject_reason", "")),
            notes                   = str(d.get("notes", "")),
            raw_response            = str(d.get("raw_response", "")),
            analysis_version        = int(d.get("analysis_version", 1)),
            model_used              = str(d.get("model_used", "")),
            timestamp               = str(d.get("timestamp", datetime.now().isoformat())),
        )

    @classmethod
    def error_result(cls, reason: str, model: str = "") -> "ImageAnalysis":
        return cls(
            reject=True,
            reject_reason=RejectReason.LLM_ERROR.value,
            notes=reason,
            model_used=model,
        )


# ─── PersonDetection (from vision/person_detector.py) ─────────────────────────

@dataclass
class BoundingBox:
    """Pixel-space bounding box."""
    x:      int = 0
    y:      int = 0
    width:  int = 0
    height: int = 0

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, d: dict) -> "BoundingBox":
        return cls(
            x=int(d.get("x", 0)),
            y=int(d.get("y", 0)),
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
        )

    def padded(self, pad_x: int, pad_y: int, img_w: int, img_h: int) -> "BoundingBox":
        """Return a new BoundingBox with symmetric padding, clamped to image bounds."""
        x = max(0, self.x - pad_x)
        y = max(0, self.y - pad_y)
        x2 = min(img_w, self.x2 + pad_x)
        y2 = min(img_h, self.y2 + pad_y)
        return BoundingBox(x=x, y=y, width=x2 - x, height=y2 - y)

    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0


@dataclass
class PersonDetection:
    """
    Detection result for a single person in an image.
    Backend-agnostic — could come from LLM, OpenCV, YOLO, or hybrid.
    """
    bbox:           BoundingBox    = field(default_factory=BoundingBox)
    confidence:     float          = 0.0   # 0–1
    relative_size:  float          = 0.0   # fraction of image area occupied
    is_main:        bool           = False
    face_visible:   bool           = False
    body_visible:   bool           = False
    position:       str            = ""    # "center", "left", "right", "top", "bottom"
    source:         str            = ""    # "llm", "opencv", "yolo", "manual"

    def to_dict(self) -> dict:
        return {
            "bbox":          self.bbox.to_dict(),
            "confidence":    self.confidence,
            "relative_size": self.relative_size,
            "is_main":       self.is_main,
            "face_visible":  self.face_visible,
            "body_visible":  self.body_visible,
            "position":      self.position,
            "source":        self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PersonDetection":
        return cls(
            bbox          = BoundingBox.from_dict(d.get("bbox", {})),
            confidence    = float(d.get("confidence", 0.0)),
            relative_size = float(d.get("relative_size", 0.0)),
            is_main       = bool(d.get("is_main", False)),
            face_visible  = bool(d.get("face_visible", False)),
            body_visible  = bool(d.get("body_visible", False)),
            position      = str(d.get("position", "")),
            source        = str(d.get("source", "")),
        )


# ─── RankingResult ────────────────────────────────────────────────────────────

@dataclass
class RankingWeights:
    """Configurable weights for the final score calculation."""
    technical_quality:  float = 0.20
    composition:        float = 0.15
    person_visibility:  float = 0.20
    subject_quality:    float = 0.15
    sharpness:          float = 0.15   # (1 - blur)
    face_visibility:    float = 0.10
    mosaic_value:       float = 0.05

    def to_dict(self) -> dict:
        return {
            "technical_quality": self.technical_quality,
            "composition":       self.composition,
            "person_visibility": self.person_visibility,
            "subject_quality":   self.subject_quality,
            "sharpness":         self.sharpness,
            "face_visibility":   self.face_visibility,
            "mosaic_value":      self.mosaic_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RankingWeights":
        return cls(**{k: float(v) for k, v in d.items() if hasattr(cls, k)})


@dataclass
class RankingResult:
    """Final ranking output for one image."""
    final_score:        float = 0.0   # 0–100
    rank:               int   = 0
    score_breakdown:    dict  = field(default_factory=dict)
    penalty_applied:    float = 0.0
    penalty_reasons:    list  = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "final_score":     self.final_score,
            "rank":            self.rank,
            "score_breakdown": self.score_breakdown,
            "penalty_applied": self.penalty_applied,
            "penalty_reasons": self.penalty_reasons,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RankingResult":
        return cls(
            final_score     = float(d.get("final_score", 0.0)),
            rank            = int(d.get("rank", 0)),
            score_breakdown = dict(d.get("score_breakdown", {})),
            penalty_applied = float(d.get("penalty_applied", 0.0)),
            penalty_reasons = list(d.get("penalty_reasons", [])),
        )


# ─── MosaicSelection ──────────────────────────────────────────────────────────

@dataclass
class MosaicSelection:
    """Final selection metadata for the mosaic export."""
    image_path:       str  = ""
    slot_index:       int  = 0
    crop_bbox:        Optional[BoundingBox] = None
    padding_px:       int  = 40
    manual_override:  bool = False

    def to_dict(self) -> dict:
        return {
            "image_path":      self.image_path,
            "slot_index":      self.slot_index,
            "crop_bbox":       self.crop_bbox.to_dict() if self.crop_bbox else None,
            "padding_px":      self.padding_px,
            "manual_override": self.manual_override,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MosaicSelection":
        cb = d.get("crop_bbox")
        return cls(
            image_path      = str(d.get("image_path", "")),
            slot_index      = int(d.get("slot_index", 0)),
            crop_bbox       = BoundingBox.from_dict(cb) if cb else None,
            padding_px      = int(d.get("padding_px", 40)),
            manual_override = bool(d.get("manual_override", False)),
        )


# ─── ImageRecord (central entity) ─────────────────────────────────────────────

@dataclass
class ImageRecord:
    """
    Central entity tracking one image through the entire pipeline.
    """
    path:             str                     = ""
    file_hash:        str                     = ""
    filename:         str                     = ""
    width:            int                     = 0
    height:           int                     = 0
    file_size:        int                     = 0
    status:           ImageStatus             = ImageStatus.PENDING
    analysis:         Optional[ImageAnalysis] = None
    detections:       list[PersonDetection]   = field(default_factory=list)
    ranking:          Optional[RankingResult] = None
    selection:        Optional[MosaicSelection] = None
    phash:            str                     = ""   # perceptual hash for dedup
    thumbnail_path:   str                     = ""
    error_message:    str                     = ""
    # Manual user overrides
    manually_included: bool                  = False
    manually_excluded: bool                  = False

    def to_dict(self) -> dict:
        return {
            "path":               self.path,
            "file_hash":          self.file_hash,
            "filename":           self.filename,
            "width":              self.width,
            "height":             self.height,
            "file_size":          self.file_size,
            "status":             self.status.value,
            "analysis":           self.analysis.to_dict() if self.analysis else None,
            "detections":         [d.to_dict() for d in self.detections],
            "ranking":            self.ranking.to_dict() if self.ranking else None,
            "selection":          self.selection.to_dict() if self.selection else None,
            "phash":              self.phash,
            "thumbnail_path":     self.thumbnail_path,
            "error_message":      self.error_message,
            "manually_included":  self.manually_included,
            "manually_excluded":  self.manually_excluded,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImageRecord":
        analysis_d = d.get("analysis")
        ranking_d  = d.get("ranking")
        selection_d = d.get("selection")
        return cls(
            path              = str(d.get("path", "")),
            file_hash         = str(d.get("file_hash", "")),
            filename          = str(d.get("filename", "")),
            width             = int(d.get("width", 0)),
            height            = int(d.get("height", 0)),
            file_size         = int(d.get("file_size", 0)),
            status            = ImageStatus(d.get("status", ImageStatus.PENDING.value)),
            analysis          = ImageAnalysis.from_dict(analysis_d) if analysis_d else None,
            detections        = [PersonDetection.from_dict(x) for x in d.get("detections", [])],
            ranking           = RankingResult.from_dict(ranking_d) if ranking_d else None,
            selection         = MosaicSelection.from_dict(selection_d) if selection_d else None,
            phash             = str(d.get("phash", "")),
            thumbnail_path    = str(d.get("thumbnail_path", "")),
            error_message     = str(d.get("error_message", "")),
            manually_included = bool(d.get("manually_included", False)),
            manually_excluded = bool(d.get("manually_excluded", False)),
        )


# ─── AppSettings ──────────────────────────────────────────────────────────────

@dataclass
class AppSettings:
    """All user-configurable settings, persisted to disk."""
    # Paths
    last_source_folder:  str   = ""
    model_path:          str   = ""
    mmproj_path:         str   = ""
    cache_directory:     str   = ""

    # Model parameters
    n_ctx:               int   = 4096
    n_gpu_layers:        int   = 0
    n_threads:           int   = 4
    n_threads_batch:     int   = 0

    # Mosaic parameters
    target_images:       int   = 12    # 0 = automatic
    canvas_width:        int   = 3840
    canvas_height:       int   = 2160
    padding_px:          int   = 40

    # Analysis thresholds
    min_image_width:     int   = 200
    min_image_height:    int   = 200
    confidence_threshold: float = 0.5

    # Ranking weights
    ranking_weights:     RankingWeights = field(default_factory=RankingWeights)

    # Similarity
    phash_threshold:     int   = 10    # max hamming distance for "similar"

    # UI
    sort_order:          str   = SortOrder.SCORE.value
    debug_mode:          bool  = False

    def to_dict(self) -> dict:
        return {
            "last_source_folder":  self.last_source_folder,
            "model_path":          self.model_path,
            "mmproj_path":         self.mmproj_path,
            "cache_directory":     self.cache_directory,
            "n_ctx":               self.n_ctx,
            "n_gpu_layers":        self.n_gpu_layers,
            "n_threads":           self.n_threads,
            "n_threads_batch":     self.n_threads_batch,
            "target_images":       self.target_images,
            "canvas_width":        self.canvas_width,
            "canvas_height":       self.canvas_height,
            "padding_px":          self.padding_px,
            "min_image_width":     self.min_image_width,
            "min_image_height":    self.min_image_height,
            "confidence_threshold": self.confidence_threshold,
            "ranking_weights":     self.ranking_weights.to_dict(),
            "phash_threshold":     self.phash_threshold,
            "sort_order":          self.sort_order,
            "debug_mode":          self.debug_mode,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AppSettings":
        weights_d = d.get("ranking_weights", {})
        s = cls(
            last_source_folder  = str(d.get("last_source_folder", "")),
            model_path          = str(d.get("model_path", "")),
            mmproj_path         = str(d.get("mmproj_path", "")),
            cache_directory     = str(d.get("cache_directory", "")),
            n_ctx               = int(d.get("n_ctx", 4096)),
            n_gpu_layers        = int(d.get("n_gpu_layers", 0)),
            n_threads           = int(d.get("n_threads", 4)),
            n_threads_batch     = int(d.get("n_threads_batch", 0)),
            target_images       = int(d.get("target_images", 12)),
            canvas_width        = int(d.get("canvas_width", 3840)),
            canvas_height       = int(d.get("canvas_height", 2160)),
            padding_px          = int(d.get("padding_px", 40)),
            min_image_width     = int(d.get("min_image_width", 200)),
            min_image_height    = int(d.get("min_image_height", 200)),
            confidence_threshold = float(d.get("confidence_threshold", 0.5)),
            ranking_weights     = RankingWeights.from_dict(weights_d) if weights_d else RankingWeights(),
            phash_threshold     = int(d.get("phash_threshold", 10)),
            sort_order          = str(d.get("sort_order", SortOrder.SCORE.value)),
            debug_mode          = bool(d.get("debug_mode", False)),
        )
        return s
