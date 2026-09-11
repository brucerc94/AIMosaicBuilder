"""
Core data models for AI Mosaic Builder.
All entities are plain dataclasses — no ORM, no database.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ImageStatus(str, Enum):
    PENDING = "pending"
    PREFILTERED = "prefiltered"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    SELECTED = "selected"
    REJECTED = "rejected"
    ERROR = "error"
    CACHED = "cached"


class RejectReason(str, Enum):
    NONE = ""
    NO_PERSON = "no_person"
    BLURRY = "blurry"
    TOO_SMALL = "too_small"
    CORRUPT = "corrupt"
    LOW_QUALITY = "low_quality"
    SIMILAR = "similar"
    OCCLUDED = "occluded"
    PERSON_TOO_SMALL = "person_too_small"
    MANUAL = "manual"
    LLM_ERROR = "llm_error"


class SortOrder(str, Enum):
    SCORE = "score"
    RANK = "rank"
    FILENAME = "filename"
    DATE = "date"


@dataclass
class MosaicRequirements:
    """User constraints and soft preferences for the final mosaic."""

    min_face_only: int = 0
    min_full_body: int = 0
    min_front: int = 0
    min_side: int = 0
    min_back: int = 0
    min_male: int = 0
    min_female: int = 0
    min_face_visible: int = 0
    min_body_visible: int = 0

    nsfw_policy: str = "any"
    exclude_blurry: bool = False
    exclude_occluded: bool = False
    min_quality: float = 0.0
    min_person_visibility: float = 0.0

    prefer_face_only: bool = False
    prefer_full_body: bool = False
    prefer_front: bool = False
    prefer_side: bool = False
    prefer_back: bool = False
    prefer_solo: bool = False
    prefer_face_visible: bool = False
    prefer_body_visible: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MosaicRequirements":
        if not isinstance(d, dict):
            return cls()
        kwargs: dict[str, Any] = {}
        int_fields = (
            "min_face_only", "min_full_body", "min_front", "min_side", "min_back",
            "min_male", "min_female", "min_face_visible", "min_body_visible",
        )
        for name in int_fields:
            try:
                kwargs[name] = max(0, int(d.get(name, 0)))
            except (TypeError, ValueError):
                kwargs[name] = 0
        bool_fields = (
            "exclude_blurry", "exclude_occluded", "prefer_face_only", "prefer_full_body",
            "prefer_front", "prefer_side", "prefer_back", "prefer_solo",
            "prefer_face_visible", "prefer_body_visible",
        )
        for name in bool_fields:
            kwargs[name] = bool(d.get(name, False))
        for name in ("min_quality", "min_person_visibility"):
            try:
                kwargs[name] = max(0.0, min(1.0, float(d.get(name, 0.0))))
            except (TypeError, ValueError):
                kwargs[name] = 0.0
        policy = str(d.get("nsfw_policy", "any"))
        kwargs["nsfw_policy"] = policy if policy in {"any", "safe_only", "nsfw_only"} else "any"
        return cls(**kwargs)


@dataclass
class ImageAnalysis:
    has_person: bool = False
    person_count: int = 0
    main_subject_is_person: bool = False
    person_visibility: float = 0.0
    face_visible: bool = False
    body_visible: bool = False
    occluded: bool = False
    blur: float = 0.0
    composition: float = 0.0
    image_quality: float = 0.0
    subject_quality: float = 0.0
    mosaic_value: float = 0.0
    visual_tags: dict[str, Any] = field(default_factory=dict)
    reject: bool = False
    reject_reason: str = ""
    notes: str = ""
    raw_response: str = ""
    analysis_version: int = 2
    model_used: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ImageAnalysis":
        if not isinstance(d, dict):
            raise ValueError("analysis must be an object")
        bool_fields = (
            "has_person", "main_subject_is_person", "face_visible", "body_visible",
            "occluded", "reject",
        )
        for field_name in bool_fields:
            if field_name in d and not isinstance(d[field_name], bool):
                raise ValueError(f"{field_name} must be boolean")
        numeric_fields = (
            "person_visibility", "blur", "composition", "image_quality", "subject_quality", "mosaic_value"
        )
        values: dict[str, Any] = {}
        for field_name in numeric_fields:
            value = float(d.get(field_name, 0.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in 0..1")
            values[field_name] = value
        person_count = int(d.get("person_count", 0))
        if person_count < 0 or person_count > 99:
            raise ValueError("person_count out of range")
        visual_tags = d.get("visual_tags", {})
        if not isinstance(visual_tags, dict):
            raise ValueError("visual_tags must be an object")
        return cls(
            has_person=bool(d.get("has_person", False)),
            person_count=person_count,
            main_subject_is_person=bool(d.get("main_subject_is_person", False)),
            **values,
            face_visible=bool(d.get("face_visible", False)),
            body_visible=bool(d.get("body_visible", False)),
            occluded=bool(d.get("occluded", False)),
            visual_tags=dict(visual_tags),
            reject=bool(d.get("reject", False)),
            reject_reason=str(d.get("reject_reason", "")),
            notes=str(d.get("notes", ""))[:1000],
            raw_response=str(d.get("raw_response", "")),
            analysis_version=int(d.get("analysis_version", 2)),
            model_used=str(d.get("model_used", "")),
            timestamp=str(d.get("timestamp", datetime.now().isoformat())),
        )

    @classmethod
    def error_result(cls, reason: str, model: str = "") -> "ImageAnalysis":
        return cls(reject=True, reject_reason=RejectReason.LLM_ERROR.value, notes=reason, model_used=model)


@dataclass
class BoundingBox:
    x: int = 0
    y: int = 0
    width: int = 0
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
        return cls(x=int(d.get("x", 0)), y=int(d.get("y", 0)), width=int(d.get("width", 0)), height=int(d.get("height", 0)))

    def padded(self, pad_x: int, pad_y: int, img_w: int, img_h: int) -> "BoundingBox":
        x = max(0, self.x - pad_x)
        y = max(0, self.y - pad_y)
        x2 = min(img_w, self.x2 + pad_x)
        y2 = min(img_h, self.y2 + pad_y)
        return BoundingBox(x=x, y=y, width=x2 - x, height=y2 - y)

    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0


@dataclass
class PersonDetection:
    bbox: BoundingBox = field(default_factory=BoundingBox)
    confidence: float = 0.0
    relative_size: float = 0.0
    is_main: bool = False
    face_visible: bool = False
    body_visible: bool = False
    position: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.to_dict(),
            "confidence": self.confidence,
            "relative_size": self.relative_size,
            "is_main": self.is_main,
            "face_visible": self.face_visible,
            "body_visible": self.body_visible,
            "position": self.position,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PersonDetection":
        return cls(
            bbox=BoundingBox.from_dict(d.get("bbox", {})),
            confidence=float(d.get("confidence", 0.0)),
            relative_size=float(d.get("relative_size", 0.0)),
            is_main=bool(d.get("is_main", False)),
            face_visible=bool(d.get("face_visible", False)),
            body_visible=bool(d.get("body_visible", False)),
            position=str(d.get("position", "")),
            source=str(d.get("source", "")),
        )


@dataclass
class RankingWeights:
    technical_quality: float = 0.20
    composition: float = 0.15
    person_visibility: float = 0.20
    subject_quality: float = 0.15
    sharpness: float = 0.15
    face_visibility: float = 0.10
    mosaic_value: float = 0.05

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RankingWeights":
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RankingResult:
    final_score: float = 0.0
    rank: int = 0
    score_breakdown: dict = field(default_factory=dict)
    penalty_applied: float = 0.0
    penalty_reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RankingResult":
        return cls(
            final_score=float(d.get("final_score", 0.0)),
            rank=int(d.get("rank", 0)),
            score_breakdown=dict(d.get("score_breakdown", {})),
            penalty_applied=float(d.get("penalty_applied", 0.0)),
            penalty_reasons=list(d.get("penalty_reasons", [])),
        )


@dataclass
class MosaicSelection:
    image_path: str = ""
    slot_index: int = 0
    crop_bbox: Optional[BoundingBox] = None
    padding_px: int = 40
    manual_override: bool = False
    zoom: float = 0.5

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "slot_index": self.slot_index,
            "crop_bbox": self.crop_bbox.to_dict() if self.crop_bbox else None,
            "padding_px": self.padding_px,
            "manual_override": self.manual_override,
            "zoom": self.zoom,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MosaicSelection":
        cb = d.get("crop_bbox")
        return cls(
            image_path=str(d.get("image_path", "")),
            slot_index=int(d.get("slot_index", 0)),
            crop_bbox=BoundingBox.from_dict(cb) if cb else None,
            padding_px=int(d.get("padding_px", 40)),
            manual_override=bool(d.get("manual_override", False)),
            zoom=float(d.get("zoom", 0.5)),
        )


@dataclass
class ImageRecord:
    path: str = ""
    file_hash: str = ""
    filename: str = ""
    width: int = 0
    height: int = 0
    file_size: int = 0
    status: ImageStatus = ImageStatus.PENDING
    analysis: Optional[ImageAnalysis] = None
    detections: list[PersonDetection] = field(default_factory=list)
    ranking: Optional[RankingResult] = None
    selection: Optional[MosaicSelection] = None
    phash: str = ""
    thumbnail_path: str = ""
    error_message: str = ""
    manually_included: bool = False
    manually_excluded: bool = False

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "file_hash": self.file_hash,
            "filename": self.filename,
            "width": self.width,
            "height": self.height,
            "file_size": self.file_size,
            "status": self.status.value,
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "detections": [d.to_dict() for d in self.detections],
            "ranking": self.ranking.to_dict() if self.ranking else None,
            "selection": self.selection.to_dict() if self.selection else None,
            "phash": self.phash,
            "thumbnail_path": self.thumbnail_path,
            "error_message": self.error_message,
            "manually_included": self.manually_included,
            "manually_excluded": self.manually_excluded,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImageRecord":
        analysis_d = d.get("analysis")
        ranking_d = d.get("ranking")
        selection_d = d.get("selection")
        return cls(
            path=str(d.get("path", "")),
            file_hash=str(d.get("file_hash", "")),
            filename=str(d.get("filename", "")),
            width=int(d.get("width", 0)),
            height=int(d.get("height", 0)),
            file_size=int(d.get("file_size", 0)),
            status=ImageStatus(d.get("status", ImageStatus.PENDING.value)),
            analysis=ImageAnalysis.from_dict(analysis_d) if analysis_d else None,
            detections=[PersonDetection.from_dict(x) for x in d.get("detections", [])],
            ranking=RankingResult.from_dict(ranking_d) if ranking_d else None,
            selection=MosaicSelection.from_dict(selection_d) if selection_d else None,
            phash=str(d.get("phash", "")),
            thumbnail_path=str(d.get("thumbnail_path", "")),
            error_message=str(d.get("error_message", "")),
            manually_included=bool(d.get("manually_included", False)),
            manually_excluded=bool(d.get("manually_excluded", False)),
        )


@dataclass
class AppSettings:
    last_source_folder: str = ""
    model_path: str = ""
    mmproj_path: str = ""
    cache_directory: str = ""
    n_ctx: int = 4096
    max_tokens: int = 576
    n_gpu_layers: int = 0
    n_threads: int = 4
    n_threads_batch: int = 0
    n_batch: int = 512
    n_ubatch: int = 512
    target_images: int = 0
    canvas_width: int = 3840
    canvas_height: int = 2160
    padding_px: int = 40
    min_subject_percent: float = 10.0
    target_subject_percent: float = 15.0
    min_image_width: int = 200
    min_image_height: int = 200
    confidence_threshold: float = 0.5
    ranking_weights: RankingWeights = field(default_factory=RankingWeights)
    phash_threshold: int = 10
    sort_order: str = SortOrder.SCORE.value
    debug_mode: bool = False
    person_detector_model: str = "yolo26n.pt"
    person_detector_confidence: float = 0.25
    mosaic_requirements: MosaicRequirements = field(default_factory=MosaicRequirements)

    def to_dict(self) -> dict:
        return {
            "last_source_folder": self.last_source_folder,
            "model_path": self.model_path,
            "mmproj_path": self.mmproj_path,
            "cache_directory": self.cache_directory,
            "n_ctx": self.n_ctx,
            "max_tokens": self.max_tokens,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
            "n_threads_batch": self.n_threads_batch,
            "n_batch": self.n_batch,
            "n_ubatch": self.n_ubatch,
            "target_images": self.target_images,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "padding_px": self.padding_px,
            "min_subject_percent": self.min_subject_percent,
            "target_subject_percent": self.target_subject_percent,
            "min_image_width": self.min_image_width,
            "min_image_height": self.min_image_height,
            "confidence_threshold": self.confidence_threshold,
            "ranking_weights": self.ranking_weights.to_dict(),
            "phash_threshold": self.phash_threshold,
            "sort_order": self.sort_order,
            "debug_mode": self.debug_mode,
            "person_detector_model": self.person_detector_model,
            "person_detector_confidence": self.person_detector_confidence,
            "mosaic_requirements": self.mosaic_requirements.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AppSettings":
        weights_d = d.get("ranking_weights", {})
        try:
            n_ctx = int(d.get("n_ctx", 4096))
        except (TypeError, ValueError):
            n_ctx = 4096
        try:
            max_tokens = int(d.get("max_tokens", 576))
        except (TypeError, ValueError):
            max_tokens = 576
        try:
            min_subject_percent = float(d.get("min_subject_percent", 10.0))
        except (TypeError, ValueError):
            min_subject_percent = 10.0
        try:
            target_subject_percent = float(d.get("target_subject_percent", 15.0))
        except (TypeError, ValueError):
            target_subject_percent = 15.0
        return cls(
            last_source_folder=str(d.get("last_source_folder", "")),
            model_path=str(d.get("model_path", "")),
            mmproj_path=str(d.get("mmproj_path", "")),
            cache_directory=str(d.get("cache_directory", "")),
            n_ctx=max(2048, n_ctx),
            max_tokens=max(64, min(4096, max_tokens)),
            n_gpu_layers=int(d.get("n_gpu_layers", 0)),
            n_threads=int(d.get("n_threads", 4)),
            n_threads_batch=int(d.get("n_threads_batch", 0)),
            n_batch=int(d.get("n_batch", 512)),
            n_ubatch=int(d.get("n_ubatch", 512)),
            target_images=max(0, int(d.get("target_images", 0))),
            canvas_width=int(d.get("canvas_width", 3840)),
            canvas_height=int(d.get("canvas_height", 2160)),
            padding_px=int(d.get("padding_px", 40)),
            min_subject_percent=max(1.0, min(50.0, min_subject_percent)),
            target_subject_percent=max(max(1.0, min(50.0, min_subject_percent)), min(75.0, target_subject_percent)),
            min_image_width=int(d.get("min_image_width", 200)),
            min_image_height=int(d.get("min_image_height", 200)),
            confidence_threshold=float(d.get("confidence_threshold", 0.5)),
            ranking_weights=RankingWeights.from_dict(weights_d) if weights_d else RankingWeights(),
            phash_threshold=int(d.get("phash_threshold", 10)),
            sort_order=str(d.get("sort_order", SortOrder.SCORE.value)),
            debug_mode=bool(d.get("debug_mode", False)),
            person_detector_model=str(d.get("person_detector_model", "yolo26n.pt")),
            person_detector_confidence=float(d.get("person_detector_confidence", 0.25)),
            mosaic_requirements=MosaicRequirements.from_dict(d.get("mosaic_requirements", {})),
        )
