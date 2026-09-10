"""Automatic mosaic layout optimizer compatible with ImageMosaicView's packing behavior."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable

from engine.models import BoundingBox, ImageRecord, ImageStatus, MosaicRequirements, MosaicSelection
from vision.cropper import compute_crop_box

logger = logging.getLogger("layout_optimizer")


@dataclass
class LayoutPlacement:
    record: ImageRecord
    crop_bbox: BoundingBox
    zoom: float
    width: int
    height: int
    x: int
    y: int


@dataclass
class LayoutEvaluation:
    placements: list[LayoutPlacement]
    target: int
    average_zoom: float
    average_fill_ratio: float
    average_subject_px: float
    canvas_fill_ratio: float


def _find_non_overlap_position(canvas_w: int, canvas_h: int, width: int, height: int, occupied: list[tuple[int, int, int, int]], step: int = 10) -> tuple[int, int] | None:
    if width <= 0 or height <= 0 or width > canvas_w or height > canvas_h:
        return None
    for yy in range(0, canvas_h - height + 1, step):
        for xx in range(0, canvas_w - width + 1, step):
            x2 = xx + width
            y2 = yy + height
            if not any(not (x2 <= ox or xx >= ox + ow or y2 <= oy or yy >= oy + oh) for ox, oy, ow, oh in occupied):
                return xx, yy
    return None


def _best_detection(record: ImageRecord):
    return max(record.detections, key=lambda d: (d.is_main, d.confidence, d.relative_size))


def _place(record: ImageRecord, canvas_w: int, canvas_h: int, padding_px: int, occupied: list[tuple[int, int, int, int]], initial_zoom: float, min_zoom: float, zoom_decay: float) -> LayoutPlacement | None:
    if not record.detections or record.width <= 0 or record.height <= 0:
        return None
    detection = _best_detection(record)
    crop = compute_crop_box(detection.bbox, record.width, record.height, padding_px)
    zoom = max(min_zoom, initial_zoom)
    while zoom >= min_zoom:
        width = max(1, int(crop.width * zoom))
        height = max(1, int(crop.height * zoom))
        position = _find_non_overlap_position(canvas_w, canvas_h, width, height, occupied)
        if position is not None:
            return LayoutPlacement(record, crop, zoom, width, height, position[0], position[1])
        zoom *= zoom_decay
    return None


def _hard_eligible(record: ImageRecord, requirements: MosaicRequirements) -> bool:
    analysis = record.analysis
    if not analysis or not analysis.has_person or record.manually_excluded:
        return False
    if analysis.image_quality < requirements.min_quality or analysis.person_visibility < requirements.min_person_visibility:
        return False
    if requirements.exclude_blurry and analysis.blur > 0.5:
        return False
    if requirements.exclude_occluded and analysis.occluded:
        return False
    rating = str(analysis.visual_tags.get("content_rating", "unknown"))
    if requirements.nsfw_policy == "safe_only" and rating != "safe":
        return False
    if requirements.nsfw_policy == "nsfw_only" and rating not in {"suggestive", "explicit"}:
        return False
    return True


def _matches_requirement(record: ImageRecord, name: str) -> bool:
    analysis = record.analysis
    if not analysis:
        return False
    tags = analysis.visual_tags
    return {
        "face_only": tags.get("framing") == "face_only",
        "full_body": tags.get("framing") == "full_body",
        "front": tags.get("orientation") == "front",
        "side": tags.get("orientation") == "side",
        "back": tags.get("orientation") in {"back", "three_quarter_back"},
        "male": tags.get("gender_presentation") == "male",
        "female": tags.get("gender_presentation") == "female",
        "face_visible": analysis.face_visible,
        "body_visible": analysis.body_visible,
    }.get(name, False)


def _required_names(requirements: MosaicRequirements) -> list[str]:
    pairs = (
        ("face_only", requirements.min_face_only), ("full_body", requirements.min_full_body),
        ("front", requirements.min_front), ("side", requirements.min_side), ("back", requirements.min_back),
        ("male", requirements.min_male), ("female", requirements.min_female),
        ("face_visible", requirements.min_face_visible), ("body_visible", requirements.min_body_visible),
    )
    result: list[str] = []
    for name, count in pairs:
        result.extend([name] * max(0, int(count)))
    return result


def _candidate_order(records: Iterable[ImageRecord], requirements: MosaicRequirements) -> list[ImageRecord]:
    candidates = [r for r in records if r.ranking and r.ranking.final_score > 0 and r.detections and _hard_eligible(r, requirements)]
    return sorted(candidates, key=lambda r: (-(r.ranking.final_score if r.ranking else 0.0), r.filename.lower()))


def _evaluate(candidates: list[ImageRecord], canvas_size: tuple[int, int], padding_px: int, target: int, requirements: MosaicRequirements, initial_zoom: float, min_zoom: float, zoom_decay: float, min_subject_px: int) -> LayoutEvaluation:
    canvas_w, canvas_h = map(int, canvas_size)
    occupied: list[tuple[int, int, int, int]] = []
    placements: list[LayoutPlacement] = []
    chosen: list[ImageRecord] = []

    for required_name in _required_names(requirements):
        if len(chosen) >= target:
            break
        matching = [r for r in candidates if r not in chosen and _matches_requirement(r, required_name)]
        for candidate in matching:
            placement = _place(candidate, canvas_w, canvas_h, padding_px, occupied, initial_zoom, min_zoom, zoom_decay)
            if placement is None:
                continue
            subject_px = int(_best_detection(candidate).bbox.height * placement.zoom)
            if subject_px < min_subject_px:
                continue
            chosen.append(candidate)
            placements.append(placement)
            occupied.append((placement.x, placement.y, placement.width, placement.height))
            break

    for candidate in candidates:
        if len(chosen) >= target:
            break
        if candidate in chosen:
            continue
        placement = _place(candidate, canvas_w, canvas_h, padding_px, occupied, initial_zoom, min_zoom, zoom_decay)
        if placement is None:
            continue
        subject_px = int(_best_detection(candidate).bbox.height * placement.zoom)
        if subject_px < min_subject_px:
            continue
        chosen.append(candidate)
        placements.append(placement)
        occupied.append((placement.x, placement.y, placement.width, placement.height))

    canvas_area = max(1, canvas_w * canvas_h)
    occupied_area = sum(p.width * p.height for p in placements)
    return LayoutEvaluation(
        placements=placements,
        target=target,
        average_zoom=sum(p.zoom for p in placements) / len(placements) if placements else 0.0,
        average_fill_ratio=sum((p.width * p.height) / canvas_area for p in placements) / len(placements) if placements else 0.0,
        average_subject_px=(sum(int(_best_detection(p.record).bbox.height * p.zoom) for p in placements) / len(placements) if placements else 0.0),
        canvas_fill_ratio=occupied_area / canvas_area,
    )


def optimize_auto_layout(records: list[ImageRecord], canvas_size: tuple[int, int], padding_px: int, requirements: MosaicRequirements | None = None, max_images: int = 100, initial_zoom: float = 0.5, min_zoom: float = 0.1, zoom_decay: float = 0.9, min_subject_px: int = 120) -> LayoutEvaluation:
    """Choose the largest readable mosaic using ImageMosaicView-like packing."""
    requirements = requirements or MosaicRequirements()
    candidates = _candidate_order(records, requirements)
    limit = min(len(candidates), max(1, int(max_images)))
    best = LayoutEvaluation([], 0, 0.0, 0.0, 0.0, 0.0)
    for target in range(1, limit + 1):
        evaluation = _evaluate(candidates, canvas_size, padding_px, target, requirements, initial_zoom, min_zoom, zoom_decay, min_subject_px)
        if len(evaluation.placements) < target:
            break
        best = evaluation
    logger.info("AUTO layout: selected=%d/%d canvas=%dx%d avg_zoom=%.3f fill=%.1f%% subject=%.0fpx", len(best.placements), len(candidates), canvas_size[0], canvas_size[1], best.average_zoom, best.canvas_fill_ratio * 100.0, best.average_subject_px)
    return best


def apply_layout_selection(evaluation: LayoutEvaluation, all_records: list[ImageRecord]) -> None:
    by_path = {p.record.path: p for p in evaluation.placements}
    placement_index = {p.record.path: index for index, p in enumerate(evaluation.placements)}
    for record in all_records:
        placement = by_path.get(record.path)
        if placement is not None:
            record.status = ImageStatus.SELECTED
            record.selection = MosaicSelection(
                image_path=record.path,
                slot_index=placement_index[record.path],
                crop_bbox=placement.crop_bbox,
                padding_px=0,
                manual_override=record.manually_included,
                zoom=placement.zoom,
            )
        elif record.status == ImageStatus.SELECTED:
            record.status = ImageStatus.REJECTED
