"""Automatic mosaic layout optimization compatible with ImageMosaicView."""
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
    unmet_requirements: list[str]


def _find_non_overlap_position(
    canvas_w: int,
    canvas_h: int,
    width: int,
    height: int,
    occupied: list[tuple[int, int, int, int]],
    step: int = 10,
) -> tuple[int, int] | None:
    """Match ImageMosaicView's 10-pixel top-left packing scan."""
    if width <= 0 or height <= 0 or width > canvas_w or height > canvas_h:
        return None
    for yy in range(0, canvas_h - height + 1, step):
        for xx in range(0, canvas_w - width + 1, step):
            x2 = xx + width
            y2 = yy + height
            if not any(
                not (x2 <= ox or xx >= ox + ow or y2 <= oy or yy >= oy + oh)
                for ox, oy, ow, oh in occupied
            ):
                return xx, yy
    return None


def _best_detection(record: ImageRecord):
    return max(record.detections, key=lambda d: (d.is_main, d.confidence, d.relative_size))


def _crop_for_record(record: ImageRecord, padding_px: int) -> BoundingBox:
    if record.selection and record.selection.crop_bbox:
        return record.selection.crop_bbox
    return compute_crop_box(_best_detection(record).bbox, record.width, record.height, padding_px)


def _place_box(
    record: ImageRecord,
    crop: BoundingBox,
    canvas_w: int,
    canvas_h: int,
    occupied: list[tuple[int, int, int, int]],
    initial_zoom: float,
    min_zoom: float,
    zoom_decay: float,
    min_subject_px: int,
    target_subject_px: int,
    max_zoom: float,
) -> LayoutPlacement | None:
    """Find the largest practical zoom that fits without overlap."""
    detection = _best_detection(record)
    subject_height = max(1, detection.bbox.height)
    ideal_zoom = target_subject_px / float(subject_height)
    preferred_zoom = max(min_zoom, min(max_zoom, ideal_zoom))
    if preferred_zoom != preferred_zoom:
        preferred_zoom = max(min_zoom, min(max_zoom, initial_zoom))

    zoom = preferred_zoom
    while zoom >= min_zoom - 1e-9:
        width = max(1, int(crop.width * zoom))
        height = max(1, int(crop.height * zoom))
        position = _find_non_overlap_position(canvas_w, canvas_h, width, height, occupied)
        if position is not None:
            displayed_subject_px = int(subject_height * zoom)
            if displayed_subject_px < min_subject_px:
                return None
            return LayoutPlacement(
                record=record,
                crop_bbox=crop,
                zoom=round(zoom, 6),
                width=width,
                height=height,
                x=position[0],
                y=position[1],
            )
        zoom *= zoom_decay
    return None


def _place(
    record: ImageRecord,
    canvas_w: int,
    canvas_h: int,
    padding_px: int,
    occupied: list[tuple[int, int, int, int]],
    initial_zoom: float,
    min_zoom: float,
    zoom_decay: float,
    min_subject_px: int,
    target_subject_px: int,
    max_zoom: float,
) -> LayoutPlacement | None:
    if not record.detections or record.width <= 0 or record.height <= 0:
        return None
    return _place_box(
        record,
        _crop_for_record(record, padding_px),
        canvas_w,
        canvas_h,
        occupied,
        initial_zoom,
        min_zoom,
        zoom_decay,
        min_subject_px,
        target_subject_px,
        max_zoom,
    )


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
        ("face_only", requirements.min_face_only),
        ("full_body", requirements.min_full_body),
        ("front", requirements.min_front),
        ("side", requirements.min_side),
        ("back", requirements.min_back),
        ("male", requirements.min_male),
        ("female", requirements.min_female),
        ("face_visible", requirements.min_face_visible),
        ("body_visible", requirements.min_body_visible),
    )
    result: list[str] = []
    for name, count in pairs:
        result.extend([name] * max(0, int(count)))
    return result


def _too_similar(record: ImageRecord, selected: list[ImageRecord], threshold: int) -> bool:
    if not record.phash:
        return False
    try:
        import imagehash
        current = imagehash.hex_to_hash(record.phash)
        return any(
            other.phash and (current - imagehash.hex_to_hash(other.phash)) <= threshold
            for other in selected
        )
    except Exception:
        return False


def _candidate_order(records: Iterable[ImageRecord], requirements: MosaicRequirements) -> list[ImageRecord]:
    candidates = [
        record for record in records
        if record.ranking
        and record.ranking.final_score > 0
        and record.detections
        and _hard_eligible(record, requirements)
    ]
    return sorted(
        candidates,
        key=lambda record: (
            -(1 if record.manually_included else 0),
            -(record.ranking.final_score if record.ranking else 0.0),
            record.filename.lower(),
        ),
    )


def optimize_auto_layout(
    records: list[ImageRecord],
    canvas_size: tuple[int, int],
    padding_px: int,
    requirements: MosaicRequirements | None = None,
    phash_threshold: int = 10,
    max_images: int = 100,
    initial_zoom: float = 0.5,
    min_zoom: float = 0.1,
    zoom_decay: float = 0.9,
    min_subject_px: int = 120,
    target_subject_px: int = 240,
    max_zoom: float = 1.5,
) -> LayoutEvaluation:
    """Fill the canvas with the largest readable set up to max_images.

    The packing algorithm matches ImageMosaicView: top-left scan in 10-pixel
    increments and repeated 0.9 zoom reduction when an item does not fit.
    Unlike the viewer's neutral default, AUTO chooses an ideal zoom from the
    detected subject height, so small face crops can zoom in and large body
    crops can zoom out while preserving a readable subject.
    """
    requirements = requirements or MosaicRequirements()
    candidates = _candidate_order(records, requirements)
    target = min(len(candidates), max(0, int(max_images)))
    if target <= 0:
        return LayoutEvaluation([], 0, 0.0, 0.0, 0.0, 0.0, _required_names(requirements))

    canvas_w, canvas_h = map(int, canvas_size)
    occupied: list[tuple[int, int, int, int]] = []
    placements: list[LayoutPlacement] = []
    chosen: list[ImageRecord] = []
    unmet: list[str] = []

    for required_name in _required_names(requirements):
        if len(chosen) >= target:
            unmet.append(required_name)
            continue
        placed = False
        for candidate in candidates:
            if candidate in chosen or _too_similar(candidate, chosen, phash_threshold):
                continue
            if not _matches_requirement(candidate, required_name):
                continue
            placement = _place(
                candidate, canvas_w, canvas_h, padding_px, occupied,
                initial_zoom, min_zoom, zoom_decay,
                min_subject_px, target_subject_px, max_zoom,
            )
            if placement is None:
                continue
            chosen.append(candidate)
            placements.append(placement)
            occupied.append((placement.x, placement.y, placement.width, placement.height))
            placed = True
            break
        if not placed:
            unmet.append(required_name)

    for candidate in candidates:
        if len(chosen) >= target:
            break
        if candidate in chosen or _too_similar(candidate, chosen, phash_threshold):
            continue
        placement = _place(
            candidate, canvas_w, canvas_h, padding_px, occupied,
            initial_zoom, min_zoom, zoom_decay,
            min_subject_px, target_subject_px, max_zoom,
        )
        if placement is None:
            continue
        chosen.append(candidate)
        placements.append(placement)
        occupied.append((placement.x, placement.y, placement.width, placement.height))

    canvas_area = max(1, canvas_w * canvas_h)
    occupied_area = sum(p.width * p.height for p in placements)
    evaluation = LayoutEvaluation(
        placements=placements,
        target=target,
        average_zoom=sum(p.zoom for p in placements) / len(placements) if placements else 0.0,
        average_fill_ratio=sum((p.width * p.height) / canvas_area for p in placements) / len(placements) if placements else 0.0,
        average_subject_px=(
            sum(int(_best_detection(p.record).bbox.height * p.zoom) for p in placements) / len(placements)
            if placements else 0.0
        ),
        canvas_fill_ratio=occupied_area / canvas_area,
        unmet_requirements=unmet,
    )
    logger.info(
        "AUTO layout: selected=%d/%d max=%d canvas=%dx%d avg_zoom=%.3f fill=%.1f%% subject=%.0fpx unmet=%s",
        len(placements), len(candidates), max_images, canvas_w, canvas_h,
        evaluation.average_zoom, evaluation.canvas_fill_ratio * 100.0,
        evaluation.average_subject_px, ",".join(unmet) if unmet else "none",
    )
    return evaluation


def simulate_viewer_layout(
    records: list[ImageRecord],
    canvas_size: tuple[int, int],
    padding_px: int,
    initial_zoom: float = 0.5,
    min_zoom: float = 0.1,
    zoom_decay: float = 0.9,
    min_subject_px: int = 0,
) -> list[LayoutPlacement]:
    """Simulate loading order for export fallback."""
    ordered = sorted(
        records,
        key=lambda record: (
            record.selection.slot_index if record.selection else 10**9,
            -(record.ranking.final_score if record.ranking else 0.0),
            record.filename.lower(),
        ),
    )
    occupied: list[tuple[int, int, int, int]] = []
    placements: list[LayoutPlacement] = []
    canvas_w, canvas_h = map(int, canvas_size)
    for record in ordered:
        crop = _crop_for_record(record, padding_px)
        placement = _place_box(
            record,
            crop,
            canvas_w,
            canvas_h,
            occupied,
            initial_zoom,
            min_zoom,
            zoom_decay,
            min_subject_px,
            target_subject_px=240,
            max_zoom=1.5,
        )
        if placement is None:
            continue
        placements.append(placement)
        occupied.append((placement.x, placement.y, placement.width, placement.height))
    return placements


def apply_layout_selection(evaluation: LayoutEvaluation, all_records: list[ImageRecord]) -> None:
    by_path = {placement.record.path: placement for placement in evaluation.placements}
    placement_index = {placement.record.path: index for index, placement in enumerate(evaluation.placements)}
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
