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
    layout_score: float = 0.0


def _find_non_overlap_position(canvas_w: int, canvas_h: int, width: int, height: int, occupied: list[tuple[int, int, int, int]], step: int = 10) -> tuple[int, int] | None:
    """Match ImageMosaicView's top-left 10 px packing scan."""
    if width <= 0 or height <= 0 or width > canvas_w or height > canvas_h:
        return None
    for yy in range(0, canvas_h - height + 1, step):
        for xx in range(0, canvas_w - width + 1, step):
            x2, y2 = xx + width, yy + height
            if not any(not (x2 <= ox or xx >= ox + ow or y2 <= oy or yy >= oy + oh) for ox, oy, ow, oh in occupied):
                return xx, yy
    return None


def _best_detection(record: ImageRecord):
    return max(record.detections, key=lambda d: (d.is_main, d.confidence, d.relative_size))


def _crop_for_record(record: ImageRecord, padding_px: int) -> BoundingBox:
    if record.selection and record.selection.crop_bbox:
        return record.selection.crop_bbox
    return compute_crop_box(_best_detection(record).bbox, record.width, record.height, padding_px)


def _preferred_zoom(record: ImageRecord, target_subject_px: int, min_zoom: float, max_zoom: float) -> float:
    subject_height = max(1, _best_detection(record).bbox.height)
    return max(min_zoom, min(max_zoom, target_subject_px / float(subject_height)))


def _place_box(record: ImageRecord, crop: BoundingBox, canvas_w: int, canvas_h: int, occupied: list[tuple[int, int, int, int]], base_zoom: float, min_zoom: float, zoom_decay: float, min_subject_px: int) -> LayoutPlacement | None:
    subject_height = max(1, _best_detection(record).bbox.height)
    zoom = max(min_zoom, base_zoom)
    while zoom >= min_zoom - 1e-9:
        width = max(1, int(crop.width * zoom))
        height = max(1, int(crop.height * zoom))
        position = _find_non_overlap_position(canvas_w, canvas_h, width, height, occupied)
        if position is not None:
            if int(subject_height * zoom) < min_subject_px:
                return None
            return LayoutPlacement(record, crop, round(zoom, 6), width, height, position[0], position[1])
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
    return [name for name, count in pairs for _ in range(max(0, int(count)))]


def _too_similar(record: ImageRecord, selected: list[ImageRecord], threshold: int) -> bool:
    if not record.phash:
        return False
    try:
        import imagehash
        current = imagehash.hex_to_hash(record.phash)
        return any(other.phash and (current - imagehash.hex_to_hash(other.phash)) <= threshold for other in selected)
    except Exception:
        return False


def _candidate_order(records: Iterable[ImageRecord], requirements: MosaicRequirements) -> list[ImageRecord]:
    candidates = [r for r in records if r.ranking and r.ranking.final_score > 0 and r.detections and _hard_eligible(r, requirements)]
    return sorted(candidates, key=lambda r: (-(1 if r.manually_included else 0), -(r.ranking.final_score if r.ranking else 0.0), r.filename.lower()))


def _select_target_set(candidates: list[ImageRecord], target: int, requirements: MosaicRequirements, phash_threshold: int) -> tuple[list[ImageRecord], list[str]]:
    selected: list[ImageRecord] = []
    unmet: list[str] = []
    for required_name in _required_names(requirements):
        if len(selected) >= target:
            unmet.append(required_name)
            continue
        matching = [r for r in candidates if r not in selected and _matches_requirement(r, required_name) and not _too_similar(r, selected, phash_threshold)]
        if not matching:
            unmet.append(required_name)
            continue
        selected.append(matching[0])
    for record in candidates:
        if len(selected) >= target:
            break
        if record in selected or _too_similar(record, selected, phash_threshold):
            continue
        selected.append(record)
    return selected, unmet


def _packing_order(records: list[ImageRecord], padding_px: int, target_subject_px: int, min_zoom: float, max_zoom: float) -> list[ImageRecord]:
    """Place larger expected footprints first to reduce fragmentation."""
    scored = []
    for record in records:
        crop = _crop_for_record(record, padding_px)
        zoom = _preferred_zoom(record, target_subject_px, min_zoom, max_zoom)
        footprint = float(crop.width * crop.height) * zoom * zoom
        score = record.ranking.final_score if record.ranking else 0.0
        scored.append((footprint, score, record.filename.lower(), record))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [x[3] for x in scored]


def _evaluate_order(ordered: list[ImageRecord], canvas_size: tuple[int, int], padding_px: int, scale_multiplier: float, min_zoom: float, zoom_decay: float, min_subject_px: int, target_subject_px: int, max_zoom: float) -> LayoutEvaluation:
    canvas_w, canvas_h = map(int, canvas_size)
    occupied: list[tuple[int, int, int, int]] = []
    placements: list[LayoutPlacement] = []
    for record in ordered:
        crop = _crop_for_record(record, padding_px)
        preferred = _preferred_zoom(record, target_subject_px, min_zoom, max_zoom)
        base_zoom = min(max_zoom, preferred * scale_multiplier)
        placement = _place_box(record, crop, canvas_w, canvas_h, occupied, base_zoom, min_zoom, zoom_decay, min_subject_px)
        if placement is None:
            continue
        placements.append(placement)
        occupied.append((placement.x, placement.y, placement.width, placement.height))
    area = max(1, canvas_w * canvas_h)
    occupied_area = sum(p.width * p.height for p in placements)
    return LayoutEvaluation(
        placements=placements,
        target=len(ordered),
        average_zoom=sum(p.zoom for p in placements) / len(placements) if placements else 0.0,
        average_fill_ratio=sum((p.width * p.height) / area for p in placements) / len(placements) if placements else 0.0,
        average_subject_px=sum(int(_best_detection(p.record).bbox.height * p.zoom) for p in placements) / len(placements) if placements else 0.0,
        canvas_fill_ratio=occupied_area / area,
        unmet_requirements=[],
    )


def _optimize_target_layout(selected: list[ImageRecord], canvas_size: tuple[int, int], padding_px: int, min_zoom: float, zoom_decay: float, min_subject_px: int, target_subject_px: int, max_zoom: float) -> LayoutEvaluation | None:
    ordered = _packing_order(selected, padding_px, target_subject_px, min_zoom, max_zoom)
    if not ordered:
        return None
    minimum_scale = max(0.10, min(1.0, min_subject_px / max(1, target_subject_px)))
    minimum_eval = _evaluate_order(ordered, canvas_size, padding_px, minimum_scale, min_zoom, zoom_decay, min_subject_px, target_subject_px, max_zoom)
    if len(minimum_eval.placements) < len(ordered):
        return None
    best = minimum_eval
    low, high = minimum_scale, 3.0
    for _ in range(11):
        mid = (low + high) / 2.0
        evaluation = _evaluate_order(ordered, canvas_size, padding_px, mid, min_zoom, zoom_decay, min_subject_px, target_subject_px, max_zoom)
        if len(evaluation.placements) == len(ordered):
            best = evaluation
            low = mid
        else:
            high = mid
    return best


def _layout_utility(evaluation: LayoutEvaluation, candidate_count: int, target_subject_px: int) -> float:
    if not evaluation.placements:
        return 0.0
    fill = max(0.0, min(1.0, evaluation.canvas_fill_ratio))
    readability = max(0.0, min(1.25, evaluation.average_subject_px / max(1.0, target_subject_px)))
    count_density = (len(evaluation.placements) / max(1, candidate_count)) ** 0.5
    return 0.55 * fill + 0.30 * readability + 0.15 * count_density


def optimize_auto_layout(records: list[ImageRecord], canvas_size: tuple[int, int], padding_px: int, requirements: MosaicRequirements | None = None, phash_threshold: int = 10, max_images: int = 100, initial_zoom: float = 0.5, min_zoom: float = 0.1, zoom_decay: float = 0.9, min_subject_px: int = 160, target_subject_px: int = 260, max_zoom: float = 3.0) -> LayoutEvaluation:
    """Automatically choose N and per-image zoom to use the canvas efficiently."""
    del initial_zoom
    requirements = requirements or MosaicRequirements()
    candidates = _candidate_order(records, requirements)
    max_target = min(len(candidates), max(0, int(max_images)))
    if max_target <= 0:
        return LayoutEvaluation([], 0, 0.0, 0.0, 0.0, 0.0, _required_names(requirements), 0.0)

    best = LayoutEvaluation([], 0, 0.0, 0.0, 0.0, 0.0, [], 0.0)
    feasible_max = 0
    for target in range(1, max_target + 1):
        selected, unmet = _select_target_set(candidates, target, requirements, phash_threshold)
        if len(selected) < target:
            break
        evaluation = _optimize_target_layout(selected, canvas_size, padding_px, min_zoom, zoom_decay, min_subject_px, target_subject_px, max_zoom)
        if evaluation is None:
            break
        feasible_max = target
        evaluation.unmet_requirements = unmet
        evaluation.layout_score = _layout_utility(evaluation, len(candidates), target_subject_px)
        if evaluation.layout_score > best.layout_score:
            best = evaluation

    logger.info(
        "AUTO layout: selected=%d/%d max=%d feasible_max=%d canvas=%dx%d avg_zoom=%.3f fill=%.1f%% subject=%.0fpx score=%.3f unmet=%s",
        len(best.placements), len(candidates), max_images, feasible_max, canvas_size[0], canvas_size[1],
        best.average_zoom, best.canvas_fill_ratio * 100.0, best.average_subject_px, best.layout_score,
        ",".join(best.unmet_requirements) if best.unmet_requirements else "none",
    )
    return best


def simulate_viewer_layout(records: list[ImageRecord], canvas_size: tuple[int, int], padding_px: int, initial_zoom: float = 0.5, min_zoom: float = 0.1, zoom_decay: float = 0.9, min_subject_px: int = 0) -> list[LayoutPlacement]:
    """Simulate ImageMosaicView's packing using saved per-image zoom."""
    del initial_zoom
    ordered = sorted(records, key=lambda r: (r.selection.slot_index if r.selection else 10**9, -(r.ranking.final_score if r.ranking else 0.0), r.filename.lower()))
    occupied: list[tuple[int, int, int, int]] = []
    placements: list[LayoutPlacement] = []
    canvas_w, canvas_h = map(int, canvas_size)
    for record in ordered:
        if not record.detections:
            continue
        crop = _crop_for_record(record, padding_px)
        requested_zoom = record.selection.zoom if record.selection else _preferred_zoom(record, 260, min_zoom, 3.0)
        placement = _place_box(record, crop, canvas_w, canvas_h, occupied, requested_zoom, min_zoom, zoom_decay, min_subject_px)
        if placement is None:
            continue
        placements.append(placement)
        occupied.append((placement.x, placement.y, placement.width, placement.height))
    return placements


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
