"""Global mosaic layout optimization compatible with ImageMosaicView."""
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


@dataclass
class _Row:
    records: list[ImageRecord]
    height: float
    zooms: dict[str, float]
    width: float


def _find_non_overlap_position(canvas_w: int, canvas_h: int, width: int, height: int, occupied: list[tuple[int, int, int, int]], step: int = 10) -> tuple[int, int] | None:
    """Exactly match ImageMosaicView's first-fit scan: y first, then x."""
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


def _order_variants(records: list[ImageRecord], padding_px: int) -> list[list[ImageRecord]]:
    """ImageMosaicView is order-sensitive, so test several useful orders."""
    def area(record: ImageRecord) -> float:
        crop = _crop_for_record(record, padding_px)
        return float(crop.width * crop.height)
    def height(record: ImageRecord) -> int:
        return _crop_for_record(record, padding_px).height
    def width(record: ImageRecord) -> int:
        return _crop_for_record(record, padding_px).width
    def aspect(record: ImageRecord) -> float:
        crop = _crop_for_record(record, padding_px)
        return crop.width / max(1.0, crop.height)

    variants = [
        list(records),
        sorted(records, key=lambda r: (-area(r), -r.ranking.final_score if r.ranking else 0.0)),
        sorted(records, key=lambda r: (-height(r), -r.ranking.final_score if r.ranking else 0.0)),
        sorted(records, key=lambda r: (-width(r), -r.ranking.final_score if r.ranking else 0.0)),
        sorted(records, key=lambda r: (-aspect(r), -r.ranking.final_score if r.ranking else 0.0)),
        sorted(records, key=lambda r: (aspect(r), -r.ranking.final_score if r.ranking else 0.0)),
    ]
    by_aspect = sorted(records, key=aspect)
    alternating: list[ImageRecord] = []
    left, right = 0, len(by_aspect) - 1
    while left <= right:
        alternating.append(by_aspect[right])
        right -= 1
        if left <= right:
            alternating.append(by_aspect[left])
            left += 1
    variants.append(alternating)

    seen: set[tuple[str, ...]] = set()
    unique: list[list[ImageRecord]] = []
    for variant in variants:
        key = tuple(r.path for r in variant)
        if key not in seen:
            seen.add(key)
            unique.append(variant)
    return unique


def _row_for(records: list[ImageRecord], canvas_w: int, gap_px: int, min_zoom: float, max_zoom: float) -> _Row | None:
    crops = [(r, _crop_for_record(r, 0)) for r in records]
    if not crops:
        return None
    ratio_sum = sum(crop.width / max(1.0, crop.height) for _, crop in crops)
    if ratio_sum <= 0:
        return None
    available = max(1.0, canvas_w - gap_px * max(0, len(records) - 1))
    ideal_height = available / ratio_sum
    min_height = max(min_zoom * crop.height for _, crop in crops)
    max_height = min(max_zoom * crop.height for _, crop in crops)
    if min_height > max_height + 1e-6:
        return None
    row_height = max(min_height, min(max_height, ideal_height))
    zooms = {record.path: row_height / max(1.0, float(crop.height)) for record, crop in crops}
    width = sum(crop.width * zooms[record.path] for record, crop in crops) + gap_px * max(0, len(records) - 1)
    if width > canvas_w + 1.0:
        return None
    return _Row(records, row_height, zooms, width)


def _build_rows(order: list[ImageRecord], canvas_w: int, canvas_h: int, desired_rows: int, gap_px: int = 2, min_zoom: float = 0.1, max_zoom: float = 3.0) -> list[_Row] | None:
    target_height = canvas_h / max(1, desired_rows)
    rows: list[_Row] = []
    current: list[ImageRecord] = []
    for record in order:
        trial = current + [record]
        trial_row = _row_for(trial, canvas_w, gap_px, min_zoom, max_zoom)
        if trial_row is None:
            if not current:
                return None
            row = _row_for(current, canvas_w, gap_px, min_zoom, max_zoom)
            if row is None:
                return None
            rows.append(row)
            current = [record]
            continue
        current = trial
        if len(current) > 1 and trial_row.height < target_height * 0.78:
            last = current.pop()
            row = _row_for(current, canvas_w, gap_px, min_zoom, max_zoom)
            if row is None:
                return None
            rows.append(row)
            current = [last]
    if current:
        row = _row_for(current, canvas_w, gap_px, min_zoom, max_zoom)
        if row is None:
            return None
        rows.append(row)

    total_height = sum(row.height for row in rows)
    if total_height <= 0:
        return None
    if total_height > canvas_h + 1e-6:
        scale = canvas_h / total_height
        adjusted: list[_Row] = []
        for row in rows:
            zooms = {path: zoom * scale for path, zoom in row.zooms.items()}
            if any(zoom < min_zoom - 1e-6 for zoom in zooms.values()):
                return None
            width = sum(_crop_for_record(record, 0).width * zooms[record.path] for record in row.records) + gap_px * max(0, len(row.records) - 1)
            if width > canvas_w + 1.0:
                return None
            adjusted.append(_Row(row.records, row.height * scale, zooms, width))
        rows = adjusted
    return rows


def _simulate_exact_viewer(order: list[ImageRecord], zooms: dict[str, float], canvas_size: tuple[int, int], padding_px: int, min_zoom: float, zoom_decay: float, min_subject_px: int) -> LayoutEvaluation | None:
    """Run the same first-fit packing as ImageMosaicView and measure the result."""
    canvas_w, canvas_h = map(int, canvas_size)
    occupied: list[tuple[int, int, int, int]] = []
    placements: list[LayoutPlacement] = []
    for record in order:
        crop = _crop_for_record(record, padding_px)
        subject_h = max(1, _best_detection(record).bbox.height)
        zoom = max(min_zoom, float(zooms.get(record.path, _preferred_zoom(record, 260, min_zoom, 3.0))))
        placement = None
        while zoom >= min_zoom - 1e-9:
            width = max(1, int(crop.width * zoom))
            height = max(1, int(crop.height * zoom))
            pos = _find_non_overlap_position(canvas_w, canvas_h, width, height, occupied)
            if pos is not None:
                if int(subject_h * zoom) < min_subject_px:
                    return None
                placement = LayoutPlacement(record, crop, round(zoom, 6), width, height, pos[0], pos[1])
                break
            zoom *= zoom_decay
        if placement is None:
            return None
        placements.append(placement)
        occupied.append((placement.x, placement.y, placement.width, placement.height))

    area = max(1, canvas_w * canvas_h)
    occupied_area = sum(p.width * p.height for p in placements)
    max_x = max(p.x + p.width for p in placements)
    max_y = max(p.y + p.height for p in placements)
    fill = occupied_area / area
    extent = (max_x / canvas_w) * (max_y / canvas_h)
    return LayoutEvaluation(
        placements=placements,
        target=len(order),
        average_zoom=sum(p.zoom for p in placements) / len(placements),
        average_fill_ratio=sum((p.width * p.height) / area for p in placements) / len(placements),
        average_subject_px=sum(int(_best_detection(p.record).bbox.height * p.zoom) for p in placements) / len(placements),
        canvas_fill_ratio=fill,
        unmet_requirements=[],
        layout_score=0.70 * max(0.0, min(1.0, fill)) + 0.20 * max(0.0, min(1.0, extent)) + 0.10 * max(0.0, min(1.25, (sum(int(_best_detection(p.record).bbox.height * p.zoom) for p in placements) / len(placements)) / 260.0)),
    )


def _optimize_selected_layout(selected: list[ImageRecord], canvas_size: tuple[int, int], padding_px: int, min_subject_px: int, target_subject_px: int, max_zoom: float) -> LayoutEvaluation | None:
    canvas_w, canvas_h = map(int, canvas_size)
    best: LayoutEvaluation | None = None
    max_rows = min(len(selected), max(1, canvas_h // max(80, min_subject_px)))
    for order in _order_variants(selected, padding_px):
        for desired_rows in range(1, max_rows + 1):
            rows = _build_rows(order, canvas_w, canvas_h, desired_rows, gap_px=2, min_zoom=0.1, max_zoom=max_zoom)
            if rows is None:
                continue
            zooms = {path: zoom for row in rows for path, zoom in row.zooms.items()}
            evaluation = _simulate_exact_viewer(order, zooms, canvas_size, padding_px, 0.1, 0.9, min_subject_px)
            if evaluation is None or len(evaluation.placements) != len(order):
                continue
            # Reward efficient use of both dimensions. More than 95% of the
            # canvas is already effectively full, so a tiny count increase wins
            # only when layouts are otherwise close.
            if best is None or evaluation.layout_score > best.layout_score + 1e-9:
                best = evaluation
    return best


def _target_values(max_target: int) -> list[int]:
    values = list(range(1, min(max_target, 20) + 1))
    if max_target > 20:
        values.extend(range(22, min(max_target, 60) + 1, 2))
    if max_target > 60:
        values.extend(v for v in (70, 80, 90, 100) if v <= max_target)
    return values


def optimize_auto_layout(records: list[ImageRecord], canvas_size: tuple[int, int], padding_px: int, requirements: MosaicRequirements | None = None, phash_threshold: int = 10, max_images: int = 100, initial_zoom: float = 0.5, min_zoom: float = 0.1, zoom_decay: float = 0.9, min_subject_px: int = 160, target_subject_px: int = 260, max_zoom: float = 3.0) -> LayoutEvaluation:
    """AUTO finds an image count and zoom distribution that uses the canvas globally."""
    del initial_zoom
    requirements = requirements or MosaicRequirements()
    candidates = _candidate_order(records, requirements)
    max_target = min(len(candidates), max(0, int(max_images)))
    if max_target <= 0:
        return LayoutEvaluation([], 0, 0.0, 0.0, 0.0, 0.0, _required_names(requirements), 0.0)

    best = LayoutEvaluation([], 0, 0.0, 0.0, 0.0, 0.0, [], 0.0)
    for target in _target_values(max_target):
        selected, unmet = _select_target_set(candidates, target, requirements, phash_threshold)
        if len(selected) != target:
            continue
        evaluation = _optimize_selected_layout(selected, canvas_size, padding_px, min_subject_px, target_subject_px, max_zoom)
        if evaluation is None:
            continue
        evaluation.unmet_requirements = unmet
        if evaluation.layout_score > best.layout_score + 1e-9 or (abs(evaluation.layout_score - best.layout_score) <= 0.015 and len(evaluation.placements) > len(best.placements)):
            best = evaluation

    logger.info(
        "AUTO layout: selected=%d/%d canvas=%dx%d avg_zoom=%.3f fill=%.1f%% subject=%.0fpx score=%.3f unmet=%s",
        len(best.placements), len(candidates), max_images, canvas_size[0], canvas_size[1], best.average_zoom,
        best.canvas_fill_ratio * 100.0, best.average_subject_px, best.layout_score,
        ",".join(best.unmet_requirements) if best.unmet_requirements else "none",
    )
    return best


def optimize_fixed_layout(records: list[ImageRecord], target: int, canvas_size: tuple[int, int], padding_px: int, requirements: MosaicRequirements | None = None, phash_threshold: int = 10, min_subject_px: int = 160, target_subject_px: int = 260, max_zoom: float = 3.0) -> LayoutEvaluation:
    """Use exactly target images with a globally optimized packing."""
    requirements = requirements or MosaicRequirements()
    candidates = _candidate_order(records, requirements)
    target = max(0, min(int(target), len(candidates)))
    if target == 0:
        return LayoutEvaluation([], 0, 0.0, 0.0, 0.0, 0.0, [], 0.0)
    selected, unmet = _select_target_set(candidates, target, requirements, phash_threshold)
    if len(selected) != target:
        return LayoutEvaluation([], target, 0.0, 0.0, 0.0, 0.0, unmet + ["selection_capacity"], 0.0)
    evaluation = _optimize_selected_layout(selected, canvas_size, padding_px, min_subject_px, target_subject_px, max_zoom)
    if evaluation is None:
        return LayoutEvaluation([], target, 0.0, 0.0, 0.0, 0.0, unmet + ["layout_capacity"], 0.0)
    evaluation.unmet_requirements = unmet
    return evaluation


def simulate_viewer_layout(records: list[ImageRecord], canvas_size: tuple[int, int], padding_px: int, initial_zoom: float = 0.5, min_zoom: float = 0.1, zoom_decay: float = 0.9, min_subject_px: int = 0) -> list[LayoutPlacement]:
    """Compatibility helper using each record's saved zoom and viewer first-fit."""
    del initial_zoom
    ordered = sorted(records, key=lambda r: (r.selection.slot_index if r.selection else 10**9, -(r.ranking.final_score if r.ranking else 0.0), r.filename.lower()))
    zooms = {r.path: (r.selection.zoom if r.selection else _preferred_zoom(r, 260, min_zoom, 3.0)) for r in ordered}
    evaluation = _simulate_exact_viewer(ordered, zooms, canvas_size, padding_px, min_zoom, zoom_decay, min_subject_px)
    return evaluation.placements if evaluation else []


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
