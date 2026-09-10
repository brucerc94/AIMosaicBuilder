"""Global mosaic layout optimizer that mirrors ImageMosaicView's packing rules."""
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


def _find_non_overlap_position(
    canvas_w: int,
    canvas_h: int,
    width: int,
    height: int,
    occupied: list[tuple[int, int, int, int]],
    step: int = 10,
) -> tuple[int, int] | None:
    """Exactly mirror ImageMosaicView's y/x 10-pixel first-fit search."""
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


def _layout_crop(record: ImageRecord, padding_px: int) -> BoundingBox:
    """Always derive the current crop from the current detection + padding."""
    return compute_crop_box(_best_detection(record).bbox, record.width, record.height, padding_px)


def _saved_or_layout_crop(record: ImageRecord, padding_px: int) -> BoundingBox:
    """Use a saved crop only for export fallback; otherwise derive it."""
    if record.selection and record.selection.crop_bbox:
        return record.selection.crop_bbox
    return _layout_crop(record, padding_px)


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
    return [name for name, count in pairs for _ in range(max(0, int(count)))]


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
        record
        for record in records
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


def _select_target_set(
    candidates: list[ImageRecord],
    target: int,
    requirements: MosaicRequirements,
    phash_threshold: int,
) -> tuple[list[ImageRecord], list[str]]:
    selected: list[ImageRecord] = []
    unmet: list[str] = []
    for required_name in _required_names(requirements):
        if len(selected) >= target:
            unmet.append(required_name)
            continue
        matching = [
            record
            for record in candidates
            if record not in selected
            and _matches_requirement(record, required_name)
            and not _too_similar(record, selected, phash_threshold)
        ]
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
    """Generate several orders because the viewer's first-fit packer is order-sensitive."""
    def area(record: ImageRecord) -> float:
        crop = _layout_crop(record, padding_px)
        return float(crop.width * crop.height)

    def height(record: ImageRecord) -> int:
        return _layout_crop(record, padding_px).height

    def width(record: ImageRecord) -> int:
        return _layout_crop(record, padding_px).width

    def aspect(record: ImageRecord) -> float:
        crop = _layout_crop(record, padding_px)
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

    unique: list[list[ImageRecord]] = []
    seen: set[tuple[str, ...]] = set()
    for variant in variants:
        key = tuple(record.path for record in variant)
        if key not in seen:
            seen.add(key)
            unique.append(variant)
    return unique


def _row_for(
    records: list[ImageRecord],
    canvas_w: int,
    padding_px: int,
    gap_px: int,
    min_zoom: float,
    max_zoom: float,
) -> _Row | None:
    if not records:
        return None
    crops = [(record, _layout_crop(record, padding_px)) for record in records]
    ratio_sum = sum(crop.width / max(1.0, crop.height) for _, crop in crops)
    if ratio_sum <= 0:
        return None
    available = max(1.0, float(canvas_w - gap_px * max(0, len(records) - 1)))
    ideal_height = available / ratio_sum
    min_height = max(min_zoom * crop.height for _, crop in crops)
    max_height = min(max_zoom * crop.height for _, crop in crops)
    if min_height > max_height + 1e-6:
        return None
    row_height = max(min_height, min(max_height, ideal_height))
    zooms = {record.path: row_height / max(1.0, crop.height) for record, crop in crops}
    width = sum(crop.width * zooms[record.path] for record, crop in crops) + gap_px * max(0, len(records) - 1)
    if width > canvas_w + 1.0:
        return None
    return _Row(records, row_height, zooms, width)


def _build_rows(
    order: list[ImageRecord],
    canvas_w: int,
    canvas_h: int,
    padding_px: int,
    desired_rows: int,
    gap_px: int = 2,
    min_zoom: float = 0.1,
    max_zoom: float = 3.0,
) -> list[_Row] | None:
    """Build justified rows and ensure their combined height fits the canvas."""
    target_height = canvas_h / max(1, desired_rows)
    rows: list[_Row] = []
    current: list[ImageRecord] = []

    for index, record in enumerate(order):
        trial = current + [record]
        trial_row = _row_for(trial, canvas_w, padding_px, gap_px, min_zoom, max_zoom)
        if trial_row is None:
            if not current:
                return None
            finalized = _row_for(current, canvas_w, padding_px, gap_px, min_zoom, max_zoom)
            if finalized is None:
                return None
            rows.append(finalized)
            current = [record]
            continue

        current = trial
        if len(current) > 1 and trial_row.height < target_height * 0.78:
            last = current.pop()
            finalized = _row_for(current, canvas_w, padding_px, gap_px, min_zoom, max_zoom)
            if finalized is None:
                return None
            rows.append(finalized)
            current = [last]

        if index == len(order) - 1:
            break

    if current:
        finalized = _row_for(current, canvas_w, padding_px, gap_px, min_zoom, max_zoom)
        if finalized is None:
            return None
        rows.append(finalized)

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
            width = sum(
                _layout_crop(record, padding_px).width * zooms[record.path]
                for record in row.records
            ) + gap_px * max(0, len(row.records) - 1)
            if width > canvas_w + 1.0:
                return None
            adjusted.append(_Row(row.records, row.height * scale, zooms, width))
        rows = adjusted
    return rows


def _simulate_exact_viewer(
    order: list[ImageRecord],
    zooms: dict[str, float],
    canvas_size: tuple[int, int],
    padding_px: int,
    min_zoom: float,
    zoom_decay: float,
    min_subject_px: int,
) -> LayoutEvaluation | None:
    """Run the same placement algorithm used by ImageMosaicView."""
    canvas_w, canvas_h = map(int, canvas_size)
    occupied: list[tuple[int, int, int, int]] = []
    placements: list[LayoutPlacement] = []

    for record in order:
        crop = _layout_crop(record, padding_px)
        subject_h = max(1, _best_detection(record).bbox.height)
        zoom = max(min_zoom, float(zooms.get(record.path, _preferred_zoom(record, 260, min_zoom, 3.0))))
        placement = None
        while zoom >= min_zoom - 1e-9:
            width = max(1, int(crop.width * zoom))
            height = max(1, int(crop.height * zoom))
            position = _find_non_overlap_position(canvas_w, canvas_h, width, height, occupied)
            if position is not None:
                if int(subject_h * zoom) < min_subject_px:
                    return None
                placement = LayoutPlacement(
                    record=record,
                    crop_bbox=crop,
                    zoom=round(zoom, 6),
                    width=width,
                    height=height,
                    x=position[0],
                    y=position[1],
                )
                break
            zoom *= zoom_decay
        if placement is None:
            return None
        placements.append(placement)
        occupied.append((placement.x, placement.y, placement.width, placement.height))

    if not placements:
        return None

    area = max(1, canvas_w * canvas_h)
    occupied_area = sum(p.width * p.height for p in placements)
    max_x = max(p.x + p.width for p in placements)
    max_y = max(p.y + p.height for p in placements)
    fill = max(0.0, min(1.0, occupied_area / area))
    extent = max(0.0, min(1.0, (max_x / canvas_w) * (max_y / canvas_h)))
    average_subject = sum(int(_best_detection(p.record).bbox.height * p.zoom) for p in placements) / len(placements)
    readability = max(0.0, min(1.25, average_subject / 260.0))

    return LayoutEvaluation(
        placements=placements,
        target=len(order),
        average_zoom=sum(p.zoom for p in placements) / len(placements),
        average_fill_ratio=sum((p.width * p.height) / area for p in placements) / len(placements),
        average_subject_px=average_subject,
        canvas_fill_ratio=fill,
        unmet_requirements=[],
        layout_score=0.70 * fill + 0.20 * extent + 0.10 * readability,
    )


def _optimize_selected_layout(
    selected: list[ImageRecord],
    canvas_size: tuple[int, int],
    padding_px: int,
    min_subject_px: int,
    target_subject_px: int,
    max_zoom: float,
) -> LayoutEvaluation | None:
    canvas_w, canvas_h = map(int, canvas_size)
    best: LayoutEvaluation | None = None
    max_rows = min(len(selected), max(1, canvas_h // max(80, min_subject_px)))

    for order in _order_variants(selected, padding_px):
        for desired_rows in range(1, max_rows + 1):
            rows = _build_rows(
                order,
                canvas_w,
                canvas_h,
                padding_px,
                desired_rows,
                gap_px=2,
                min_zoom=0.1,
                max_zoom=max_zoom,
            )
            if rows is None:
                continue
            zooms = {path: zoom for row in rows for path, zoom in row.zooms.items()}
            evaluation = _simulate_exact_viewer(
                order,
                zooms,
                canvas_size,
                padding_px,
                min_zoom=0.1,
                zoom_decay=0.9,
                min_subject_px=min_subject_px,
            )
            if evaluation is None or len(evaluation.placements) != len(order):
                continue
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
    min_subject_px: int = 160,
    target_subject_px: int = 260,
    max_zoom: float = 3.0,
) -> LayoutEvaluation:
    """AUTO selects N and per-image zoom using global canvas-aware packing."""
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
        evaluation = _optimize_selected_layout(
            selected,
            canvas_size,
            padding_px,
            min_subject_px,
            target_subject_px,
            max_zoom,
        )
        if evaluation is None:
            continue
        evaluation.unmet_requirements = unmet
        if evaluation.layout_score > best.layout_score + 1e-9 or (
            abs(evaluation.layout_score - best.layout_score) <= 0.015
            and len(evaluation.placements) > len(best.placements)
        ):
            best = evaluation

    logger.info(
        "AUTO layout: selected=%d/%d max=%d canvas=%dx%d avg_zoom=%.3f fill=%.1f%% subject=%.0fpx score=%.3f unmet=%s",
        len(best.placements), len(candidates), max_images,
        canvas_size[0], canvas_size[1], best.average_zoom,
        best.canvas_fill_ratio * 100.0, best.average_subject_px,
        best.layout_score,
        ",".join(best.unmet_requirements) if best.unmet_requirements else "none",
    )
    return best


def optimize_fixed_layout(
    records: list[ImageRecord],
    target: int,
    canvas_size: tuple[int, int],
    padding_px: int,
    requirements: MosaicRequirements | None = None,
    phash_threshold: int = 10,
    min_subject_px: int = 160,
    target_subject_px: int = 260,
    max_zoom: float = 3.0,
) -> LayoutEvaluation:
    """Use exactly target images and globally optimize their layout."""
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
        logger.warning("FIXED layout: could not place all %d requested images", target)
        return LayoutEvaluation([], target, 0.0, 0.0, 0.0, 0.0, unmet + ["layout_capacity"], 0.0)
    evaluation.unmet_requirements = unmet
    logger.info(
        "FIXED layout: selected=%d target=%d canvas=%dx%d avg_zoom=%.3f fill=%.1f%% subject=%.0fpx unmet=%s",
        len(evaluation.placements), target, canvas_size[0], canvas_size[1],
        evaluation.average_zoom, evaluation.canvas_fill_ratio * 100.0,
        evaluation.average_subject_px,
        ",".join(unmet) if unmet else "none",
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
    """Compatibility helper that uses saved zoom and the viewer first-fit scan."""
    del initial_zoom
    ordered = sorted(
        records,
        key=lambda record: (
            record.selection.slot_index if record.selection else 10**9,
            -(record.ranking.final_score if record.ranking else 0.0),
            record.filename.lower(),
        ),
    )
    zooms = {
        record.path: (
            record.selection.zoom
            if record.selection
            else _preferred_zoom(record, 260, min_zoom, 3.0)
        )
        for record in ordered
    }
    # For compatibility/export fallback, use a saved crop when present.
    occupied: list[tuple[int, int, int, int]] = []
    placements: list[LayoutPlacement] = []
    canvas_w, canvas_h = map(int, canvas_size)
    for record in ordered:
        crop = _saved_or_layout_crop(record, padding_px)
        zoom = max(min_zoom, float(zooms[record.path]))
        placement = None
        while zoom >= min_zoom - 1e-9:
            width = max(1, int(crop.width * zoom))
            height = max(1, int(crop.height * zoom))
            position = _find_non_overlap_position(canvas_w, canvas_h, width, height, occupied)
            if position is not None:
                placement = LayoutPlacement(record, crop, round(zoom, 6), width, height, position[0], position[1])
                break
            zoom *= zoom_decay
        if placement is None:
            continue
        placements.append(placement)
        occupied.append((placement.x, placement.y, placement.width, placement.height))
    return placements


def apply_layout_selection(evaluation: LayoutEvaluation, all_records: list[ImageRecord]) -> None:
    by_path = {placement.record.path: placement for placement in evaluation.placements}
    placement_index = {
        placement.record.path: index
        for index, placement in enumerate(evaluation.placements)
    }
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
