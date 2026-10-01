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


def _order_variants(records: list[ImageRecord], padding_px: int) -> list[list[ImageRecord]]:
    """Generate several orders because the real viewer's first-fit packer is order-sensitive."""
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
        sorted(records, key=lambda r: (-area(r), -(r.ranking.final_score if r.ranking else 0.0))),
        sorted(records, key=lambda r: (-height(r), -(r.ranking.final_score if r.ranking else 0.0))),
        sorted(records, key=lambda r: (-width(r), -(r.ranking.final_score if r.ranking else 0.0))),
        sorted(records, key=lambda r: (-aspect(r), -(r.ranking.final_score if r.ranking else 0.0))),
        sorted(records, key=lambda r: (aspect(r), -(r.ranking.final_score if r.ranking else 0.0))),
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


def _simulate_exact_viewer(
    order: list[ImageRecord],
    zooms: dict[str, float],
    canvas_size: tuple[int, int],
    padding_px: int,
    min_zoom: float,
    zoom_decay: float,
    min_subject_px: int,
    target_subject_px: int,
) -> LayoutEvaluation | None:
    """Simulate the actual first-fit mosaic placement used by Preview/export."""
    canvas_w, canvas_h = map(int, canvas_size)
    if canvas_w <= 0 or canvas_h <= 0:
        return None

    occupied: list[tuple[int, int, int, int]] = []
    placements: list[LayoutPlacement] = []

    for record in order:
        crop = _layout_crop(record, padding_px)
        subject_h = max(1, _best_detection(record).bbox.height)
        zoom = max(
            min_zoom,
            float(
                zooms.get(
                    record.path,
                    _preferred_zoom(record, target_subject_px, min_zoom, 3.0),
                )
            ),
        )

        placement = None
        while zoom >= min_zoom - 1e-9:
            width = max(1, int(round(crop.width * zoom)))
            height = max(1, int(round(crop.height * zoom)))
            position = _find_non_overlap_position(
                canvas_w,
                canvas_h,
                width,
                height,
                occupied,
            )
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
        occupied.append(
            (
                placement.x,
                placement.y,
                placement.width,
                placement.height,
            )
        )

    if not placements:
        return None

    area = max(1, canvas_w * canvas_h)
    occupied_area = sum(p.width * p.height for p in placements)
    min_x = min(p.x for p in placements)
    min_y = min(p.y for p in placements)
    max_x = max(p.x + p.width for p in placements)
    max_y = max(p.y + p.height for p in placements)

    fill = max(0.0, min(1.0, occupied_area / area))
    extent = max(
        0.0,
        min(
            1.0,
            ((max_x - min_x) / canvas_w)
            * ((max_y - min_y) / canvas_h),
        ),
    )

    average_subject = sum(
        int(_best_detection(p.record).bbox.height * p.zoom)
        for p in placements
    ) / len(placements)

    target = max(1.0, float(target_subject_px))
    target_fits = []
    for placement in placements:
        subject_px = max(
            1.0,
            float(_best_detection(placement.record).bbox.height) * placement.zoom,
        )
        relative_error = abs(subject_px - target) / target
        target_fits.append(1.0 / (1.0 + relative_error))

    target_fit = sum(target_fits) / len(target_fits)
    layout_score = (
        0.75 * fill
        + 0.15 * extent
        + 0.10 * target_fit
    )

    return LayoutEvaluation(
        placements=placements,
        target=len(order),
        average_zoom=sum(p.zoom for p in placements) / len(placements),
        average_fill_ratio=sum(
            (p.width * p.height) / area for p in placements
        ) / len(placements),
        average_subject_px=average_subject,
        canvas_fill_ratio=fill,
        unmet_requirements=[],
        layout_score=layout_score,
    )


def _zoom_scale_candidates(
    selected: list[ImageRecord],
    canvas_size: tuple[int, int],
    padding_px: int,
    target_subject_px: int,
    min_subject_px: int,
    max_zoom: float,
) -> list[float]:
    """Build bounded global zoom scales around the canvas-area estimate."""
    canvas_w, canvas_h = map(int, canvas_size)
    canvas_area = max(1.0, float(canvas_w * canvas_h))
    base_area = 0.0

    for record in selected:
        crop = _layout_crop(record, padding_px)
        subject_h = max(1, _best_detection(record).bbox.height)
        base_zoom = max(0.1, target_subject_px / float(subject_h))
        base_zoom = min(max_zoom, max(min_subject_px / float(subject_h), base_zoom))
        base_area += crop.width * crop.height * base_zoom * base_zoom

    if base_area <= 0:
        estimated = 1.0
    else:
        estimated = (0.82 * canvas_area / base_area) ** 0.5

    estimated = max(0.35, min(2.5, estimated))
    values = {
        0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95,
        1.05, 1.15, 1.25, 1.40, 1.60, 1.80, 2.00, 2.25, 2.50,
    }
    for offset in (
        -0.55, -0.45, -0.35, -0.25, -0.15,
        -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55,
    ):
        values.add(max(0.25, min(3.0, estimated + offset)))

    return sorted(values)


def _optimize_selected_layout(
    selected: list[ImageRecord],
    canvas_size: tuple[int, int],
    padding_px: int,
    min_subject_px: int,
    target_subject_px: int,
    max_zoom: float,
) -> LayoutEvaluation | None:
    """Optimize the same first-fit geometry used by Preview/export."""
    if not selected:
        return None

    best: LayoutEvaluation | None = None
    zoom_factors = _zoom_scale_candidates(
        selected,
        canvas_size,
        padding_px,
        target_subject_px,
        min_subject_px,
        max_zoom,
    )

    for order in _order_variants(selected, padding_px):
        base_zooms = {
            record.path: max(
                0.1,
                min(
                    max_zoom,
                    max(
                        min_subject_px / max(1, _best_detection(record).bbox.height),
                        target_subject_px / max(1, _best_detection(record).bbox.height),
                    ),
                ),
            )
            for record in order
        }

        for factor in zoom_factors:
            zooms = {
                path: min(max_zoom, zoom * factor)
                for path, zoom in base_zooms.items()
            }
            evaluation = _simulate_exact_viewer(
                order,
                zooms,
                canvas_size,
                padding_px,
                min_zoom=0.1,
                zoom_decay=0.9,
                min_subject_px=min_subject_px,
                target_subject_px=target_subject_px,
            )
            if evaluation is None or len(evaluation.placements) != len(order):
                continue

            if best is None or evaluation.layout_score > best.layout_score + 1e-9:
                best = evaluation

    return best



def _target_values(max_target: int) -> list[int]:
    """Return every feasible target count; AUTO must not silently skip counts."""
    return list(range(1, max(0, int(max_target)) + 1))


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
    min_image_width: int = 0,
    min_image_height: int = 0,
) -> LayoutEvaluation:
    """AUTO selects N and per-image zoom using global canvas-aware packing."""
    del initial_zoom
    requirements = requirements or MosaicRequirements()
    candidates = _candidate_order(
        records,
        requirements,
        min_image_width=min_image_width,
        min_image_height=min_image_height,
    )
    max_target = min(len(candidates), max(0, int(max_images)))
    if max_target <= 0:
        unmet = _unmet_requirement_messages(_required_counts(requirements))
        if not unmet:
            unmet = ["No eligible images are available for the current filters"]
        return LayoutEvaluation([], 0, 0.0, 0.0, 0.0, 0.0, unmet, 0.0)

    best = LayoutEvaluation([], 0, 0.0, 0.0, 0.0, 0.0, [], 0.0)
    last_failure: list[str] = []

    for target in _target_values(max_target):
        selected, unmet = _select_target_set(candidates, target, requirements, phash_threshold)
        if len(selected) != target or unmet:
            if unmet:
                last_failure = list(unmet)
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
            last_failure = [
                f"Target Images={target}: layout cannot satisfy Min Subject in the current canvas"
            ]
            continue
        if evaluation.layout_score > best.layout_score + 1e-9 or (
            abs(evaluation.layout_score - best.layout_score) <= 0.015
            and len(evaluation.placements) > len(best.placements)
        ):
            best = evaluation

    if not best.placements and last_failure:
        best.unmet_requirements = last_failure

    logger.info(
        "AUTO layout: selected=%d/%d max=%d canvas=%dx%d avg_zoom=%.3f fill=%.1f%% subject=%.0fpx score=%.3f unmet=%s",
        len(best.placements), len(candidates), max_images,
        canvas_size[0], canvas_size[1], best.average_zoom,
        best.canvas_fill_ratio * 100.0, best.average_subject_px,
        best.layout_score,
        ",".join(best.unmet_requirements) if best.unmet_requirements else "none",
    )
    return best


def _layout_proxy(record: ImageRecord, padding_px: int, min_subject_px: int) -> float:
    """Approximate how much row height this image needs to satisfy Min Subject."""
    crop = _layout_crop(record, padding_px)
    subject_h = max(1, _best_detection(record).bbox.height)
    return max(
        0.1 * crop.height,
        min_subject_px * crop.height / float(subject_h),
    )


def _try_layout_selection_repair(
    selected: list[ImageRecord],
    candidates: list[ImageRecord],
    target: int,
    requirements: MosaicRequirements,
    phash_threshold: int,
    canvas_size: tuple[int, int],
    padding_px: int,
    min_subject_px: int,
    target_subject_px: int,
    max_zoom: float,
) -> LayoutEvaluation | None:
    """Try bounded one-for-one selection repairs when the first exact set cannot fit."""
    if len(selected) != target:
        return None

    removable = [
        record
        for record in selected
        if not record.manually_included
    ]
    if not removable:
        return None

    replacement_pool = [
        record
        for record in candidates
        if record not in selected
    ]
    if not replacement_pool:
        return None

    removable.sort(
        key=lambda record: (
            -_layout_proxy(record, padding_px, min_subject_px),
            record.ranking.final_score if record.ranking else 0.0,
        )
    )
    replacement_pool.sort(
        key=lambda record: (
            _layout_proxy(record, padding_px, min_subject_px),
            -(record.ranking.final_score if record.ranking else 0.0),
            record.filename.lower(),
        )
    )

    # Keep the recovery bounded: this path runs only after the normal layout
    # search failed, and it should not turn every Generate operation into a
    # combinatorial search.
    removable = removable[: min(6, len(removable))]
    replacement_pool = replacement_pool[: min(16, len(replacement_pool))]

    best: LayoutEvaluation | None = None
    for remove in removable:
        base = [record for record in selected if record is not remove]
        for add in replacement_pool:
            if add in base:
                continue
            if _too_similar(add, base, phash_threshold):
                continue

            trial = base + [add]
            if any(_requirement_deficits(trial, requirements).values()):
                continue

            evaluation = _optimize_selected_layout(
                trial,
                canvas_size,
                padding_px,
                min_subject_px,
                target_subject_px,
                max_zoom,
            )
            if evaluation is None or len(evaluation.placements) != target:
                continue

            if best is None or evaluation.layout_score > best.layout_score + 1e-9:
                best = evaluation

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
    min_image_width: int = 0,
    min_image_height: int = 0,
) -> LayoutEvaluation:
    """Use exactly target images and globally optimize their layout."""
    requirements = requirements or MosaicRequirements()
    candidates = _candidate_order(
        records,
        requirements,
        min_image_width=min_image_width,
        min_image_height=min_image_height,
    )
    target = int(target)
    if target <= 0:
        return LayoutEvaluation(
            [],
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            ["Target Images must be greater than 0 in Fixed mode"],
            0.0,
        )
    if target > len(candidates):
        return LayoutEvaluation(
            [],
            target,
            0.0,
            0.0,
            0.0,
            0.0,
            [f"Target Images: requested {target}, only {len(candidates)} eligible images"],
            0.0,
        )

    selected, unmet = _select_target_set(candidates, target, requirements, phash_threshold)
    if len(selected) != target or unmet:
        return LayoutEvaluation(
            [],
            target,
            0.0,
            0.0,
            0.0,
            0.0,
            unmet or [f"Target Images: could not select exactly {target} images"],
            0.0,
        )

    evaluation = _optimize_selected_layout(
        selected,
        canvas_size,
        padding_px,
        min_subject_px,
        target_subject_px,
        max_zoom,
    )

    if evaluation is None and target <= 40:
        logger.warning(
            "FIXED layout: initial selection could not place all %d requested images; "
            "trying bounded layout-aware selection repair",
            target,
        )
        repaired = _try_layout_selection_repair(
            selected=selected,
            candidates=candidates,
            target=target,
            requirements=requirements,
            phash_threshold=phash_threshold,
            canvas_size=canvas_size,
            padding_px=padding_px,
            min_subject_px=min_subject_px,
            target_subject_px=target_subject_px,
            max_zoom=max_zoom,
        )
        if repaired is not None:
            evaluation = repaired
            selected = [placement.record for placement in repaired.placements]

    if evaluation is None:
        logger.warning("FIXED layout: could not place all %d requested images", target)
        return LayoutEvaluation(
            [],
            target,
            0.0,
            0.0,
            0.0,
            0.0,
            unmet + [
                f"Target Images={target}: no canvas-fitting layout found with the current Min Subject and image set",
            ],
            0.0,
        )

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
    """Reproduce saved optimizer positions; fall back to first-fit for old sessions."""
    del initial_zoom
    ordered = sorted(
        records,
        key=lambda record: (
            record.selection.slot_index if record.selection else 10**9,
            -(record.ranking.final_score if record.ranking else 0.0),
            record.filename.lower(),
        ),
    )

    canvas_w, canvas_h = map(int, canvas_size)
    occupied: list[tuple[int, int, int, int]] = []
    placements: list[LayoutPlacement] = []

    for record in ordered:
        crop = _saved_or_layout_crop(record, padding_px)
        zoom = max(
            min_zoom,
            float(
                record.selection.zoom
                if record.selection
                else _preferred_zoom(record, 260, min_zoom, 3.0)
            ),
        )
        width = max(1, int(round(crop.width * zoom)))
        height = max(1, int(round(crop.height * zoom)))

        saved_x = int(getattr(record.selection, "x", -1)) if record.selection else -1
        saved_y = int(getattr(record.selection, "y", -1)) if record.selection else -1

        if saved_x >= 0 and saved_y >= 0:
            position = (saved_x, saved_y)
            valid_saved_position = (
                saved_x + width <= canvas_w
                and saved_y + height <= canvas_h
                and not any(
                    not (
                        saved_x + width <= ox
                        or saved_x >= ox + ow
                        or saved_y + height <= oy
                        or saved_y >= oy + oh
                    )
                    for ox, oy, ow, oh in occupied
                )
            )
            if not valid_saved_position:
                position = None
        else:
            position = None

        if position is None:
            zoom_candidate = zoom
            while zoom_candidate >= min_zoom - 1e-9:
                width = max(1, int(crop.width * zoom_candidate))
                height = max(1, int(crop.height * zoom_candidate))
                position = _find_non_overlap_position(
                    canvas_w,
                    canvas_h,
                    width,
                    height,
                    occupied,
                )
                if position is not None:
                    zoom = zoom_candidate
                    break
                zoom_candidate *= zoom_decay

        if position is None:
            continue

        if min_subject_px > 0:
            subject_h = max(1, _best_detection(record).bbox.height)
            if int(subject_h * zoom) < min_subject_px:
                continue

        placement = LayoutPlacement(
            record,
            crop,
            round(zoom, 6),
            width,
            height,
            position[0],
            position[1],
        )
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
                x=placement.x,
                y=placement.y,
            )
        elif record.status == ImageStatus.SELECTED:
            record.status = ImageStatus.REJECTED