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




def _find_non_overlap_position(
    canvas_w: int,
    canvas_h: int,
    width: int,
    height: int,
    occupied: list[tuple[int, int, int, int]],
    step: int = 10,
) -> tuple[int, int] | None:
    """Mirror ImageMosaicView's y/x 10-pixel first-fit scan."""
    if width <= 0 or height <= 0 or width > canvas_w or height > canvas_h:
        return None
    for yy in range(0, canvas_h - height + 1, step):
        for xx in range(0, canvas_w - width + 1, step):
            x2 = xx + width
            y2 = yy + height
            if not any(
                not (
                    x2 <= ox
                    or xx >= ox + ow
                    or y2 <= oy
                    or yy >= oy + oh
                )
                for ox, oy, ow, oh in occupied
            ):
                return xx, yy
    return None


def _best_detection(record: ImageRecord):
    return max(
        record.detections,
        key=lambda detection: (
            detection.is_main,
            detection.confidence,
            detection.relative_size,
        ),
    )


def _layout_crop(record: ImageRecord, padding_px: int) -> BoundingBox:
    """Derive the crop from the current main detection and configured padding."""
    return compute_crop_box(
        _best_detection(record).bbox,
        record.width,
        record.height,
        padding_px,
    )


def _saved_or_layout_crop(record: ImageRecord, padding_px: int) -> BoundingBox:
    if record.selection and record.selection.crop_bbox:
        return record.selection.crop_bbox
    return _layout_crop(record, padding_px)


def _preferred_zoom(
    record: ImageRecord,
    target_subject_px: int,
    min_zoom: float,
    max_zoom: float,
) -> float:
    subject_height = max(1, _best_detection(record).bbox.height)
    return max(
        min_zoom,
        min(max_zoom, target_subject_px / float(subject_height)),
    )


def _hard_eligible(
    record: ImageRecord,
    requirements: MosaicRequirements,
    min_image_width: int = 0,
    min_image_height: int = 0,
) -> bool:
    analysis = record.analysis
    if not analysis or not analysis.has_person or record.manually_excluded:
        return False

    if min_image_width > 0 and record.width < int(min_image_width):
        return False
    if min_image_height > 0 and record.height < int(min_image_height):
        return False

    if (
        analysis.image_quality < requirements.min_quality
        or analysis.person_visibility < requirements.min_person_visibility
    ):
        return False
    if requirements.exclude_blurry and analysis.blur > 0.5:
        return False
    if requirements.exclude_occluded and analysis.occluded:
        return False

    rating = str(analysis.visual_tags.get("content_rating", "unknown"))
    if requirements.nsfw_policy == "safe_only" and rating != "safe":
        return False
    if (
        requirements.nsfw_policy == "nsfw_only"
        and rating not in {"suggestive", "explicit"}
    ):
        return False
    return True


_REQUIREMENT_FIELDS = (
    ("face_only", "min_face_only", "Face only"),
    ("full_body", "min_full_body", "Full body"),
    ("front", "min_front", "Front"),
    ("side", "min_side", "Side"),
    ("back", "min_back", "Back"),
    ("male", "min_male", "Male"),
    ("female", "min_female", "Female"),
    ("face_visible", "min_face_visible", "Face visible"),
    ("body_visible", "min_body_visible", "Body visible"),
)


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


def _required_counts(requirements: MosaicRequirements) -> dict[str, int]:
    return {
        name: max(0, int(getattr(requirements, field_name)))
        for name, field_name, _label in _REQUIREMENT_FIELDS
    }


def _required_names(requirements: MosaicRequirements) -> list[str]:
    counts = _required_counts(requirements)
    return [
        name
        for name, _field, _label in _REQUIREMENT_FIELDS
        for _ in range(counts[name])
    ]


def _requirement_deficits(
    selected: list[ImageRecord],
    requirements: MosaicRequirements,
) -> dict[str, int]:
    deficits = _required_counts(requirements)
    for record in selected:
        for name in deficits:
            if deficits[name] > 0 and _matches_requirement(record, name):
                deficits[name] -= 1
    return deficits


def _format_requirement_deficit(name: str, deficit: int) -> str:
    label = next(
        label
        for field_name, _field, label in _REQUIREMENT_FIELDS
        if field_name == name
    )
    return f"{label}: need {deficit} more"


def _unmet_requirement_messages(deficits: dict[str, int]) -> list[str]:
    return [
        _format_requirement_deficit(name, deficit)
        for name, deficit in deficits.items()
        if deficit > 0
    ]


def _ensure_phash(record: ImageRecord) -> str:
    if record.phash:
        return record.phash
    if not record.path:
        return ""
    try:
        from engine.image_analyzer import compute_phash
        record.phash = compute_phash(record.path)
    except Exception as exc:
        logger.debug(
            "[layout] Could not compute pHash for %s: %s",
            record.path,
            exc,
        )
    return record.phash


def _too_similar(
    record: ImageRecord,
    selected: list[ImageRecord],
    threshold: int,
) -> bool:
    current_phash = _ensure_phash(record)
    if not current_phash:
        return False
    try:
        import imagehash
        current = imagehash.hex_to_hash(current_phash)
        for other in selected:
            other_phash = _ensure_phash(other)
            if (
                other_phash
                and (
                    current - imagehash.hex_to_hash(other_phash)
                ) <= threshold
            ):
                return True
        return False
    except Exception:
        return False


def _candidate_order(
    records: Iterable[ImageRecord],
    requirements: MosaicRequirements,
    min_image_width: int = 0,
    min_image_height: int = 0,
) -> list[ImageRecord]:
    candidates = [
        record
        for record in records
        if record.ranking
        and record.ranking.final_score > 0
        and record.detections
        and _hard_eligible(
            record,
            requirements,
            min_image_width=min_image_width,
            min_image_height=min_image_height,
        )
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
    """Select exactly target candidates while honoring required attributes."""
    target = max(0, int(target))
    if target <= 0:
        return [], []

    forced = [
        record for record in candidates
        if record.manually_included
    ]
    if len(forced) > target:
        return [], [
            f"Manual includes: {len(forced)} images exceed Target Images={target}"
        ]

    selected = list(forced)

    def base_score(record: ImageRecord) -> float:
        return float(record.ranking.final_score if record.ranking else 0.0)

    deficits = _requirement_deficits(selected, requirements)
    match_counts = {
        name: sum(
            1 for record in candidates
            if _matches_requirement(record, name)
        )
        for name, deficit in deficits.items()
        if deficit > 0
    }

    while len(selected) < target:
        has_deficit = any(value > 0 for value in deficits.values())
        available = [
            record
            for record in candidates
            if record not in selected
            and (
                record.manually_included
                or not _too_similar(record, selected, phash_threshold)
            )
        ]

        # Diversity is normally respected, but a required composition
        # constraint has priority when every matching image is blocked only
        # by pHash similarity. This keeps "Required Composition" meaningful.
        if not available and has_deficit:
            available = [
                record
                for record in candidates
                if record not in selected
                and any(
                    deficits[name] > 0 and _matches_requirement(record, name)
                    for name in deficits
                )
            ]
        if not available:
            break

        scored: list[tuple[float, float, float, str, ImageRecord]] = []
        for record in available:
            coverage = 0.0
            if has_deficit:
                for name, deficit in deficits.items():
                    if deficit > 0 and _matches_requirement(record, name):
                        scarcity = 1.0 / max(1, match_counts.get(name, 1))
                        coverage += 1.0 + 4.0 * scarcity

            score = base_score(record)
            scored.append(
                (
                    coverage,
                    1.0 if record.manually_included else 0.0,
                    score,
                    record.filename.lower(),
                    record,
                )
            )

        covering = [item for item in scored if item[0] > 0.0]
        if covering:
            pool = covering
        else:
            # There may still be a requirement-matching image blocked only by
            # similarity. Prefer it before falling back to a normal candidate.
            similar_covering = [
                item
                for item in (
                    (
                        sum(
                            (
                                1.0
                                + 4.0 / max(1, match_counts.get(name, 1))
                            )
                            for name, deficit in deficits.items()
                            if deficit > 0 and _matches_requirement(record, name)
                        ),
                        1.0 if record.manually_included else 0.0,
                        base_score(record),
                        record.filename.lower(),
                        record,
                    )
                    for record in candidates
                    if record not in selected
                    and any(
                        deficits[name] > 0 and _matches_requirement(record, name)
                        for name in deficits
                    )
                )
                if _too_similar(item[-1], selected, phash_threshold)
            ]
            pool = similar_covering or scored

        chosen = max(
            pool,
            key=lambda item: (item[0], item[1], item[2], item[3]),
        )
        record = chosen[-1]
        selected.append(record)

        for name in deficits:
            if deficits[name] > 0 and _matches_requirement(record, name):
                deficits[name] -= 1

    repair_limit = max(1, target * len(_REQUIREMENT_FIELDS))
    for _ in range(repair_limit):
        deficits = _requirement_deficits(selected, requirements)
        total_deficit = sum(deficits.values())
        if total_deficit == 0:
            break

        best_swap = None
        best_key = None
        fallback_swap = None
        fallback_key = None
        removable = [
            record for record in selected
            if not record.manually_included
        ]

        for add in candidates:
            if add in selected:
                continue
            if not any(
                deficits[name] > 0 and _matches_requirement(add, name)
                for name in deficits
            ):
                continue

            for remove in removable:
                trial = [
                    record for record in selected
                    if record is not remove
                ]
                similar = (
                    not add.manually_included
                    and _too_similar(add, trial, phash_threshold)
                )

                trial.append(add)
                trial_deficits = _requirement_deficits(
                    trial,
                    requirements,
                )
                reduction = total_deficit - sum(trial_deficits.values())
                if reduction <= 0:
                    continue

                add_score = base_score(add)
                remove_score = base_score(remove)
                key = (
                    reduction,
                    add_score - remove_score,
                    add_score,
                    add.filename.lower(),
                )
                if similar:
                    if fallback_key is None or key > fallback_key:
                        fallback_key = key
                        fallback_swap = (remove, add)
                elif best_key is None or key > best_key:
                    best_key = key
                    best_swap = (remove, add)

        # Required composition wins over diversity only when a non-similar
        # repair is unavailable.
        if best_swap is None:
            best_swap = fallback_swap
        if best_swap is None:
            break

        remove, add = best_swap
        selected.remove(remove)
        selected.append(add)

    unmet = _unmet_requirement_messages(
        _requirement_deficits(selected, requirements)
    )
    if len(selected) < target:
        unmet.append(
            f"Target Images: could select only {len(selected)} of {target} "
            f"with the current Similarity setting"
        )
    return selected, unmet

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
    allow_partial: bool = False,
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
                    # Partial AUTO may skip this image and continue through the
                    # already-selected ranked pool.
                    if allow_partial:
                        placement = None
                        break
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
            if allow_partial:
                continue
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

    # Penalize a single oversized tile from dominating the canvas. The ideal
    # area is based on the images actually placed, which also works for partial
    # AUTO layouts where some ranked images cannot fit.
    ideal_tile_area = area / max(1, len(placements))
    area_fits = []
    for placement in placements:
        tile_area = max(1.0, float(placement.width * placement.height))
        ratio = tile_area / ideal_tile_area
        area_fits.append(min(1.0, ratio, 1.0 / max(1e-9, ratio)))
    area_balance = sum(area_fits) / len(area_fits)

    layout_score = (
        0.55 * fill
        + 0.10 * extent
        + 0.20 * target_fit
        + 0.15 * area_balance
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


def _balanced_base_zooms(
    selected: list[ImageRecord],
    canvas_size: tuple[int, int],
    padding_px: int,
    min_subject_px: int,
    target_subject_px: int,
    max_zoom: float,
    target_fill: float = 0.65,
) -> dict[str, float]:
    """Choose per-image zooms from a shared tile-area budget.

    Subject target is preferred, but very wide/tall crops are reduced toward a
    common tile area so one aspect ratio cannot dominate the mosaic.
    """
    canvas_w, canvas_h = map(int, canvas_size)
    canvas_area = max(1.0, float(canvas_w * canvas_h))
    tile_budget = max(
        1.0,
        target_fill * canvas_area / max(1, len(selected)),
    )

    zooms: dict[str, float] = {}
    for record in selected:
        crop = _layout_crop(record, padding_px)
        crop_area = max(1.0, float(crop.width * crop.height))
        subject_h = max(1, _best_detection(record).bbox.height)

        min_subject_zoom = min_subject_px / float(subject_h)
        target_subject_zoom = target_subject_px / float(subject_h)
        area_zoom = (tile_budget / crop_area) ** 0.5

        # Keep the target subject size as the preferred upper bound, while
        # allowing a modest 20% area margin for better packing.
        preferred = min(
            target_subject_zoom,
            area_zoom * 1.20,
            max_zoom,
        )
        zooms[record.path] = max(
            0.1,
            min(max_zoom, max(min_subject_zoom, preferred)),
        )

    return zooms


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
        base_zooms = _balanced_base_zooms(
            order,
            canvas_size,
            padding_px,
            min_subject_px,
            target_subject_px,
            max_zoom,
        )

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



def _optimize_ranked_auto_layout(
    selected: list[ImageRecord],
    canvas_size: tuple[int, int],
    padding_px: int,
    min_subject_px: int,
    target_subject_px: int,
    max_zoom: float,
    min_zoom: float,
    zoom_decay: float,
    requirements: MosaicRequirements,
) -> LayoutEvaluation | None:
    """Place a preselected ranked pool without changing image identity.

    AUTO deliberately has two phases:
      1. Select the image pool from ranking/requirements/diversity.
      2. Compute zoom and positions for that fixed pool.

    Geometry may skip an image that cannot fit, but it never reaches back into
    the ranking phase to replace that image with a different candidate.
    """
    if not selected:
        return None

    zoom_factors = _zoom_scale_candidates(
        selected,
        canvas_size,
        padding_px,
        target_subject_px,
        min_subject_px,
        max_zoom,
    )
    base_zooms = _balanced_base_zooms(
        selected,
        canvas_size,
        padding_px,
        min_subject_px,
        target_subject_px,
        max_zoom,
    )

    best: LayoutEvaluation | None = None
    best_key: tuple[int, int, float, float, float] | None = None

    for factor in zoom_factors:
        zooms = {
            path: min(max_zoom, zoom * factor)
            for path, zoom in base_zooms.items()
        }
        evaluation = _simulate_exact_viewer(
            selected,
            zooms,
            canvas_size,
            padding_px,
            min_zoom=min_zoom,
            zoom_decay=zoom_decay,
            min_subject_px=min_subject_px,
            target_subject_px=target_subject_px,
            allow_partial=True,
        )
        if evaluation is None or not evaluation.placements:
            continue

        placed_records = [placement.record for placement in evaluation.placements]
        deficits = _requirement_deficits(placed_records, requirements)
        unmet_count = sum(deficits.values())
        evaluation.unmet_requirements = _unmet_requirement_messages(deficits)

        key = (
            1 if unmet_count == 0 else 0,
            len(evaluation.placements),
            evaluation.canvas_fill_ratio,
            evaluation.layout_score,
            -abs(evaluation.average_subject_px - target_subject_px),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = evaluation

    return best


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
    """AUTO selects the ranked image pool first, then computes placement."""
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

    # Phase 1: image identity is fixed here. Ranking, requirements, manual
    # inclusion and pHash diversity decide the pool; canvas geometry is not used.
    selected, selection_messages = _select_target_set(
        candidates,
        target=max_target,
        requirements=requirements,
        phash_threshold=phash_threshold,
    )
    # AUTO must continue when a composition requirement cannot be fully
    # satisfied. The requirement still influences ranking/selection, but an
    # unmet requirement is a constraint note, not a reason to discard the
    # entire ranked pool before placement.
    selection_errors = [
        message
        for message in selection_messages
        if message.startswith("Manual includes:")
    ]
    if not selected or selection_errors:
        messages = selection_errors or selection_messages
        return LayoutEvaluation(
            [],
            len(selected),
            0.0,
            0.0,
            0.0,
            0.0,
            messages or ["AUTO selection produced no images"],
            0.0,
        )

    logger.info(
        "AUTO selection phase: ranked_candidates=%d selected_pool=%d top=%s",
        len(candidates),
        len(selected),
        ", ".join(record.filename for record in selected[:8]),
    )
    if selection_messages:
        logger.info(
            "AUTO selection phase notes: %s",
            "; ".join(selection_messages),
        )

    # Phase 2: placement works only on the fixed ranked pool from phase 1.
    evaluation = _optimize_ranked_auto_layout(
        selected=selected,
        canvas_size=canvas_size,
        padding_px=padding_px,
        min_subject_px=min_subject_px,
        target_subject_px=target_subject_px,
        max_zoom=max_zoom,
        min_zoom=min_zoom,
        zoom_decay=zoom_decay,
        requirements=requirements,
    )
    if evaluation is None:
        return LayoutEvaluation(
            [],
            len(selected),
            0.0,
            0.0,
            0.0,
            0.0,
            ["AUTO layout could not place any image from the selected ranked pool"],
            0.0,
        )

    logger.info(
        "AUTO layout phase: selected_pool=%d placed=%d canvas=%dx%d avg_zoom=%.3f fill=%.1f%% subject=%.0fpx unmet=%s",
        len(selected),
        len(evaluation.placements),
        canvas_size[0],
        canvas_size[1],
        evaluation.average_zoom,
        evaluation.canvas_fill_ratio * 100.0,
        evaluation.average_subject_px,
        ",".join(evaluation.unmet_requirements) if evaluation.unmet_requirements else "none",
    )
    return evaluation

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
    if len(selected) != target:
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
    if unmet:
        logger.warning(
            "FIXED selection: required composition could not be fully satisfied; "
            "continuing with the selected ranked pool: %s",
            "; ".join(unmet),
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
    """Reproduce the real first-fit packing used by Preview/export."""
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
                if min_subject_px > 0:
                    subject_h = max(1, _best_detection(record).bbox.height)
                    if int(subject_h * zoom) < min_subject_px:
                        position = None
                    else:
                        break
                else:
                    break
            zoom *= zoom_decay

        if position is None:
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
        occupied.append(
            (placement.x, placement.y, placement.width, placement.height)
        )

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