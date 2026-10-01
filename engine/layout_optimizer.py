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


def _hard_eligible(
    record: ImageRecord,
    requirements: MosaicRequirements,
    min_image_width: int = 0,
    min_image_height: int = 0,
) -> bool:
    analysis = record.analysis
    if not analysis or not analysis.has_person or record.manually_excluded:
        return False
    min_width = max(0, int(min_image_width))
    min_height = max(0, int(min_image_height))
    if min_width > 0 and record.width < min_width:
        return False
    if min_height > 0 and record.height < min_height:
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


def _required_counts(requirements: MosaicRequirements) -> dict[str, int]:
    return {
        name: max(0, int(getattr(requirements, field_name)))
        for name, field_name, _label in _REQUIREMENT_FIELDS
    }


def _required_names(requirements: MosaicRequirements) -> list[str]:
    counts = _required_counts(requirements)
    return [name for name, _field, _label in _REQUIREMENT_FIELDS for _ in range(counts[name])]


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
    label = next(label for field_name, _field, label in _REQUIREMENT_FIELDS if field_name == name)
    return f"{label}: need {deficit} more"


def _unmet_requirement_messages(deficits: dict[str, int]) -> list[str]:
    return [
        _format_requirement_deficit(name, deficit)
        for name, deficit in deficits.items()
        if deficit > 0
    ]


def _ensure_phash(record: ImageRecord) -> str:
    """Return a record's pHash, computing it lazily for cache/session-loaded records."""
    if record.phash:
        return record.phash
    if not record.path:
        return ""
    try:
        from engine.image_analyzer import compute_phash
        record.phash = compute_phash(record.path)
    except Exception as exc:
        logger.debug("[layout] Could not compute pHash for %s: %s", record.path, exc)
    return record.phash


def _too_similar(record: ImageRecord, selected: list[ImageRecord], threshold: int) -> bool:
    current_phash = _ensure_phash(record)
    if not current_phash:
        return False
    try:
        import imagehash
        current = imagehash.hex_to_hash(current_phash)
        for other in selected:
            other_phash = _ensure_phash(other)
            if other_phash and (current - imagehash.hex_to_hash(other_phash)) <= threshold:
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

    forced = [record for record in candidates if record.manually_included]
    if len(forced) > target:
        return [], [f"Manual includes: {len(forced)} images exceed Target Images={target}"]

    selected: list[ImageRecord] = list(forced)

    def base_score(record: ImageRecord) -> float:
        return float(record.ranking.final_score if record.ranking else 0.0)

    deficits = _requirement_deficits(selected, requirements)
    match_counts = {
        name: sum(1 for record in candidates if _matches_requirement(record, name))
        for name, deficit in deficits.items()
        if deficit > 0
    }

    while len(selected) < target:
        available = [
            record for record in candidates
            if record not in selected
            and (record.manually_included or not _too_similar(record, selected, phash_threshold))
        ]
        if not available:
            break

        scored: list[tuple[float, float, float, str, ImageRecord]] = []
        has_deficit = any(value > 0 for value in deficits.values())
        for record in available:
            coverage = 0.0
            if has_deficit:
                for name, deficit in deficits.items():
                    if deficit > 0 and _matches_requirement(record, name):
                        scarcity = 1.0 / max(1, match_counts.get(name, 1))
                        coverage += 1.0 + 4.0 * scarcity
            score = base_score(record)
            scored.append((coverage, 1.0 if record.manually_included else 0.0, score, record.filename.lower(), record))

        covering = [item for item in scored if item[0] > 0.0]
        pool = covering if covering else scored
        chosen = max(pool, key=lambda item: (item[0], item[1], item[2], item[3]))
        record = chosen[-1]
        selected.append(record)
        for name in deficits:
            if deficits[name] > 0 and _matches_requirement(record, name):
                deficits[name] -= 1

    # Repair greedy choices that left a requirement short. Try replacing one
    # automatically selected image with one that improves requirement coverage.
    repair_limit = max(1, target * len(_REQUIREMENT_FIELDS))
    for _ in range(repair_limit):
        deficits = _requirement_deficits(selected, requirements)
        total_deficit = sum(deficits.values())
        if total_deficit == 0:
            break
        best_swap = None
        best_key = None
        removable = [record for record in selected if not record.manually_included]
        for add in candidates:
            if add in selected or not any(deficits[n] > 0 and _matches_requirement(add, n) for n in deficits):
                continue
            for remove in removable:
                trial = [record for record in selected if record is not remove]
                if not add.manually_included and _too_similar(add, trial, phash_threshold):
                    continue
                trial.append(add)
                trial_deficits = _requirement_deficits(trial, requirements)
                reduction = total_deficit - sum(trial_deficits.values())
                if reduction <= 0:
                    continue
                add_score = base_score(add)
                remove_score = base_score(remove)
                key = (reduction, add_score - remove_score, add_score, add.filename.lower())
                if best_key is None or key > best_key:
                    best_key = key
                    best_swap = (remove, add)
        if best_swap is None:
            break
        remove, add = best_swap
        selected.remove(remove)
        selected.append(add)

    deficits = _requirement_deficits(selected, requirements)
    unmet = _unmet_requirement_messages(deficits)
    if len(selected) < target:
        unmet.append(f"Target Images: could select only {len(selected)} of {target} with the current Similarity setting")
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
    min_subject_px: int = 0,
) -> _Row | None:
    if not records:
        return None

    crops = [(record, _layout_crop(record, padding_px)) for record in records]
    ratio_sum = sum(crop.width / max(1.0, crop.height) for _, crop in crops)
    if ratio_sum <= 0:
        return None

    available = max(1.0, float(canvas_w - gap_px * max(0, len(records) - 1)))
    ideal_height = available / ratio_sum

    min_heights = [
        min_zoom * crop.height
        for _, crop in crops
    ]
    if min_subject_px > 0:
        min_heights.extend(
            min_subject_px
            * crop.height
            / max(1.0, float(_best_detection(record).bbox.height))
            for record, crop in crops
        )

    min_height = max(min_heights, default=min_zoom)
    max_height = min(
        max_zoom * crop.height
        for _, crop in crops
    )
    if min_height > max_height + 1e-6:
        return None

    row_height = max(min_height, min(max_height, ideal_height))
    zooms = {
        record.path: row_height / max(1.0, crop.height)
        for record, crop in crops
    }
    width = sum(
        crop.width * zooms[record.path]
        for record, crop in crops
    ) + gap_px * max(0, len(records) - 1)

    if width > canvas_w + 1.0:
        return None

    return _Row(records, row_height, zooms, width)


def _build_rows_greedy(
    order: list[ImageRecord],
    canvas_w: int,
    canvas_h: int,
    padding_px: int,
    desired_rows: int,
    gap_px: int,
    min_zoom: float,
    max_zoom: float,
    min_subject_px: int,
) -> list[_Row] | None:
    """Fast row builder used for larger selections."""
    rows: list[_Row] = []
    current: list[ImageRecord] = []

    for record in order:
        trial = current + [record]
        trial_row = _row_for(
            trial,
            canvas_w,
            padding_px,
            gap_px,
            min_zoom,
            max_zoom,
            min_subject_px,
        )
        if trial_row is None:
            if not current:
                return None
            finalized = _row_for(
                current,
                canvas_w,
                padding_px,
                gap_px,
                min_zoom,
                max_zoom,
                min_subject_px,
            )
            if finalized is None:
                return None
            rows.append(finalized)
            current = [record]
            continue
        current = trial

    if current:
        finalized = _row_for(
            current,
            canvas_w,
            padding_px,
            gap_px,
            min_zoom,
            max_zoom,
            min_subject_px,
        )
        if finalized is None:
            return None
        rows.append(finalized)

    if len(rows) != desired_rows:
        return None

    total_height = sum(row.height for row in rows) + gap_px * max(0, len(rows) - 1)
    if total_height <= 0 or total_height > canvas_h + 1e-6:
        return None

    return rows


def _build_rows_partitioned(
    order: list[ImageRecord],
    canvas_w: int,
    canvas_h: int,
    padding_px: int,
    desired_rows: int,
    gap_px: int,
    min_zoom: float,
    max_zoom: float,
    min_subject_px: int,
) -> list[_Row] | None:
    """Find a contiguous partition into exactly desired_rows that fits the canvas."""
    n = len(order)
    desired_rows = max(1, min(int(desired_rows), n))
    if n == 0:
        return None

    target_height = canvas_h / desired_rows
    row_cache: dict[tuple[int, int], _Row | None] = {}

    def get_row(start: int, end: int) -> _Row | None:
        key = (start, end)
        if key not in row_cache:
            row_cache[key] = _row_for(
                order[start:end],
                canvas_w,
                padding_px,
                gap_px,
                min_zoom,
                max_zoom,
                min_subject_px,
            )
        return row_cache[key]

    # Keep several non-dominated alternatives per state. Keeping only the
    # highest current height can hide a lower-height prefix that is necessary
    # to fit the remaining rows inside the canvas.
    state_limit = 1000000 if n <= 20 else 8
    states: dict[tuple[int, int], list[tuple[float, float, list[_Row]]]] = {
        (0, 0): [(0.0, 0.0, [])]
    }

    for rows_used in range(1, desired_rows + 1):
        for end in range(rows_used, n + 1):
            candidates_state: list[tuple[float, float, list[_Row]]] = []

            for start in range(rows_used - 1, end):
                previous_states = states.get((rows_used - 1, start), [])
                if not previous_states:
                    continue

                row = get_row(start, end)
                if row is None:
                    continue

                for previous_total, previous_imbalance, previous_rows in previous_states:
                    total_height = previous_total + row.height
                    if rows_used > 1:
                        total_height += gap_px
                    if total_height > canvas_h + 1e-6:
                        continue

                    imbalance = previous_imbalance + abs(row.height - target_height)
                    candidates_state.append(
                        (total_height, imbalance, previous_rows + [row])
                    )

            if not candidates_state:
                continue

            # Deduplicate equivalent totals and retain both high-fill and
            # low-height solutions, with a bias toward balanced row heights.
            candidates_state.sort(
                key=lambda item: (-item[0], item[1])
            )
            kept: list[tuple[float, float, list[_Row]]] = []
            seen_totals: set[int] = set()

            for candidate in candidates_state:
                total_height, imbalance, _rows = candidate
                total_key = round(total_height * 10.0)
                if total_key in seen_totals:
                    continue
                seen_totals.add(total_key)
                kept.append(candidate)

                if len(kept) >= state_limit:
                    break

            states[(rows_used, end)] = kept

    final_states = states.get((desired_rows, n), [])
    if not final_states:
        return None

    # Highest usable height fills the canvas best; balance breaks ties.
    final_states.sort(key=lambda item: (-item[0], item[1]))
    return final_states[0][2]

def _build_rows(
    order: list[ImageRecord],
    canvas_w: int,
    canvas_h: int,
    padding_px: int,
    desired_rows: int,
    gap_px: int = 2,
    min_zoom: float = 0.1,
    max_zoom: float = 3.0,
    min_subject_px: int = 0,
) -> list[_Row] | None:
    """Build canvas-fitting justified rows with a robust partition fallback."""
    if not order:
        return None

    desired_rows = max(1, min(int(desired_rows), len(order)))

    greedy = _build_rows_greedy(
        order,
        canvas_w,
        canvas_h,
        padding_px,
        desired_rows,
        gap_px,
        min_zoom,
        max_zoom,
        min_subject_px,
    )

    if len(order) <= 30:
        partitioned = _build_rows_partitioned(
            order,
            canvas_w,
            canvas_h,
            padding_px,
            desired_rows,
            gap_px,
            min_zoom,
            max_zoom,
            min_subject_px,
        )
        if greedy is None:
            return partitioned
        if partitioned is None:
            return greedy

        greedy_height = (
            sum(row.height for row in greedy)
            + gap_px * max(0, len(greedy) - 1)
        )
        partitioned_height = (
            sum(row.height for row in partitioned)
            + gap_px * max(0, len(partitioned) - 1)
        )
        target_height = canvas_h / max(1, desired_rows)

        greedy_imbalance = sum(abs(row.height - target_height) for row in greedy)
        partitioned_imbalance = sum(abs(row.height - target_height) for row in partitioned)

        if (
            partitioned_height > greedy_height + 1e-6
            or (
                abs(partitioned_height - greedy_height) <= 1e-6
                and partitioned_imbalance < greedy_imbalance - 1e-6
            )
        ):
            return partitioned
        return greedy

    return greedy



def _evaluate_rows(
    rows: list[_Row],
    canvas_size: tuple[int, int],
    padding_px: int,
    min_zoom: float,
    min_subject_px: int,
    target_subject_px: int,
    gap_px: int = 2,
) -> LayoutEvaluation | None:
    """Evaluate the exact row geometry against the configured canvas."""
    canvas_w, canvas_h = map(int, canvas_size)
    if canvas_w <= 0 or canvas_h <= 0 or not rows:
        return None

    prepared_rows: list[tuple[list[tuple[ImageRecord, BoundingBox, float, int, int]], int, int]] = []
    total_height = 0

    for row in rows:
        actual_sizes: list[tuple[ImageRecord, BoundingBox, float, int, int]] = []
        actual_row_width = gap_px * max(0, len(row.records) - 1)

        for record in row.records:
            zoom = max(min_zoom, float(row.zooms.get(record.path, min_zoom)))
            crop = _layout_crop(record, padding_px)
            width = max(1, int(round(crop.width * zoom)))
            height = max(1, int(round(crop.height * zoom)))
            actual_sizes.append((record, crop, zoom, width, height))
            actual_row_width += width

            subject_h = max(1, _best_detection(record).bbox.height)
            if int(subject_h * zoom) < min_subject_px:
                return None

        if not actual_sizes or actual_row_width > canvas_w:
            return None

        actual_row_height = max(item[4] for item in actual_sizes)
        prepared_rows.append((actual_sizes, actual_row_width, actual_row_height))
        total_height += actual_row_height

    total_height += gap_px * max(0, len(prepared_rows) - 1)
    if total_height > canvas_h:
        return None

    # Center the complete optimized mosaic inside the configured canvas when
    # the content does not need all available vertical space.
    y = max(0.0, (canvas_h - total_height) / 2.0)
    placements: list[LayoutPlacement] = []
    occupied: list[tuple[int, int, int, int]] = []

    for row_index, (actual_sizes, actual_row_width, actual_row_height) in enumerate(prepared_rows):
        x = max(0.0, (canvas_w - actual_row_width) / 2.0)

        for index, (record, crop, zoom, width, height) in enumerate(actual_sizes):
            pos_x = int(round(x))
            pos_y = int(round(y))
            if pos_x < 0 or pos_y < 0 or pos_x + width > canvas_w or pos_y + height > canvas_h:
                return None

            if any(
                not (
                    pos_x + width <= ox
                    or pos_x >= ox + ow
                    or pos_y + height <= oy
                    or pos_y >= oy + oh
                )
                for ox, oy, ow, oh in occupied
            ):
                return None

            placements.append(
                LayoutPlacement(
                    record=record,
                    crop_bbox=crop,
                    zoom=round(zoom, 6),
                    width=width,
                    height=height,
                    x=pos_x,
                    y=pos_y,
                )
            )
            occupied.append((pos_x, pos_y, width, height))
            x += width
            if index < len(actual_sizes) - 1:
                x += gap_px

        y += actual_row_height
        if row_index < len(prepared_rows) - 1:
            y += gap_px

    if not placements:
        return None

    area = max(1, canvas_w * canvas_h)
    occupied_area = sum(p.width * p.height for p in placements)
    max_x = max(p.x + p.width for p in placements)
    max_y = max(p.y + p.height for p in placements)
    fill = max(0.0, min(1.0, occupied_area / area))
    extent = max(0.0, min(1.0, (max_x / canvas_w) * (max_y / canvas_h)))

    average_subject = sum(
        int(_best_detection(p.record).bbox.height * p.zoom)
        for p in placements
    ) / len(placements)
    target = max(1.0, float(target_subject_px))
    readability = max(0.0, min(1.25, average_subject / target))
    subject_fits = []
    for placement in placements:
        subject_px = max(
            1.0,
            float(_best_detection(placement.record).bbox.height) * placement.zoom,
        )
        relative_error = abs(subject_px - target) / target
        subject_fits.append(1.0 / (1.0 + relative_error))
    target_fit = sum(subject_fits) / len(subject_fits)

    return LayoutEvaluation(
        placements=placements,
        target=len(placements),
        average_zoom=sum(p.zoom for p in placements) / len(placements),
        average_fill_ratio=sum((p.width * p.height) / area for p in placements) / len(placements),
        average_subject_px=average_subject,
        canvas_fill_ratio=fill,
        unmet_requirements=[],
        layout_score=(
            0.55 * fill
            + 0.25 * extent
            + 0.10 * readability
            + 0.10 * target_fit
        ),
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
    # Necessary capacity check: if the minimum per-image footprint required
    # by Min Subject already exceeds the canvas area, no row arrangement can fit.
    canvas_area = max(1, canvas_w * canvas_h)
    minimum_required_area = 0.0
    for record in selected:
        crop = _layout_crop(record, padding_px)
        subject_h = max(1, _best_detection(record).bbox.height)
        minimum_zoom = max(
            0.1,
            min_subject_px / float(subject_h),
        )
        minimum_required_area += (
            crop.width * minimum_zoom
        ) * (
            crop.height * minimum_zoom
        )
    if minimum_required_area > canvas_area * 1.001:
        logger.info(
            "FIXED layout: minimum subject footprint %.1f%% of canvas area "
            "already exceeds capacity for %d images",
            minimum_required_area / canvas_area * 100.0,
            len(selected),
        )
        return None

    best: LayoutEvaluation | None = None
    # Every row must be at least Min Subject tall. Include the inter-row
    # gap in the bound so we do not search impossible row counts.
    row_floor = max(1, min_subject_px)
    max_rows = min(
        len(selected),
        max(1, (canvas_h + 2) // (row_floor + 2)),
    )

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
                min_subject_px=min_subject_px,
            )
            if rows is None:
                continue
            evaluation = _evaluate_rows(
                rows,
                canvas_size,
                padding_px,
                min_zoom=0.1,
                min_subject_px=min_subject_px,
                target_subject_px=target_subject_px,
                gap_px=2,
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