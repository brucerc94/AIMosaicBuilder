"""Deterministic ranking and diversity selection."""
from __future__ import annotations

import logging
from typing import Optional

from engine.models import ImageAnalysis, ImageRecord, ImageStatus, RankingResult, RankingWeights, RejectReason

logger = logging.getLogger("ranking")


def compute_score(analysis: ImageAnalysis, weights: Optional[RankingWeights] = None) -> RankingResult:
    if analysis.reject or not analysis.has_person:
        return RankingResult(
            final_score=0.0,
            penalty_applied=100.0,
            penalty_reasons=[analysis.reject_reason or RejectReason.NO_PERSON.value],
        )

    w = weights or RankingWeights()
    components = {
        "technical_quality": analysis.image_quality * w.technical_quality,
        "composition": analysis.composition * w.composition,
        "person_visibility": analysis.person_visibility * w.person_visibility,
        "subject_quality": analysis.subject_quality * w.subject_quality,
        "sharpness": max(0.0, 1.0 - analysis.blur) * w.sharpness,
        "face_visibility": (1.0 if analysis.face_visible else 0.0) * w.face_visibility,
        "mosaic_value": analysis.mosaic_value * w.mosaic_value,
    }
    penalty = 0.0
    reasons: list[str] = []
    if analysis.blur > 0.5:
        penalty += min(0.2, (analysis.blur - 0.5) * 0.4)
        reasons.append(f"blur={analysis.blur:.2f}")
    if analysis.occluded:
        penalty += 0.15
        reasons.append("occluded")
    if analysis.person_visibility < 0.4:
        penalty += (0.4 - analysis.person_visibility) * 0.5
        reasons.append(f"low_visibility={analysis.person_visibility:.2f}")
    if not analysis.face_visible and not analysis.body_visible:
        penalty += 0.10
        reasons.append("no_face_no_body")
    if analysis.composition < 0.3:
        penalty += (0.3 - analysis.composition) * 0.2
        reasons.append(f"bad_composition={analysis.composition:.2f}")
    if analysis.image_quality < 0.3:
        penalty += (0.3 - analysis.image_quality) * 0.2
        reasons.append(f"low_quality={analysis.image_quality:.2f}")

    raw = min(1.0, sum(components.values()))
    penalty = min(0.9, penalty)
    return RankingResult(
        final_score=round(max(0.0, raw - penalty) * 100.0, 2),
        score_breakdown={key: round(value * 100.0, 2) for key, value in components.items()},
        penalty_applied=round(penalty * 100.0, 2),
        penalty_reasons=reasons,
    )


def rank_records(records: list[ImageRecord], weights: Optional[RankingWeights] = None) -> list[ImageRecord]:
    for record in records:
        if record.analysis and not record.manually_excluded:
            record.ranking = compute_score(record.analysis, weights)
            if record.manually_included:
                record.ranking.final_score = max(record.ranking.final_score, 50.0)
        else:
            record.ranking = RankingResult()

    ordered = sorted(
        records,
        key=lambda record: (
            -(record.ranking.final_score if record.ranking else 0.0),
            record.filename.lower(),
        ),
    )
    rank = 1
    for record in ordered:
        valid = bool(
            record.analysis
            and record.analysis.has_person
            and record.ranking
            and record.ranking.final_score > 0
            and not record.manually_excluded
        )
        if valid:
            record.ranking.rank = rank
            rank += 1
        elif record.ranking:
            record.ranking.rank = 0
    return ordered


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


def apply_diversity_filter(records: list[ImageRecord], target: int, phash_threshold: int = 10) -> list[ImageRecord]:
    for record in records:
        if record.status == ImageStatus.SELECTED:
            record.status = ImageStatus.ANALYZED

    candidates = [
        record for record in records
        if record.ranking
        and record.ranking.rank > 0
        and record.detections
        and not record.manually_excluded
    ]
    if target <= 0:
        return records

    manual = [record for record in candidates if record.manually_included]
    automatic = [record for record in candidates if not record.manually_included]
    selected: list[ImageRecord] = []

    for record in manual:
        if len(selected) >= target:
            break
        record.status = ImageStatus.SELECTED
        selected.append(record)

    for record in automatic:
        if len(selected) >= target:
            break
        if _too_similar(record, selected, phash_threshold):
            continue
        record.status = ImageStatus.SELECTED
        selected.append(record)

    selected_paths = {record.path for record in selected}
    for record in records:
        if (
            record.path not in selected_paths
            and record.status in {ImageStatus.ANALYZED, ImageStatus.CACHED}
            and record.ranking
            and record.ranking.rank > 0
        ):
            record.status = ImageStatus.REJECTED

    logger.info("Diversity selection: %d/%d candidates (target=%d)", len(selected), len(candidates), target)
    return records


def auto_target_count(n_valid: int, canvas_w: int = 3840, canvas_h: int = 2160, min_cell_px: int = 300) -> int:
    if n_valid <= 0:
        return 0
    cols = max(1, canvas_w // max(1, min_cell_px))
    rows = max(1, canvas_h // max(1, min_cell_px))
    limit = min(n_valid, cols * rows, 20)
    for target in (20, 16, 12, 9, 6, 4):
        if target <= limit:
            return target
    return limit
