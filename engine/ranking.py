"""Deterministic ranking, diversity and mosaic-requirement selection."""
from __future__ import annotations

import logging
from typing import Optional

from engine.models import (
    ImageAnalysis,
    ImageRecord,
    ImageStatus,
    RankingResult,
    RankingWeights,
    RejectReason,
    MosaicRequirements,
)

logger = logging.getLogger("ranking")


def _tags(record: ImageRecord) -> dict:
    return record.analysis.visual_tags if record.analysis else {}


def _hard_eligible(record: ImageRecord, requirements: MosaicRequirements) -> bool:
    analysis = record.analysis
    if not analysis or not analysis.has_person or record.manually_excluded:
        return False
    if analysis.image_quality < requirements.min_quality:
        return False
    if analysis.person_visibility < requirements.min_person_visibility:
        return False
    if requirements.exclude_blurry and analysis.blur > 0.5:
        return False
    if requirements.exclude_occluded and analysis.occluded:
        return False

    rating = str(_tags(record).get("content_rating", "unknown"))
    if requirements.nsfw_policy == "safe_only" and rating != "safe":
        return False
    if requirements.nsfw_policy == "nsfw_only" and rating not in {"suggestive", "explicit"}:
        return False
    return True


def _matches_requirement(record: ImageRecord, name: str) -> bool:
    analysis = record.analysis
    tags = _tags(record)
    if not analysis:
        return False
    if name == "face_only":
        return tags.get("framing") == "face_only"
    if name == "full_body":
        return tags.get("framing") == "full_body"
    if name == "front":
        return tags.get("orientation") == "front"
    if name == "side":
        return tags.get("orientation") == "side"
    if name == "back":
        return tags.get("orientation") in {"back", "three_quarter_back"}
    if name == "male":
        return tags.get("gender_presentation") == "male"
    if name == "female":
        return tags.get("gender_presentation") == "female"
    if name == "face_visible":
        return analysis.face_visible
    if name == "body_visible":
        return analysis.body_visible
    return False


def _preference_bonus(record: ImageRecord, requirements: MosaicRequirements) -> float:
    """Return a soft bonus in points (0..10) for user preferences."""
    if not record.analysis:
        return 0.0
    tags = _tags(record)
    bonus = 0.0
    checks = (
        (requirements.prefer_face_only, tags.get("framing") == "face_only"),
        (requirements.prefer_full_body, tags.get("framing") == "full_body"),
        (requirements.prefer_front, tags.get("orientation") == "front"),
        (requirements.prefer_side, tags.get("orientation") == "side"),
        (requirements.prefer_back, tags.get("orientation") in {"back", "three_quarter_back"}),
        (requirements.prefer_solo, record.analysis.person_count == 1),
        (requirements.prefer_face_visible, record.analysis.face_visible),
        (requirements.prefer_body_visible, record.analysis.body_visible),
    )
    for enabled, matches in checks:
        if enabled and matches:
            bonus += 1.25
    return min(10.0, bonus)


def compute_score(
    analysis: ImageAnalysis,
    weights: Optional[RankingWeights] = None,
    preference_bonus: float = 0.0,
) -> RankingResult:
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
    score = max(0.0, raw - penalty) * 100.0
    score = min(100.0, score + preference_bonus)
    return RankingResult(
        final_score=round(score, 2),
        score_breakdown={key: round(value * 100.0, 2) for key, value in components.items()},
        penalty_applied=round(penalty * 100.0, 2),
        penalty_reasons=reasons,
    )


def rank_records(
    records: list[ImageRecord],
    weights: Optional[RankingWeights] = None,
    requirements: Optional[MosaicRequirements] = None,
) -> list[ImageRecord]:
    requirements = requirements or MosaicRequirements()
    for record in records:
        if record.analysis and not record.manually_excluded:
            record.ranking = compute_score(
                record.analysis,
                weights,
                _preference_bonus(record, requirements),
            )
            if not _hard_eligible(record, requirements) and not record.manually_included:
                record.ranking.penalty_reasons.append("fails_mosaic_requirements")
                record.ranking.final_score = 0.0
            elif record.manually_included:
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
            and _hard_eligible(record, requirements)
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


def apply_diversity_filter(
    records: list[ImageRecord],
    target: int,
    phash_threshold: int = 10,
    requirements: Optional[MosaicRequirements] = None,
) -> list[ImageRecord]:
    requirements = requirements or MosaicRequirements()
    for record in records:
        if record.status == ImageStatus.SELECTED:
            record.status = ImageStatus.ANALYZED

    candidates = [
        record for record in records
        if record.ranking
        and record.ranking.final_score > 0
        and record.detections
        and not record.manually_excluded
        and _hard_eligible(record, requirements)
    ]
    if target <= 0:
        return records

    candidates.sort(
        key=lambda r: (
            -(r.ranking.final_score if r.ranking else 0.0),
            r.filename.lower(),
        )
    )

    selected: list[ImageRecord] = []
    unmet: list[str] = []

    requirement_counts = (
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

    # Reserve slots for mandatory categories first. A selected image can satisfy
    # several categories at once, so we always count it only once in the mosaic.
    for name, minimum in requirement_counts:
        if minimum <= 0:
            continue
        for _ in range(minimum):
            matching = [
                r for r in candidates
                if r not in selected
                and _matches_requirement(r, name)
                and not _too_similar(r, selected, phash_threshold)
            ]
            if not matching:
                unmet.append(name)
                break
            best = matching[0]
            best.status = ImageStatus.SELECTED
            selected.append(best)
            if len(selected) >= target:
                break
        if len(selected) >= target:
            break

    # Fill remaining positions by score while maintaining diversity.
    if len(selected) < target:
        for record in candidates:
            if len(selected) >= target:
                break
            if record in selected:
                continue
            if _too_similar(record, selected, phash_threshold):
                continue
            record.status = ImageStatus.SELECTED
            selected.append(record)

    selected_paths = {record.path for record in selected}
    for record in records:
        if (
            record.path not in selected_paths
            and record.status in {ImageStatus.ANALYZED, ImageStatus.CACHED, ImageStatus.SELECTED}
            and record.ranking
            and record.ranking.rank > 0
        ):
            record.status = ImageStatus.REJECTED

    if unmet:
        logger.warning("Mosaic requirements not fully satisfied: %s", ", ".join(unmet))
    logger.info(
        "Diversity/requirements selection: %d/%d candidates (target=%d, unmet=%s)",
        len(selected), len(candidates), target, ",".join(unmet) if unmet else "none",
    )
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
