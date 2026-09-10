"""
Ranking engine for AI Mosaic Builder.

Combines LLM analysis scores with configurable weights to produce a
final_score (0–100) for each image. Also penalises images for specific
quality issues (blur, occlusion, person too small, etc.).

After ranking, a diversity pass removes near-duplicate images using
perceptual hash (imagehash), keeping only the best from each "cluster".
"""

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
)

logger = logging.getLogger("ranking")


# ─── Scoring ──────────────────────────────────────────────────────────────────

def compute_score(
    analysis: ImageAnalysis,
    weights: Optional[RankingWeights] = None,
) -> RankingResult:
    """
    Compute a final_score (0–100) from an ImageAnalysis.
    Returns a RankingResult with breakdown and penalties.
    """
    w = weights or RankingWeights()

    # Guard: rejected images get score 0
    if analysis.reject or not analysis.has_person:
        return RankingResult(
            final_score=0.0,
            score_breakdown={},
            penalty_applied=100.0,
            penalty_reasons=[analysis.reject_reason or RejectReason.NO_PERSON.value],
        )

    # ── Raw weighted components (all 0–1) ──────────────────────────────────────
    sharpness   = max(0.0, 1.0 - analysis.blur)
    face_score  = 1.0 if analysis.face_visible else 0.0

    breakdown = {
        "technical_quality": analysis.image_quality  * w.technical_quality,
        "composition":       analysis.composition    * w.composition,
        "person_visibility": analysis.person_visibility * w.person_visibility,
        "subject_quality":   analysis.subject_quality * w.subject_quality,
        "sharpness":         sharpness              * w.sharpness,
        "face_visibility":   face_score             * w.face_visibility,
        "mosaic_value":      analysis.mosaic_value  * w.mosaic_value,
    }

    raw_score = sum(breakdown.values())  # 0–1

    # ── Penalties ──────────────────────────────────────────────────────────────
    penalty = 0.0
    reasons = []

    if analysis.blur > 0.5:
        p = (analysis.blur - 0.5) * 0.4   # up to 0.2 penalty
        penalty += p
        reasons.append(f"blur={analysis.blur:.2f}")

    if analysis.occluded:
        penalty += 0.15
        reasons.append("occluded")

    if analysis.person_visibility < 0.4:
        p = (0.4 - analysis.person_visibility) * 0.5
        penalty += p
        reasons.append(f"low_visibility={analysis.person_visibility:.2f}")

    if not analysis.face_visible and not analysis.body_visible:
        penalty += 0.10
        reasons.append("no_face_no_body")

    if analysis.composition < 0.3:
        p = (0.3 - analysis.composition) * 0.2
        penalty += p
        reasons.append(f"bad_composition={analysis.composition:.2f}")

    if analysis.image_quality < 0.3:
        p = (0.3 - analysis.image_quality) * 0.2
        penalty += p
        reasons.append(f"low_quality={analysis.image_quality:.2f}")

    # Clamp penalty
    penalty = min(penalty, 0.9)

    final_raw = max(0.0, raw_score - penalty)
    final_score = round(final_raw * 100, 2)

    return RankingResult(
        final_score=final_score,
        score_breakdown={k: round(v * 100, 2) for k, v in breakdown.items()},
        penalty_applied=round(penalty * 100, 2),
        penalty_reasons=reasons,
    )


# ─── Rank all records ─────────────────────────────────────────────────────────

def rank_records(
    records: list[ImageRecord],
    weights: Optional[RankingWeights] = None,
) -> list[ImageRecord]:
    """
    Score all valid records, assign ranks (1 = best), return sorted list.
    Records without analysis or with reject=True get rank 0 / score 0.
    """
    for record in records:
        if record.analysis and not record.manually_excluded:
            record.ranking = compute_score(record.analysis, weights)
        else:
            record.ranking = RankingResult(final_score=0.0)

    # Apply manual overrides
    for record in records:
        if record.manually_included and record.ranking:
            record.ranking.final_score = max(record.ranking.final_score, 50.0)
        if record.manually_excluded and record.ranking:
            record.ranking.final_score = 0.0

    # Sort descending by score
    scored = sorted(records, key=lambda r: (r.ranking.final_score if r.ranking else 0.0), reverse=True)

    # Assign rank (only to images with person + positive score)
    rank = 1
    for record in scored:
        if record.ranking and record.ranking.final_score > 0 and record.analysis and record.analysis.has_person:
            record.ranking.rank = rank
            rank += 1
        else:
            if record.ranking:
                record.ranking.rank = 0

    logger.info(f"[ranking] Ranked {rank - 1} valid images out of {len(records)} total")
    return scored


# ─── Diversity pass ───────────────────────────────────────────────────────────

def apply_diversity_filter(
    records: list[ImageRecord],
    target: int,
    phash_threshold: int = 10,
) -> list[ImageRecord]:
    """
    From a ranked list, select up to `target` images while maximising diversity.

    Algorithm:
      1. Only consider images with ranking.rank > 0 (has person, score > 0).
      2. Iterate in rank order (best first).
      3. For each candidate, check if it's too similar (phash distance ≤ threshold)
         to any already-selected image.
      4. If not too similar, add to selection.
      5. Stop when `target` images are selected.

    Returns the full list with ImageStatus.SELECTED set on chosen images.
    All others remain at their current status.
    """
    if target <= 0:
        return records

    candidates = [
        r for r in records
        if r.ranking and r.ranking.rank > 0
        and not r.manually_excluded
    ]

    # Put manually included images first
    manual = [r for r in candidates if r.manually_included]
    auto   = [r for r in candidates if not r.manually_included]

    selected: list[ImageRecord] = []
    selected_phashes: list[str] = []

    def _is_too_similar(rec: ImageRecord) -> bool:
        if not rec.phash or not selected_phashes:
            return False
        try:
            import imagehash  # type: ignore
            h = imagehash.hex_to_hash(rec.phash)
            for existing_phash in selected_phashes:
                eh = imagehash.hex_to_hash(existing_phash)
                if (h - eh) <= phash_threshold:
                    return True
        except Exception:
            pass
        return False

    # First: include all manual overrides (up to target)
    for rec in manual:
        if len(selected) >= target:
            break
        rec.status = ImageStatus.SELECTED
        selected.append(rec)
        if rec.phash:
            selected_phashes.append(rec.phash)

    # Then: fill remaining slots with diversity-filtered auto candidates
    for rec in auto:
        if len(selected) >= target:
            break
        if _is_too_similar(rec):
            logger.debug(f"[ranking] Diversity filter excluded: {rec.filename}")
            continue
        rec.status = ImageStatus.SELECTED
        selected.append(rec)
        if rec.phash:
            selected_phashes.append(rec.phash)

    # Mark remaining valid candidates as rejected (similar or not needed)
    selected_paths = {r.path for r in selected}
    for rec in records:
        if rec.ranking and rec.ranking.rank > 0 and rec.path not in selected_paths:
            if rec.status == ImageStatus.ANALYZED or rec.status == ImageStatus.CACHED:
                rec.status = ImageStatus.REJECTED

    logger.info(
        f"[ranking] Diversity filter: {len(selected)} selected "
        f"(target={target}, candidates={len(candidates)}, threshold={phash_threshold})"
    )
    return records


# ─── Automatic target count ───────────────────────────────────────────────────

def auto_target_count(
    n_valid: int,
    canvas_w: int = 3840,
    canvas_h: int = 2160,
    min_cell_px: int = 300,
) -> int:
    """
    Suggest a target image count based on how many valid images exist
    and what the canvas can display at a reasonable minimum cell size.
    """
    if n_valid <= 0:
        return 0

    # Max images that fit at minimum cell size
    cols = canvas_w // min_cell_px
    rows = canvas_h // min_cell_px
    max_cells = cols * rows

    # Prefer square-ish grids
    preferred = [4, 6, 9, 12, 16, 20]
    for t in preferred:
        if t <= n_valid and t <= max_cells:
            result = t
    else:
        result = min(n_valid, max_cells, 20)

    return max(result, 4)
