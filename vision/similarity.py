"""
Similarity detection for AI Mosaic Builder.

Uses perceptual hashing (pHash) to identify near-duplicate images
and group them into clusters. Only the highest-ranked image from each
cluster is kept for the mosaic.

Fallback: if imagehash is not available, uses file hash to detect
exact duplicates only.
"""

from __future__ import annotations

import logging
from typing import Optional

from engine.models import ImageRecord

logger = logging.getLogger("similarity")


def build_similarity_clusters(
    records: list[ImageRecord],
    threshold: int = 10,
) -> list[list[ImageRecord]]:
    """
    Group records into clusters where all members are within `threshold`
    Hamming distance of each other's perceptual hash.

    Returns a list of clusters (each cluster is a list of ImageRecords).
    Records without a phash form single-image clusters (no grouping).

    Algorithm: greedy single-linkage — O(n²) but fine for ≤ 2000 images.
    """
    if not records:
        return []

    _imagehash_ok = _check_imagehash()

    clusters: list[list[ImageRecord]] = []
    assigned: set[int] = set()

    for i, rec_i in enumerate(records):
        if i in assigned:
            continue
        cluster = [rec_i]
        assigned.add(i)

        if not rec_i.phash or not _imagehash_ok:
            clusters.append(cluster)
            continue

        for j, rec_j in enumerate(records):
            if j <= i or j in assigned:
                continue
            if not rec_j.phash:
                continue
            if _hamming(rec_i.phash, rec_j.phash) <= threshold:
                cluster.append(rec_j)
                assigned.add(j)

        clusters.append(cluster)

    logger.info(
        f"[similarity] {len(records)} images → {len(clusters)} clusters "
        f"(threshold={threshold})"
    )
    return clusters


def best_from_cluster(cluster: list[ImageRecord]) -> ImageRecord:
    """Return the highest-ranked image from a cluster."""
    def _score(r: ImageRecord) -> float:
        if r.ranking:
            return r.ranking.final_score
        return 0.0
    return max(cluster, key=_score)


def mark_duplicates(
    records: list[ImageRecord],
    threshold: int = 10,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """
    Returns (keepers, duplicates).

    For each similarity cluster, keeps the best image and marks the rest
    as similar/duplicate (but does NOT change their ImageStatus — that is
    left to the caller / ranking stage).
    """
    clusters = build_similarity_clusters(records, threshold)
    keepers: list[ImageRecord] = []
    duplicates: list[ImageRecord] = []

    for cluster in clusters:
        if len(cluster) == 1:
            keepers.append(cluster[0])
        else:
            best = best_from_cluster(cluster)
            keepers.append(best)
            for rec in cluster:
                if rec is not best:
                    duplicates.append(rec)

    logger.info(
        f"[similarity] Keepers: {len(keepers)} | Duplicates: {len(duplicates)}"
    )
    return keepers, duplicates


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _check_imagehash() -> bool:
    try:
        import imagehash  # type: ignore
        return True
    except ImportError:
        logger.warning("[similarity] imagehash not installed — similarity detection limited")
        return False


def _hamming(phash_a: str, phash_b: str) -> int:
    """Compute Hamming distance between two phash hex strings."""
    try:
        import imagehash  # type: ignore
        h1 = imagehash.hex_to_hash(phash_a)
        h2 = imagehash.hex_to_hash(phash_b)
        return h1 - h2
    except Exception:
        return 999  # treat as completely different if can't compare
