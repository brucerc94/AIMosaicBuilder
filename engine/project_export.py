"""Export AI Mosaic Builder results in ImageMosaicView-compatible format."""
from __future__ import annotations

import json
import os
from pathlib import Path

from engine.layout_optimizer import simulate_viewer_layout
from engine.models import ImageRecord, ImageStatus
from vision.cropper import compute_crop_box


def _relative_coords(box, width: int, height: int) -> list[float]:
    """Convert a pixel-space crop box to ImageMosaicView's 0..1 coordinates."""
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    return [
        max(0.0, min(1.0, box.x / width)),
        max(0.0, min(1.0, box.y / height)),
        max(0.0, min(1.0, box.x2 / width)),
        max(0.0, min(1.0, box.y2 / height)),
    ]


def _relative_source_filename(source: Path, project_root: Path) -> str:
    """Return the original image path relative to the folder containing mosaic.json."""
    source_abs = Path(os.path.abspath(source))
    root_abs = Path(os.path.abspath(project_root))
    try:
        relative = source_abs.relative_to(root_abs)
    except ValueError as exc:
        raise ValueError(
            f"Selected image is outside the source/project folder: {source_abs}"
        ) from exc
    return relative.as_posix()


def export_project(
    output_dir: str,
    records: list[ImageRecord],
    canvas_size: tuple[int, int],
    padding_px: int,
) -> Path:
    """Export only the JSON consumed by ImageMosaicView.

    No cropped image files are created. The export stores original-image
    filenames, crop coordinates and the per-image zoom chosen by the layout
    optimizer. ImageMosaicView reconstructs the crop dynamically.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    canvas_w, canvas_h = map(int, canvas_size)

    selected_records = [
        record for record in records
        if record.status == ImageStatus.SELECTED and record.detections
    ]
    if not selected_records:
        raise ValueError("No selected images are available for export.")

    # Prefer the zoom/crop produced by the optimizer. For older session records
    # without that metadata, simulate ImageMosaicView's packing now.
    fallback_layout = simulate_viewer_layout(
        selected_records,
        canvas_size=(canvas_w, canvas_h),
        padding_px=padding_px,
        initial_zoom=0.5,
        min_zoom=0.1,
        zoom_decay=0.9,
        min_subject_px=0,
    )
    fallback_by_path = {placement.record.path: placement for placement in fallback_layout}

    ordered_records = sorted(
        selected_records,
        key=lambda record: (
            record.selection.slot_index if record.selection else 10**9,
            record.filename.lower(),
        ),
    )

    viewer_entries: list[dict] = []
    for record in ordered_records:
        crop = record.selection.crop_bbox if record.selection and record.selection.crop_bbox else None
        zoom = record.selection.zoom if record.selection else None
        fallback = fallback_by_path.get(record.path)
        if crop is None and fallback is not None:
            crop = fallback.crop_bbox
        if zoom is None and fallback is not None:
            zoom = fallback.zoom
        if crop is None:
            detection = max(record.detections, key=lambda d: (d.is_main, d.confidence, d.relative_size))
            crop = compute_crop_box(detection.bbox, record.width, record.height, padding_px)
        if zoom is None:
            zoom = 0.5

        viewer_entries.append({
            "type": "body",
            "filename": _relative_source_filename(Path(record.path), root),
            "coords": _relative_coords(crop, record.width, record.height),
            "zoom": float(zoom),
            "canvas_size": [canvas_w, canvas_h],
        })

    path = root / "mosaic.json"
    path.write_text(
        json.dumps(viewer_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
