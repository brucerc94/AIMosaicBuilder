"""Export AI Mosaic Builder results in ImageMosaicView-compatible format."""
from __future__ import annotations

import json
import os
from pathlib import Path

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
    """
    Export ONLY the project JSON consumed by ImageMosaicView.

    ImageMosaicView does the crop dynamically when it loads the project. The
    builder therefore must not copy images, create crop files, or create an
    auxiliary project format. It only records the original image path,
    relative crop coordinates, zoom, and canvas size.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    canvas_w, canvas_h = map(int, canvas_size)
    viewer_entries: list[dict] = []

    selected_records = [
        r for r in records
        if r.status == ImageStatus.SELECTED and r.detections
    ]

    for record in selected_records:
        detection = max(
            record.detections,
            key=lambda d: (d.is_main, d.confidence, d.relative_size),
        )
        crop = compute_crop_box(
            detection.bbox,
            record.width,
            record.height,
            padding_px,
        )

        viewer_entries.append({
            "type": "body",
            "filename": _relative_source_filename(Path(record.path), root),
            "coords": _relative_coords(crop, record.width, record.height),
            "zoom": 0.5,
            "canvas_size": [canvas_w, canvas_h],
        })

    path = root / "mosaic.json"
    path.write_text(
        json.dumps(viewer_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
