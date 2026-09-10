"""Export AI Mosaic Builder results in ImageMosaicView-compatible format."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from engine.models import ImageRecord, ImageStatus
from vision.cropper import compute_crop_box, save_crop


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


def export_project(
    output_dir: str,
    records: list[ImageRecord],
    canvas_size: tuple[int, int],
    padding_px: int,
) -> Path:
    """
    Export a project that ImageMosaicView can load directly.

    The main ``mosaic.json`` file intentionally uses ImageMosaicView's real
    project format: a top-level list of entries. Builder-specific metadata is
    written separately so it never breaks the viewer's loader.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    selected_dir = root / "selected"
    crops_dir = root / "crops"
    selected_dir.mkdir(exist_ok=True)
    crops_dir.mkdir(exist_ok=True)

    canvas_w, canvas_h = map(int, canvas_size)
    viewer_entries: list[dict] = []
    metadata_entries: list[dict] = []

    selected_records = [
        r for r in records
        if r.status == ImageStatus.SELECTED and r.detections
    ]

    for slot, record in enumerate(selected_records):
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
        relative_coords = _relative_coords(crop, record.width, record.height)

        source = Path(record.path)
        selected_name = f"{slot + 1:03d}_{source.name}"
        selected_path = selected_dir / selected_name
        shutil.copy2(source, selected_path)

        crop_name = f"{slot + 1:03d}_{source.stem}_crop.jpg"
        crop_path = crops_dir / crop_name
        save_crop(
            record.path,
            detection.bbox,
            str(crop_path),
            padding_px=padding_px,
            quality=92,
        )

        # EXACT viewer contract. Do not add builder metadata here.
        viewer_entries.append({
            "type": "body",
            "filename": (Path("selected") / selected_name).as_posix(),
            "coords": relative_coords,
            "zoom": 0.5,
            "canvas_size": [canvas_w, canvas_h],
        })

        metadata_entries.append({
            "slot": slot,
            "source_filename": record.filename,
            "selected_file": (Path("selected") / selected_name).as_posix(),
            "crop_file": (Path("crops") / crop_name).as_posix(),
            "score": record.ranking.final_score if record.ranking else 0.0,
            "rank": record.ranking.rank if record.ranking else 0,
            "person_bbox": detection.bbox.to_dict(),
            "crop_bbox": crop.to_dict(),
            "coords": relative_coords,
            "padding_px": padding_px,
        })

    # This is the file that should be opened by ImageMosaicView.
    mosaic_path = root / "mosaic.json"
    mosaic_path.write_text(
        json.dumps(viewer_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Keep application-specific data separate from the viewer project format.
    metadata_payload = {
        "format": "ai-mosaic-builder",
        "version": 2,
        "created_at": datetime.now().isoformat(),
        "canvas_size": [canvas_w, canvas_h],
        "entries": metadata_entries,
    }
    metadata_path = root / "ai_mosaic_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return mosaic_path
