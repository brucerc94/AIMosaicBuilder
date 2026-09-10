"""Independent AI Mosaic Builder project export.

This format is intentionally independent of ImageMosaicView. A future adapter can
translate it to another application's project format without touching the analysis pipeline.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from engine.models import ImageRecord, ImageStatus
from vision.cropper import compute_crop_box, save_crop


def export_project(output_dir: str, records: list[ImageRecord], canvas_size: tuple[int, int], padding_px: int) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    selected_dir = root / "selected"
    crops_dir = root / "crops"
    selected_dir.mkdir(exist_ok=True)
    crops_dir.mkdir(exist_ok=True)

    entries = []
    for slot, record in enumerate((r for r in records if r.status == ImageStatus.SELECTED and r.detections)):
        detection = max(record.detections, key=lambda d: (d.is_main, d.confidence, d.relative_size))
        crop = compute_crop_box(detection.bbox, record.width, record.height, padding_px)
        source = Path(record.path)
        selected_name = f"{slot + 1:03d}_{source.name}"
        shutil.copy2(source, selected_dir / selected_name)
        crop_name = f"{slot + 1:03d}_{source.stem}_crop.jpg"
        save_crop(record.path, detection.bbox, str(crops_dir / crop_name), padding_px=padding_px, quality=92)
        entries.append({
            "slot": slot,
            "source_filename": record.filename,
            "selected_file": str(Path("selected") / selected_name),
            "crop_file": str(Path("crops") / crop_name),
            "score": record.ranking.final_score if record.ranking else 0.0,
            "rank": record.ranking.rank if record.ranking else 0,
            "person_bbox": detection.bbox.to_dict(),
            "crop_bbox": crop.to_dict(),
            "padding_px": padding_px,
        })

    payload = {
        "format": "ai-mosaic-builder",
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "canvas_size": list(canvas_size),
        "entries": entries,
    }
    path = root / "ai_mosaic_project.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
