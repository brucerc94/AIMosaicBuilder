"""Render the generated mosaic layout to a final image file."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from engine.layout_optimizer import simulate_viewer_layout
from engine.models import ImageRecord, ImageStatus


def render_mosaic_image(
    records: list[ImageRecord],
    canvas_size: tuple[int, int],
    padding_px: int,
    output_path: str,
    image_format: str | None = None,
) -> Path:
    """Render the selected mosaic at full canvas resolution.

    The placement is reconstructed with the same viewer-compatible layout helper
    used by Preview/JSON export. Only the original source images are read; no
    intermediate crop files are created.
    """
    selected = [
        record
        for record in records
        if record.status == ImageStatus.SELECTED and record.detections
    ]
    if not selected:
        raise ValueError("No selected images are available for mosaic export.")

    canvas_w, canvas_h = map(int, canvas_size)
    if canvas_w <= 0 or canvas_h <= 0:
        raise ValueError("Canvas dimensions must be positive.")

    placements = simulate_viewer_layout(
        selected,
        canvas_size=(canvas_w, canvas_h),
        padding_px=int(padding_px),
        initial_zoom=0.5,
        min_zoom=0.1,
        zoom_decay=0.9,
        min_subject_px=0,
    )
    if not placements:
        raise ValueError("Could not place any selected images on the canvas.")

    canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
    for placement in placements:
        crop = placement.crop_bbox
        with Image.open(placement.record.path) as source:
            source = source.convert("RGB")
            cropped = source.crop((crop.x, crop.y, crop.x2, crop.y2))
            cropped = cropped.resize(
                (placement.width, placement.height),
                Image.Resampling.LANCZOS,
            )
            canvas.paste(cropped, (placement.x, placement.y))

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fmt = (image_format or target.suffix.lstrip(".") or "png").upper()
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt not in {"PNG", "JPEG", "WEBP"}:
        raise ValueError("Supported mosaic formats are PNG, JPEG and WEBP.")
    save_kwargs = {"quality": 95} if fmt in {"JPEG", "WEBP"} else {}
    canvas.save(target, format=fmt, **save_kwargs)
    return target
