"""
Auto-crop engine for AI Mosaic Builder.

Given an image and a bounding box (from person_detector), produces a
cropped version centred on the main subject, with configurable padding.

Features:
  - Configurable padding (absolute pixels or percentage)
  - Validates crop won't cut off head/feet
  - Produces both a PIL Image object (for preview) and can save to disk
  - Preserves aspect ratio option for uniform mosaic cells
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from engine.models import BoundingBox, PersonDetection

logger = logging.getLogger("cropper")


def compute_crop_box(
    bbox: BoundingBox,
    img_w: int,
    img_h: int,
    padding_px: int = 40,
    target_aspect: Optional[float] = None,
) -> BoundingBox:
    """
    Compute the final crop bounding box given a detection bbox and padding.

    Steps:
    1. Apply padding symmetrically
    2. Clamp to image bounds
    3. Optionally expand to hit a target aspect ratio (without shrinking)

    target_aspect: width/height ratio to enforce. None = no enforcement.
    """
    # Step 1: pad
    x = max(0, bbox.x - padding_px)
    y = max(0, bbox.y - padding_px)
    x2 = min(img_w, bbox.x2 + padding_px)
    y2 = min(img_h, bbox.y2 + padding_px)
    w = x2 - x
    h = y2 - y

    # Step 2: enforce target aspect ratio (expand, not shrink)
    if target_aspect and target_aspect > 0 and w > 0 and h > 0:
        current_aspect = w / h
        if current_aspect < target_aspect:
            # Too tall — expand width
            new_w = int(h * target_aspect)
            diff = new_w - w
            x = max(0, x - diff // 2)
            x2 = min(img_w, x + new_w)
            w = x2 - x
        elif current_aspect > target_aspect:
            # Too wide — expand height
            new_h = int(w / target_aspect)
            diff = new_h - h
            y = max(0, y - diff // 2)
            y2 = min(img_h, y + new_h)
            h = y2 - y

    return BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h))


def crop_image(
    image_path: str,
    bbox: BoundingBox,
    padding_px: int = 40,
    target_aspect: Optional[float] = None,
    output_size: Optional[tuple[int, int]] = None,
) -> Optional[object]:  # returns PIL.Image.Image
    """
    Crop the image at image_path to the given bbox with padding.
    Returns a PIL Image (resized to output_size if specified), or None on error.
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        logger.error("[cropper] Pillow not installed")
        return None

    try:
        img = PILImage.open(image_path)
        img_w, img_h = img.size
    except Exception as e:
        logger.error(f"[cropper] Cannot open {image_path}: {e}")
        return None

    crop_box = compute_crop_box(bbox, img_w, img_h, padding_px, target_aspect)
    if not crop_box.is_valid():
        logger.warning(f"[cropper] Invalid crop box for {image_path}: {crop_box}")
        return None

    cropped = img.crop((
        crop_box.x, crop_box.y,
        crop_box.x2, crop_box.y2,
    ))

    if output_size:
        cropped = cropped.resize(output_size, PILImage.LANCZOS)

    return cropped


def save_crop(
    image_path: str,
    bbox: BoundingBox,
    output_path: str,
    padding_px: int = 40,
    target_aspect: Optional[float] = None,
    output_size: Optional[tuple[int, int]] = None,
    quality: int = 92,
) -> bool:
    """
    Crop and save to output_path. Returns True on success.
    """
    cropped = crop_image(image_path, bbox, padding_px, target_aspect, output_size)
    if cropped is None:
        return False

    try:
        ext = Path(output_path).suffix.lower()
        fmt = "JPEG" if ext in (".jpg", ".jpeg") else "PNG"
        save_kwargs = {"quality": quality} if fmt == "JPEG" else {}
        cropped.save(output_path, format=fmt, **save_kwargs)
        logger.debug(f"[cropper] Saved crop to {output_path}")
        return True
    except Exception as e:
        logger.error(f"[cropper] Cannot save crop: {e}")
        return False


def make_thumbnail(image_path: str, size: tuple[int, int] = (200, 200)) -> Optional[object]:
    """Generate a thumbnail PIL Image for the grid view."""
    try:
        from PIL import Image as PILImage
        img = PILImage.open(image_path)
        img.thumbnail(size, PILImage.LANCZOS)
        return img
    except Exception as e:
        logger.debug(f"[cropper] Thumbnail failed for {image_path}: {e}")
        return None


def make_thumbnail_bytes(image_path: str, size: tuple[int, int] = (200, 200)) -> bytes:
    """Return thumbnail as PNG bytes (for Qt pixmap loading)."""
    import io
    thumb = make_thumbnail(image_path, size)
    if thumb is None:
        return b""
    buf = io.BytesIO()
    try:
        thumb.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return b""
