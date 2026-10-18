"""Nodule region-of-interest (ROI) cropping, shared by data preparation and inference.

A model trained with `data.crop_to_bbox: true` has only seen nodule crops, so inference
must crop the uploaded scan the same way: pass the nodule box to `crop_square`.
"""
from __future__ import annotations

from PIL import Image

Box = tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax) in pixels


def crop_square(img: Image.Image, box: Box, margin: float) -> Image.Image:
    """Square crop centred on the box, `margin` x box size of context on each side, clipped to the image."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    right, bottom = min(img.width, int(cx + side / 2)), min(img.height, int(cy + side / 2))
    return img.crop((left, top, right, bottom))


def parse_box(text: str, width: int, height: int) -> Box:
    """Parse "xmin,ymin,xmax,ymax" and check it lies inside a width x height image."""
    try:
        x0, y0, x1, y1 = (int(round(float(v))) for v in text.split(","))
    except ValueError as e:
        raise ValueError('box must be "xmin,ymin,xmax,ymax" (pixels)') from e
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"box {x0},{y0},{x1},{y1} is not inside the {width}x{height} image")
    return x0, y0, x1, y1
