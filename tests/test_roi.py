"""Nodule ROI cropping shared by prepare.py and inference."""
import pytest
from PIL import Image

from thyrocheck.data.roi import crop_square, parse_box


def test_crop_square_adds_margin_and_is_square():
    crop = crop_square(Image.new("RGB", (400, 300)), (100, 100, 200, 200), margin=0.15)
    assert crop.size == (130, 130)


def test_crop_square_clips_to_image():
    crop = crop_square(Image.new("RGB", (400, 300)), (0, 0, 100, 100), margin=0.5)
    assert crop.size == (150, 150)


def test_parse_box():
    assert parse_box("10, 20, 110.4, 220", 400, 300) == (10, 20, 110, 220)
    for bad in ("1,2,3", "a,b,c,d", "50,20,10,220", "10,20,500,220"):
        with pytest.raises(ValueError):
            parse_box(bad, 400, 300)
